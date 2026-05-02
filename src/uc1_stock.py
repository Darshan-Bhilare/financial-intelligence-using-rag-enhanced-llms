import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import retrieve, format_context
from llm_client  import predict_stock

# ── config ──────────────────────────────────────────────────────────────────
TICKERS   = ["TCS", "INFY", "ICICI", "HDFC", "RELIANCE"]
RAG_MODES = ["no_rag", "top_3", "top_5"]
RESULTS   = []

# ── load stock data ──────────────────────────────────────────────────────────
def load_stock(ticker):
    path = f"data/raw/{ticker}_stock.csv"
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.sort_values("date").reset_index(drop=True)
    return df

# ── compute ground truth label ───────────────────────────────────────────────
def get_label(df, window=5):
    """
    Ground truth: if avg close of next 5 days > today's close → UP else DOWN
    """
    labels = []
    for i in range(len(df)):
        if i + window >= len(df):
            labels.append(None)
        else:
            future_avg  = df["close"].iloc[i+1 : i+window+1].mean()
            today_close = df["close"].iloc[i]
            labels.append("UP" if future_avg > today_close else "DOWN")
    df["label"] = labels
    return df

# ── build query ──────────────────────────────────────────────────────────────
def build_query(ticker, row):
    return (
        f"{ticker} stock price on {row['date'].date()} "
        f"close={row['close']} open={row['open']} "
        f"high={row['high']} low={row['low']} volume={row['volume']}"
    )

# ── run one prediction ────────────────────────────────────────────────────────
def run_prediction(ticker, row, mode):
    query   = build_query(ticker, row)
    chunks  = retrieve(query, mode=mode)
    context = format_context(chunks)
    result  = predict_stock(context, ticker, query)

    prediction = result.get("prediction", "UNKNOWN").upper().strip()
    # normalize
    if prediction not in ["UP", "DOWN"]:
        prediction = "UNKNOWN"

    return {
        "ticker":           ticker,
        "date":             str(row["date"].date()),
        "mode":             mode,
        "ground_truth":     row["label"],
        "prediction":       prediction,
        "confidence":       result.get("confidence",       "N/A"),
        "hallucination_risk": result.get("hallucination_risk", "N/A"),
        "reasoning":        result.get("reasoning",        "N/A"),
        "correct":          prediction == row["label"],
    }

# ── evaluate one ticker across all modes ─────────────────────────────────────
def evaluate_ticker(ticker, sample_size=10):
    print(f"\n{'='*50}")
    print(f"  {ticker} — evaluating {sample_size} samples x 3 modes")
    print(f"{'='*50}")

    df = load_stock(ticker)
    df = get_label(df)
    df = df.dropna(subset=["label"])

    # sample evenly across the date range
    step    = max(1, len(df) // sample_size)
    samples = df.iloc[::step].head(sample_size)

    for _, row in samples.iterrows():
        for mode in RAG_MODES:
            print(f"  [{mode}] {row['date'].date()} | GT={row['label']}", end=" → ")
            result = run_prediction(ticker, row, mode)
            RESULTS.append(result)
            print(f"Pred={result['prediction']} | Correct={result['correct']}")

# ── compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(results_df, mode):
    subset = results_df[
        (results_df["mode"] == mode) &
        (results_df["prediction"].isin(["UP", "DOWN"]))
    ]
    if subset.empty:
        return {"mode": mode, "accuracy": 0, "total": 0, "correct": 0}

    correct  = subset["correct"].sum()
    total    = len(subset)
    accuracy = round(correct / total * 100, 2)

    return {
        "mode":     mode,
        "accuracy": accuracy,
        "correct":  int(correct),
        "total":    int(total),
    }

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   UC1 — Stock Direction Prediction")
    print("   RAG-Enhanced LLM vs No-RAG Baseline")
    print("=" * 55)

    # run evaluation for all tickers
    for ticker in TICKERS:
        evaluate_ticker(ticker, sample_size=10)

    # save raw results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(RESULTS)
    df.to_csv("results/uc1_stock_results.csv", index=False)
    print(f"\n[SAVED] results/uc1_stock_results.csv")

    # print ablation table
    print("\n" + "=" * 55)
    print("   UC1 ABLATION RESULTS")
    print("=" * 55)
    print(f"{'Mode':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)

    for mode in RAG_MODES:
        m = compute_metrics(df, mode)
        print(f"{m['mode']:<12} {m['accuracy']:>9}% {m['correct']:>10} {m['total']:>10}")

    print("=" * 55)
    print("\n[DONE] UC1 complete. Results saved to results/uc1_stock_results.csv")