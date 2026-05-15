import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import retrieve, format_context
from llm_client  import assess_credit_risk

# ── config ──────────────────────────────────────────────────────────────────
RAG_MODES = ["no_rag", "top_3", "top_5"]
RESULTS   = []

# ── synthetic Indian bank profiles ───────────────────────────────────────────
# Based on real RBI FSR 2023-24 reported ranges
BANK_PROFILES = [
    {"bank": "Bank A", "npa_ratio": "2.1%", "credit_growth": "14%", "capital_adequacy": "16.2%", "ground_truth": "LOW"},
    {"bank": "Bank B", "npa_ratio": "5.8%", "credit_growth": "8%",  "capital_adequacy": "13.1%", "ground_truth": "HIGH"},
    {"bank": "Bank C", "npa_ratio": "3.4%", "credit_growth": "11%", "capital_adequacy": "14.8%", "ground_truth": "MEDIUM"},
    {"bank": "Bank D", "npa_ratio": "7.2%", "credit_growth": "6%",  "capital_adequacy": "11.9%", "ground_truth": "HIGH"},
    {"bank": "Bank E", "npa_ratio": "1.8%", "credit_growth": "18%", "capital_adequacy": "17.5%", "ground_truth": "LOW"},
    {"bank": "Bank F", "npa_ratio": "4.9%", "credit_growth": "9%",  "capital_adequacy": "13.8%", "ground_truth": "MEDIUM"},
    {"bank": "Bank G", "npa_ratio": "6.5%", "credit_growth": "5%",  "capital_adequacy": "12.2%", "ground_truth": "HIGH"},
    {"bank": "Bank H", "npa_ratio": "2.8%", "credit_growth": "15%", "capital_adequacy": "15.9%", "ground_truth": "LOW"},
    {"bank": "Bank I", "npa_ratio": "3.9%", "credit_growth": "10%", "capital_adequacy": "14.2%", "ground_truth": "MEDIUM"},
    {"bank": "Bank J", "npa_ratio": "8.1%", "credit_growth": "4%",  "capital_adequacy": "11.2%", "ground_truth": "HIGH"},
]

# ── ground truth logic ────────────────────────────────────────────────────────
def get_ground_truth(npa_ratio):
    """
    Rule-based ground truth aligned with RBI NPA thresholds:
    < 3%   → LOW
    3-6%   → MEDIUM
    > 6%   → HIGH
    """
    npa = float(npa_ratio.replace("%", ""))
    if npa < 3.0:
        return "LOW"
    elif npa <= 6.0:
        return "MEDIUM"
    else:
        return "HIGH"

# ── run one assessment ────────────────────────────────────────────────────────
def run_assessment(bank, mode):
    query   = (
        f"NPA risk assessment for Indian bank with "
        f"NPA ratio {bank['npa_ratio']}, "
        f"credit growth {bank['credit_growth']}, "
        f"capital adequacy {bank['capital_adequacy']}"
    )
    chunks  = retrieve(query, mode=mode)
    context = format_context(chunks)
    result  = assess_credit_risk(context, bank, query)

    prediction = result.get("risk_level", "UNKNOWN").upper().strip()
    if prediction not in ["LOW", "MEDIUM", "HIGH"]:
        prediction = "UNKNOWN"

    ground_truth = get_ground_truth(bank["npa_ratio"])

    return {
        "bank":               bank["bank"],
        "npa_ratio":          bank["npa_ratio"],
        "credit_growth":      bank["credit_growth"],
        "capital_adequacy":   bank["capital_adequacy"],
        "mode":               mode,
        "ground_truth":       ground_truth,
        "prediction":         prediction,
        "npa_assessment":     result.get("npa_assessment",      "N/A"),
        "rbi_policy_ref":     result.get("rbi_policy_reference","N/A"),
        "hallucination_risk": result.get("hallucination_risk",  "N/A"),
        "reasoning":          result.get("reasoning",           "N/A"),
        "correct":            prediction == ground_truth,
    }

# ── compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(results_df, mode):
    subset = results_df[
        (results_df["mode"] == mode) &
        (results_df["prediction"].isin(["LOW", "MEDIUM", "HIGH"]))
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
    print("   UC2 — Credit Risk / NPA Assessment")
    print("   RAG-Enhanced LLM vs No-RAG Baseline")
    print("=" * 55)

    for bank in BANK_PROFILES:
        print(f"\n  Bank: {bank['bank']} | NPA={bank['npa_ratio']}")
        for mode in RAG_MODES:
            print(f"  [{mode}]", end=" → ")
            result = run_assessment(bank, mode)
            RESULTS.append(result)
            print(
                f"GT={result['ground_truth']} | "
                f"Pred={result['prediction']} | "
                f"Correct={result['correct']}"
            )

    # save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(RESULTS)
    df.to_csv("results/uc2_credit_results.csv", index=False)
    print(f"\n[SAVED] results/uc2_credit_results.csv")

    # ablation table
    print("\n" + "=" * 55)
    print("   UC2 ABLATION RESULTS")
    print("=" * 55)
    print(f"{'Mode':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)

    for mode in RAG_MODES:
        m = compute_metrics(df, mode)
        print(f"{m['mode']:<12} {m['accuracy']:>9}% {m['correct']:>10} {m['total']:>10}")

    print("=" * 55)
    print("\n[DONE] UC2 complete. Results saved to results/uc2_credit_results.csv")