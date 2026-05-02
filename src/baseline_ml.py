import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
RESULTS_DIR = "results"
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# UC1 — Stock Direction Prediction (XGBoost baseline)
# ══════════════════════════════════════════════════════════════════════════════
def baseline_uc1():
    print("\n" + "="*55)
    print("  UC1 Baseline — Stock Direction (XGBoost)")
    print("="*55)

    tickers = ["TCS", "INFY", "ICICI", "HDFC", "RELIANCE"]
    all_results = []

    for ticker in tickers:
        path = f"data/raw/{ticker}_stock.csv"
        if not os.path.exists(path):
            print(f"[SKIP] {path}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna()

        # features: OHLCV + rolling averages
        df["ma5"]    = df["close"].rolling(5).mean()
        df["ma10"]   = df["close"].rolling(10).mean()
        df["return"] = df["close"].pct_change()
        df["hl_gap"] = df["high"] - df["low"]
        df           = df.dropna()

        # label: UP if next close > current close
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna()

        features = ["open", "high", "low", "close", "volume",
                    "ma5", "ma10", "return", "hl_gap"]
        X = df[features]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
        )

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        f1  = round(f1_score(y_test, preds, average="weighted") * 100, 2)

        print(f"  {ticker:<12} Accuracy={acc}%  F1={f1}%")
        all_results.append({"ticker": ticker, "accuracy": acc, "f1": f1})

    df_res = pd.DataFrame(all_results)
    avg_acc = round(df_res["accuracy"].mean(), 2)
    avg_f1  = round(df_res["f1"].mean(), 2)
    print(f"\n  {'Average':<12} Accuracy={avg_acc}%  F1={avg_f1}%")

    df_res.to_csv(os.path.join(RESULTS_DIR, "baseline_uc1_stock.csv"), index=False)
    print(f"[SAVED] results/baseline_uc1_stock.csv")
    return avg_acc, avg_f1


# ══════════════════════════════════════════════════════════════════════════════
# UC2 — Credit Risk (XGBoost baseline)
# ══════════════════════════════════════════════════════════════════════════════
def baseline_uc2():
    print("\n" + "="*55)
    print("  UC2 Baseline — Credit Risk (XGBoost)")
    print("="*55)

    # synthetic dataset matching our bank profiles
    data = {
        "npa_ratio":        [2.1, 5.8, 3.4, 7.2, 1.8, 4.9, 6.5, 2.8, 3.9, 8.1],
        "credit_growth":    [14,  8,   11,  6,   18,  9,   5,   15,  10,  4  ],
        "capital_adequacy": [16.2,13.1,14.8,11.9,17.5,13.8,12.2,15.9,14.2,11.2],
        "label":            [0,   2,   1,   2,   0,   1,   2,   0,   1,   2  ]
        # 0=LOW, 1=MEDIUM, 2=HIGH
    }
    df = pd.DataFrame(data)

    # augment with noise for more training samples
    augmented = []
    for _ in range(20):
        noise = df.copy()
        noise["npa_ratio"]        += np.random.normal(0, 0.3, len(df))
        noise["credit_growth"]    += np.random.normal(0, 0.5, len(df))
        noise["capital_adequacy"] += np.random.normal(0, 0.2, len(df))
        augmented.append(noise)
    df_aug = pd.concat([df] + augmented, ignore_index=True)

    X = df_aug[["npa_ratio", "credit_growth", "capital_adequacy"]]
    y = df_aug["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = round(accuracy_score(y_test, preds) * 100, 2)
    f1  = round(f1_score(y_test, preds, average="weighted") * 100, 2)

    print(f"  Accuracy={acc}%  F1={f1}%")
    print(f"\n{classification_report(y_test, preds, target_names=['LOW','MEDIUM','HIGH'])}")

    pd.DataFrame([{"accuracy": acc, "f1": f1}]).to_csv(
        os.path.join(RESULTS_DIR, "baseline_uc2_credit.csv"), index=False
    )
    print(f"[SAVED] results/baseline_uc2_credit.csv")
    return acc, f1


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Loan Default (XGBoost baseline)
# ══════════════════════════════════════════════════════════════════════════════
def baseline_uc3():
    print("\n" + "="*55)
    print("  UC3 Baseline — Loan Default (XGBoost)")
    print("="*55)

    path = "data/raw/loan_default.csv"
    if not os.path.exists(path):
        print(f"[SKIP] {path}")
        return 0, 0

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "seriousdlqin2yrs":                    "label",
        "revolvingutilizationofunsecuredlines": "revolving_util",
        "numberoftime30-59dayspastduenotworse": "past_due_30_59",
        "debtratio":                            "debt_ratio",
        "monthlyincome":                        "monthly_income",
        "numberofopencreditlinesandloans":      "open_credit_lines",
        "numberoftimes90dayslate":              "times_90_late",
    })

    features = ["revolving_util", "age", "past_due_30_59",
                "debt_ratio", "monthly_income",
                "open_credit_lines", "times_90_late"]

    df = df[features + ["label"]].dropna().head(10000)

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        scale_pos_weight=10,  # handle class imbalance
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds     = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]

    acc    = round(accuracy_score(y_test, preds) * 100, 2)
    f1     = round(f1_score(y_test, preds, average="weighted") * 100, 2)
    auc    = round(roc_auc_score(y_test, preds_prob) * 100, 2)

    print(f"  Accuracy={acc}%  F1={f1}%  AUC-ROC={auc}%")
    print(f"\n{classification_report(y_test, preds, target_names=['No Default','Default'])}")

    pd.DataFrame([{"accuracy": acc, "f1": f1, "auc_roc": auc}]).to_csv(
        os.path.join(RESULTS_DIR, "baseline_uc3_loan.csv"), index=False
    )
    print(f"[SAVED] results/baseline_uc3_loan.csv")
    return acc, f1


# ══════════════════════════════════════════════════════════════════════════════
# Final comparison table
# ══════════════════════════════════════════════════════════════════════════════
def print_comparison(uc1, uc2, uc3):
    # RAG-LLM best results from ablation
    llm_results = {
        "UC1 Stock":  58.0,   # no_rag best
        "UC2 Credit": 66.67,  # top_3 best
        "UC3 Loan":   60.0,   # no_rag best
    }
    ml_results = {
        "UC1 Stock":  uc1[0],
        "UC2 Credit": uc2[0],
        "UC3 Loan":   uc3[0],
    }

    print("\n" + "="*65)
    print("   FINAL COMPARISON — RAG-LLM vs XGBoost Baseline")
    print("="*65)
    print(f"{'Use Case':<15} {'RAG-LLM (best)':>16} {'XGBoost':>10} {'Winner':>10}")
    print("-"*55)

    for uc in llm_results:
        llm = llm_results[uc]
        ml  = ml_results[uc]
        winner = "RAG-LLM" if llm >= ml else "XGBoost"
        print(f"{uc:<15} {str(llm)+'%':>16} {str(ml)+'%':>10} {winner:>10}")

    print("="*65)

    # save comparison
    rows = []
    for uc in llm_results:
        rows.append({
            "use_case":    uc,
            "rag_llm_acc": llm_results[uc],
            "xgboost_acc": ml_results[uc],
            "winner":      "RAG-LLM" if llm_results[uc] >= ml_results[uc] else "XGBoost",
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "final_comparison.csv"), index=False
    )
    print(f"[SAVED] results/final_comparison.csv")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   FinRAG — XGBoost Baseline Comparison")
    print("=" * 55)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    uc1 = baseline_uc1()
    uc2 = baseline_uc2()
    uc3 = baseline_uc3()

    print_comparison(uc1, uc2, uc3)

    print("\n[DONE] Baseline comparison complete.")