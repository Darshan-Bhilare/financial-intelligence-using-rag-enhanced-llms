import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import retrieve, format_context
from llm_client  import assess_loan_default

# ── config ──────────────────────────────────────────────────────────────────
RAG_MODES = ["no_rag", "top_3", "top_5"]
RESULTS   = []

# ── load loan dataset ─────────────────────────────────────────────────────────
def load_loan_data(sample_size=10):
    df = pd.read_csv("data/raw/loan_default.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    # rename columns for clarity
    df = df.rename(columns={
        "seriousdlqin2yrs":                          "default_label",
        "revolvingutilizationofunsecuredlines":       "revolving_util",
        "numberoftime30-59dayspastduenotworse":       "past_due_30_59",
        "debtratio":                                  "debt_ratio",
        "monthlyincome":                              "monthly_income",
        "numberofopencreditlinesandloans":            "open_credit_lines",
        "numberoftimes90dayslate":                    "times_90_late",
        "numberofreal estateloansorlines":            "real_estate_loans",
        "numberoftime60-89dayspastduenotworse":       "past_due_60_89",
        "numberofdependents":                         "dependents",
    })

    df = df.dropna(subset=["default_label", "debt_ratio", "monthly_income"])
    df["ground_truth"] = df["default_label"].apply(
        lambda x: "HIGH" if x == 1 else "LOW"
    )

    # sample balanced: half defaulters half non-defaulters
    defaulters     = df[df["default_label"] == 1].head(sample_size // 2)
    non_defaulters = df[df["default_label"] == 0].head(sample_size // 2)
    return pd.concat([defaulters, non_defaulters]).reset_index(drop=True)

# ── run one assessment ────────────────────────────────────────────────────────
def run_assessment(row, mode):
    applicant = {
        "age":                  int(row.get("age",               0)),
        "monthly_income":       float(row.get("monthly_income",  0)),
        "debt_ratio":           float(row.get("debt_ratio",      0)),
        "revolving_util":       float(row.get("revolving_util",  0)),
        "times_90_late":        int(row.get("times_90_late",     0)),
        "open_credit_lines":    int(row.get("open_credit_lines", 0)),
        "past_due_30_59":       int(row.get("past_due_30_59",    0)),
    }

    query = (
        f"Loan default risk for applicant: "
        f"age={applicant['age']}, "
        f"income={applicant['monthly_income']}, "
        f"debt_ratio={applicant['debt_ratio']}, "
        f"revolving_util={applicant['revolving_util']}, "
        f"times_90_late={applicant['times_90_late']}"
    )

    chunks  = retrieve(query, mode=mode)
    context = format_context(chunks)
    result  = assess_loan_default(context, applicant, query)

    raw_pred   = result.get("default_risk", "UNKNOWN").upper().strip()
    prediction = raw_pred if raw_pred in ["HIGH", "MEDIUM", "LOW"] else "UNKNOWN"

    # align: HIGH = default, LOW/MEDIUM = no default
    pred_binary = "HIGH" if prediction == "HIGH" else "LOW"
    ground_truth = row["ground_truth"]

    return {
        "age":                int(applicant["age"]),
        "monthly_income":     applicant["monthly_income"],
        "debt_ratio":         applicant["debt_ratio"],
        "revolving_util":     applicant["revolving_util"],
        "times_90_late":      applicant["times_90_late"],
        "mode":               mode,
        "ground_truth":       ground_truth,
        "prediction":         prediction,
        "default_probability":result.get("default_probability", "N/A"),
        "key_risk_factors":   str(result.get("key_risk_factors", [])),
        "hallucination_risk": result.get("hallucination_risk",   "N/A"),
        "reasoning":          result.get("reasoning",            "N/A"),
        "correct":            pred_binary == ground_truth,
    }

# ── compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(results_df, mode):
    subset = results_df[results_df["mode"] == mode]
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
    print("   UC3 — Loan Default Risk Assessment")
    print("   RAG-Enhanced LLM vs No-RAG Baseline")
    print("=" * 55)

    samples = load_loan_data(sample_size=10)
    print(f"\n  Loaded {len(samples)} applicants "
          f"({samples['default_label'].sum()} defaulters, "
          f"{(samples['default_label']==0).sum()} non-defaulters)\n")

    for idx, row in samples.iterrows():
        print(f"  Applicant {idx+1} | Age={row['age']} | "
              f"Income={row['monthly_income']} | GT={row['ground_truth']}")
        for mode in RAG_MODES:
            print(f"  [{mode}]", end=" → ")
            result = run_assessment(row, mode)
            RESULTS.append(result)
            print(
                f"Pred={result['prediction']} | "
                f"Prob={result['default_probability']} | "
                f"Correct={result['correct']}"
            )

    # save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(RESULTS)
    df.to_csv("results/uc3_loan_results.csv", index=False)
    print(f"\n[SAVED] results/uc3_loan_results.csv")

    # ablation table
    print("\n" + "=" * 55)
    print("   UC3 ABLATION RESULTS")
    print("=" * 55)
    print(f"{'Mode':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)

    for mode in RAG_MODES:
        m = compute_metrics(df, mode)
        print(f"{m['mode']:<12} {m['accuracy']:>9}% {m['correct']:>10} {m['total']:>10}")

    print("=" * 55)
    print("\n[DONE] UC3 complete. Results saved to results/uc3_loan_results.csv")