import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

# ── config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
RAG_MODES   = ["no_rag", "top_3", "top_5"]

# ── load results ─────────────────────────────────────────────────────────────
def load_results():
    files = {
        "UC1 Stock":  "uc1_stock_results.csv",
        "UC2 Credit": "uc2_credit_results.csv",
        "UC3 Loan":   "uc3_loan_results.csv",
    }
    dfs = {}
    for name, filename in files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"[OK] Loaded {name} — {len(dfs[name])} rows")
        else:
            print(f"[MISSING] {path}")
    return dfs

# ── compute accuracy per mode ─────────────────────────────────────────────────
def compute_accuracy(df, use_case):
    rows = []
    for mode in RAG_MODES:
        subset = df[df["mode"] == mode]
        if subset.empty:
            continue
        # filter out unknowns
        subset = subset[subset["prediction"].str.upper() != "UNKNOWN"]
        correct  = subset["correct"].sum()
        total    = len(subset)
        accuracy = round(correct / total * 100, 2) if total > 0 else 0
        rows.append({
            "use_case": use_case,
            "mode":     mode,
            "accuracy": accuracy,
            "correct":  int(correct),
            "total":    int(total),
        })
    return rows

# ── compute hallucination rate ────────────────────────────────────────────────
def compute_hallucination(df, use_case):
    rows = []
    for mode in RAG_MODES:
        subset = df[df["mode"] == mode]
        if subset.empty:
            continue
        if "hallucination_risk" not in subset.columns:
            continue
        high_risk = subset[
            subset["hallucination_risk"].str.upper() == "HIGH"
        ]
        rate = round(len(high_risk) / len(subset) * 100, 2)
        rows.append({
            "use_case":          use_case,
            "mode":              mode,
            "hallucination_rate": rate,
            "high_risk_count":   len(high_risk),
            "total":             len(subset),
        })
    return rows

# ── print ablation table ──────────────────────────────────────────────────────
def print_ablation_table(accuracy_rows):
    df = pd.DataFrame(accuracy_rows)

    print("\n" + "=" * 65)
    print("   FULL ABLATION TABLE — Accuracy by Use Case and RAG Mode")
    print("=" * 65)
    print(f"{'Use Case':<15} {'no_rag':>10} {'top_3':>10} {'top_5':>10}")
    print("-" * 50)

    for uc in df["use_case"].unique():
        subset = df[df["use_case"] == uc]
        row    = {}
        for _, r in subset.iterrows():
            row[r["mode"]] = f"{r['accuracy']}%"
        print(
            f"{uc:<15} "
            f"{row.get('no_rag', 'N/A'):>10} "
            f"{row.get('top_3',  'N/A'):>10} "
            f"{row.get('top_5',  'N/A'):>10}"
        )

    print("-" * 50)

    # average across use cases
    avg = df.groupby("mode")["accuracy"].mean().round(2)
    print(
        f"{'Average':<15} "
        f"{str(avg.get('no_rag', 0))+'%':>10} "
        f"{str(avg.get('top_3',  0))+'%':>10} "
        f"{str(avg.get('top_5',  0))+'%':>10}"
    )
    print("=" * 65)

# ── print hallucination table ─────────────────────────────────────────────────
def print_hallucination_table(hall_rows):
    df = pd.DataFrame(hall_rows)

    print("\n" + "=" * 65)
    print("   HALLUCINATION RATE by Use Case and RAG Mode")
    print("=" * 65)
    print(f"{'Use Case':<15} {'no_rag':>10} {'top_3':>10} {'top_5':>10}")
    print("-" * 50)

    for uc in df["use_case"].unique():
        subset = df[df["use_case"] == uc]
        row    = {}
        for _, r in subset.iterrows():
            row[r["mode"]] = f"{r['hallucination_rate']}%"
        print(
            f"{uc:<15} "
            f"{row.get('no_rag', 'N/A'):>10} "
            f"{row.get('top_3',  'N/A'):>10} "
            f"{row.get('top_5',  'N/A'):>10}"
        )
    print("=" * 65)

# ── plot ablation chart ───────────────────────────────────────────────────────
def plot_ablation(accuracy_rows):
    df  = pd.DataFrame(accuracy_rows)
    ucs = df["use_case"].unique()

    x      = range(len(ucs))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, mode in enumerate(RAG_MODES):
        subset = df[df["mode"] == mode]
        vals   = [
            subset[subset["use_case"] == uc]["accuracy"].values[0]
            if not subset[subset["use_case"] == uc].empty else 0
            for uc in ucs
        ]
        bars = ax.bar(
            [xi + i * width for xi in x],
            vals,
            width,
            label=mode,
            color=colors[i],
            alpha=0.85,
        )
        # add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Use Case",    fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "FinRAG Ablation Study — Accuracy by RAG Mode\n"
        "Financial Intelligence using RAG-Enhanced LLMs",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(ucs, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(title="RAG Mode", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ablation_chart.png")
    plt.savefig(path, dpi=150)
    print(f"\n[SAVED] {path}")
    plt.close()

# ── save combined CSV ─────────────────────────────────────────────────────────
def save_combined(accuracy_rows, hall_rows):
    acc_df  = pd.DataFrame(accuracy_rows)
    hall_df = pd.DataFrame(hall_rows)

    acc_df.to_csv(os.path.join(RESULTS_DIR, "ablation_accuracy.csv"),      index=False)
    hall_df.to_csv(os.path.join(RESULTS_DIR, "ablation_hallucination.csv"), index=False)
    print(f"[SAVED] results/ablation_accuracy.csv")
    print(f"[SAVED] results/ablation_hallucination.csv")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   FinRAG — Full Ablation Study")
    print("=" * 55 + "\n")

    dfs          = load_results()
    accuracy_rows = []
    hall_rows     = []

    for use_case, df in dfs.items():
        accuracy_rows.extend(compute_accuracy(df,      use_case))
        hall_rows.extend(compute_hallucination(df, use_case))

    print_ablation_table(accuracy_rows)
    print_hallucination_table(hall_rows)
    plot_ablation(accuracy_rows)
    save_combined(accuracy_rows, hall_rows)

    print("\n[DONE] Ablation study complete.")