import pandas as pd
import os

files = {
    "TCS stock":      "data/raw/TCS_stock.csv",
    "INFY stock":     "data/raw/INFY_stock.csv",
    "ICICI stock":    "data/raw/ICICI_stock.csv",
    "HDFC stock":     "data/raw/HDFC_stock.csv",
    "Reliance stock": "data/raw/RELIANCE_stock.csv",
    "Financial news": "data/raw/indian_financial_news.csv",
    "Loan default":   "data/raw/loan_default.csv",
}

print("=== Data Verification ===\n")
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[OK] {name}: {len(df):,} rows x {len(df.columns)} cols")
    else:
        print(f"[MISSING] {name}: {path}")

for pdf in ["data/raw/RBI_FSR_Dec2023.pdf", "data/raw/RBI_FSR_Jun2024.pdf"]:
    if os.path.exists(pdf):
        size_kb = os.path.getsize(pdf) // 1024
        print(f"[OK] {os.path.basename(pdf)}: {size_kb} KB")
    else:
        print(f"[MISSING] {pdf}")