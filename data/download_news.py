from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading Indian Financial News dataset...")
dataset = load_dataset("kdave/Indian_Financial_News", split="train")
df = dataset.to_pandas()
df.to_csv("data/raw/indian_financial_news.csv", index=False)
print(f"Saved: data/raw/indian_financial_news.csv — {len(df)} rows")
print(df.head(3))