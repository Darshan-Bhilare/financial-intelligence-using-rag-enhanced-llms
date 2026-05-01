import yfinance as yf
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

tickers = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "ICICI": "ICICIBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS"
}

for name, ticker in tickers.items():
    print(f"Downloading {name}...")
    df = yf.download(ticker, start="2022-01-01", end="2024-12-31", auto_adjust=True)
    df.to_csv(f"data/raw/{name}_stock.csv")
    print(f"  Saved: data/raw/{name}_stock.csv — {len(df)} rows")

print("\nAll stocks downloaded.")