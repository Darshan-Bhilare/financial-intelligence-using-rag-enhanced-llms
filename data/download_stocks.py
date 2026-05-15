import yfinance as yf
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

tickers = {
    "TCS":      "TCS.NS",
    "INFY":     "INFY.NS",
    "ICICI":    "ICICIBANK.NS",
    "HDFC":     "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS"
}

for name, ticker in tickers.items():
    print(f"Downloading {name}...")
    df = yf.download(ticker, start="2022-01-01", end="2024-12-31", auto_adjust=True)
    
    # flatten multi-level columns
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # bring date from index
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)
    
    # keep only needed columns
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    
    df.to_csv(f"data/raw/{name}_stock.csv", index=False)
    print(f"  Saved: {len(df)} rows — sample: {df['date'].iloc[0]}")

print("\nAll stocks downloaded.")