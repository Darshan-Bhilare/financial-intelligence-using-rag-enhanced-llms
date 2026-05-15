import os
import pandas as pd
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions

# ── config ─────────────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
CHROMA_DIR     = "chroma_db"
CHUNK_SIZE     = 300
NEWS_LIMIT     = 3000
LOAN_LIMIT     = 5000
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ── chromadb ────────────────────────────────────────────────────────────────
def get_collection(name="finrag"):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name=EMBED_MODEL
             )
    return client.get_or_create_collection(name=name, embedding_function=ef)

# ── chunking ────────────────────────────────────────────────────────────────
def chunk_text(text, source):
    words  = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 50:
            chunks.append({"text": chunk, "source": source})
    return chunks

# ── helper: batch add to chromadb ───────────────────────────────────────────
def add_to_collection(collection, chunks, prefix):
    if not chunks:
        print(f"  [WARN] No chunks to add for prefix '{prefix}'")
        return
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=[c["text"]           for c in batch],
            metadatas =[{"source": c["source"]} for c in batch],
            ids       =[f"{prefix}_{i + j}" for j, _ in enumerate(batch)],
        )

# ── ingest: financial news ──────────────────────────────────────────────────
def ingest_news(collection):
    path = os.path.join(RAW_DIR, "indian_financial_news.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return

    df       = pd.read_csv(path).head(NEWS_LIMIT)
    text_col = next(
        (c for c in df.columns if c.lower() in ["text", "content", "article", "body"]),
        df.columns[0]
    )
    print(f"[NEWS] Column='{text_col}' · Rows={len(df)}")

    chunks = []
    for idx, row in df.iterrows():
        chunks.extend(chunk_text(str(row[text_col]), source="indian_financial_news"))
        if idx % 500 == 0:
            print(f"  → article {idx}/{len(df)} | chunks so far: {len(chunks)}")

    add_to_collection(collection, chunks, prefix="news")
    print(f"[NEWS] Done — {len(chunks)} chunks ingested\n")

# ── ingest: RBI PDFs ────────────────────────────────────────────────────────
def ingest_pdfs(collection):
    pdfs = {
        "RBI_FSR_Dec2023.pdf": "rbi_fsr_dec2023",
        "RBI_FSR_Jun2024.pdf": "rbi_fsr_jun2024",
    }
    for filename, source in pdfs.items():
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue

        text = ""
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
                if i % 20 == 0:
                    print(f"  → page {i}/{len(pdf.pages)}")

        chunks = chunk_text(text, source=source)
        add_to_collection(collection, chunks, prefix=source)
        print(f"[PDF] {filename} — {len(chunks)} chunks ingested\n")

# ── ingest: stock OHLCV ─────────────────────────────────────────────────────
def ingest_stocks(collection):
    tickers = ["TCS", "INFY", "ICICI", "HDFC", "RELIANCE"]
    chunks  = []

    for ticker in tickers:
        path = os.path.join(RAW_DIR, f"{ticker}_stock.csv")
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        # ensure date column exists
        if "date" not in df.columns:
            print(f"  [WARN] No date column found in {ticker}, columns: {list(df.columns)}")
            continue

        ticker_chunks = 0
        for _, row in df.iterrows():
            try:
                summary = (
                    f"{ticker} stock on {row['date']}: "
                    f"Open={round(float(row.get('open',   0)), 2)}, "
                    f"High={round(float(row.get('high',   0)), 2)}, "
                    f"Low={round(float(row.get('low',     0)), 2)}, "
                    f"Close={round(float(row.get('close', 0)), 2)}, "
                    f"Volume={int(float(row.get('volume', 0)))}."
                )
                chunks.append({"text": summary, "source": f"{ticker}_stock"})
                ticker_chunks += 1
            except Exception as e:
                continue

        print(f"  → {ticker}: {ticker_chunks} rows | sample: {df['date'].iloc[0]}")

    add_to_collection(collection, chunks, prefix="stock")
    print(f"[STOCK] Done — {len(chunks)} OHLCV summaries ingested\n")

# ── ingest: loan default ────────────────────────────────────────────────────
def ingest_loans(collection):
    path = os.path.join(RAW_DIR, "loan_default.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return

    df = pd.read_csv(path).head(LOAN_LIMIT)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[LOAN] Rows={len(df)}")

    chunks = []
    for idx, row in df.iterrows():
        try:
            summary = (
                f"Loan applicant profile: "
                f"revolving utilization={row.get('revolvingutilizationofunsecuredlines', '?')}, "
                f"age={row.get('age', '?')}, "
                f"past due 30-59 days={row.get('numberoftime30-59dayspastduenotworse', '?')}, "
                f"debt ratio={row.get('debtratio', '?')}, "
                f"monthly income={row.get('monthlyincome', '?')}, "
                f"open credit lines={row.get('numberofopencreditlinesandloans', '?')}, "
                f"times 90 days late={row.get('numberoftimes90dayslate', '?')}, "
                f"default label={row.get('seriousdlqin2yrs', '?')}."
            )
            chunks.append({"text": summary, "source": "loan_default"})
        except Exception:
            continue
        if idx % 1000 == 0:
            print(f"  → loan {idx}/{len(df)}")

    add_to_collection(collection, chunks, prefix="loan")
    print(f"[LOAN] Done — {len(chunks)} loan summaries ingested\n")

# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FinRAG Data Pipeline")
    print("=" * 50 + "\n")

    collection = get_collection()

    ingest_news(collection)
    ingest_pdfs(collection)
    ingest_stocks(collection)
    ingest_loans(collection)

    print("=" * 50)
    print(f"  DONE — Total docs in ChromaDB: {collection.count()}")
    print("=" * 50)