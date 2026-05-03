# TriFin: Financial Intelligence using RAG-Enhanced LLMs
### A Unified Explainable GenAI System for Indian Banking & Finance
> CS5202 — GenAI & LLM · Domain: Banking & Finance
---

## What Is This Project?

**Financial Intelligence using RAG-Enhanced LLMs** is a GenAI-first financial intelligence system that grounds LLaMA3 predictions in verified Indian financial documents using Retrieval-Augmented Generation (RAG). Unlike traditional ML models that output a number with no explanation, TriFin produces **cited, explainable predictions** for three high-stakes Indian finance use cases.

| Use Case | Task | Data Source |
|---|---|---|
| Use Case 1 — Stock Prediction | Predict UP/DOWN direction for 5 NSE stocks | yFinance + Indian Financial News |
| Use Case 2 — Credit Risk / NPA | Assess bank NPA risk level (LOW/MEDIUM/HIGH) | RBI Financial Stability Reports |
| Use Case 3 — Loan Default | Predict loan default risk with justification | Kaggle Give Me Some Credit |

### Why RAG instead of plain ML?

- XGBoost predicts a number. FinRAG explains **why**, citing the source.
- Generic LLMs hallucinate. FinRAG is **grounded** in verified documents.
- Regulators need auditability. FinRAG produces **RBI-cited reasoning** for every prediction.

---

## System Architecture

```
Data Sources
├── Indian Financial News (HuggingFace — 26k articles)
├── RBI FSR PDFs (Dec 2023 + Jun 2024)
├── NIFTY-100 OHLCV (yFinance — 5 stocks, 738 days each)
└── Give Me Some Credit (Kaggle — 150k loan records)
         │
         ▼
Data Pipeline (src/data_pipeline.py)
   Chunk (300 words) → Embed (all-MiniLM-L6-v2) → Store
         │
         ▼
ChromaDB Vector Store (16,508 documents)
         │
         ▼
RAG Retriever (src/rag_pipeline.py)
   no_rag (k=0) | top_3 (k=3) | top_5 (k=5)  ← ablation variable
         │
         ▼
LLaMA3 8B via Ollama (src/llm_client.py)
   Grounded prompt → JSON output with source citations
         │
    ┌────┴────┬────┐
   UC1       UC2  UC3
 Stock    Credit  Loan
Predict    Risk  Default
         │
         ▼
Ablation Study + XGBoost Baseline
```

---

## Key Results

### Ablation Study — RAG Mode vs Accuracy

| Use Case | no_rag | top_3 | top_5 |
|---|---|---|---|
| Use Case 1 — Stock | **58.0%** | 52.0% | 44.0% |
| Use Case 2 — Credit Risk | 50.0% | **66.67%** | 60.0% |
| Use Case 3 — Loan Default | **60.0%** | 50.0% | 50.0% |
| Average | 56.0% | 56.22% | 51.33% |

Hallucination rate = **0%** across all 210 LLM calls

### RAG-LLM vs XGBoost Baseline

| Use Case | RAG-LLM Best | XGBoost | Winner |
|---|---|---|---|
| UC1 — Stock | 58.0% | 51.51% | **RAG-LLM (+6.5%)** |
| UC2 — Credit | 66.67% | 92.86% | XGBoost |
| UC3 — Loan | 60.0% | 85.2% | XGBoost |

> **Key insight**: RAG-LLM wins on news-driven tasks. XGBoost wins on structured tabular data. But only RAG-LLM can cite RBI policy and explain its reasoning — critical for regulated banking environments.

---

## Repository Structure

```
project-finrag-<rollno>/
│
├── README.md                     # This file
├── domain_note.pdf               # 1-page domain note (Milestone 1)
├── report.pdf                    # Final 4-6 page report (Milestone 2)
├── requirements.txt              # All Python dependencies (pinned)
│
├── data/
│   ├── raw/                      # Downloaded datasets (not committed)
│   │   ├── TCS_stock.csv
│   │   ├── INFY_stock.csv
│   │   ├── ICICI_stock.csv
│   │   ├── HDFC_stock.csv
│   │   ├── RELIANCE_stock.csv
│   │   ├── indian_financial_news.csv
│   │   ├── RBI_FSR_Dec2023.pdf
│   │   ├── RBI_FSR_Jun2024.pdf
│   │   └── loan_default.csv
│   ├── download_stocks.py        # yFinance download script
│   ├── download_news.py          # HuggingFace download script
│   └── verify_data.py            # Dataset verification script
│
├── src/
│   ├── data_pipeline.py          # Chunk + embed + ingest into ChromaDB
│   ├── rag_pipeline.py           # RAG retriever (no_rag / top_3 / top_5)
│   ├── llm_client.py             # Ollama LLaMA3 grounded prompt engine
│   ├── uc1_stock.py              # UC1: Stock direction prediction
│   ├── uc2_credit.py             # UC2: Credit risk / NPA assessment
│   ├── uc3_loan.py               # UC3: Loan default risk prediction
│   ├── ablation.py               # Full ablation study + chart
│   ├── baseline_ml.py            # XGBoost baseline comparison
│   └── app.py                    # Streamlit web UI (live demo)
│
├── notebooks/                    # Exploratory notebooks
│
└── results/
    ├── uc1_stock_results.csv
    ├── uc2_credit_results.csv
    ├── uc3_loan_results.csv
    ├── ablation_accuracy.csv
    ├── ablation_hallucination.csv
    ├── ablation_chart.png
    ├── baseline_uc1_stock.csv
    ├── baseline_uc2_credit.csv
    ├── baseline_uc3_loan.csv
    └── final_comparison.csv
```

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Ollama** — [ollama.com/download](https://ollama.com/download)
- **Git** — [git-scm.com](https://git-scm.com/)
- **8GB+ RAM** recommended for LLaMA3 8B inference
- **5GB free disk** for model + ChromaDB + datasets

---

## Steps to Reproduce

### Step 1 — Clone the repository

```bash
git clone https://github.com/Darshan-Bhilare/financial-intelligence-using-rag-enhanced-llms.git
cd financial-intelligence-using-rag-enhanced-llms
```

---

### Step 2 — Set up Python virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### Step 3 — Install Ollama and pull LLaMA3

```bash
# Download and install Ollama from https://ollama.com/download
# Then pull the LLaMA3 model (4GB download — do this once)
ollama pull llama3

# Verify it works
ollama run llama3 "Say hello"
```

> Ollama runs as a background service automatically. No manual startup needed.

---

### Step 4 — Download all datasets

**4a. Stock price data (automated)**
```bash
python data/download_stocks.py
```
Downloads TCS, INFY, ICICI, HDFC, RELIANCE from yFinance (2022–2024).

**4b. Indian Financial News (automated)**
```bash
python data/download_news.py
```
Downloads 26,961 articles from HuggingFace — `kdave/Indian_Financial_News`.

**4c. RBI Financial Stability Reports (manual)**

Download these two PDFs and save them to `data/raw/`:

| File | URL |
|---|---|
| `RBI_FSR_Dec2023.pdf` | https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=22075 |
| `RBI_FSR_Jun2024.pdf` | https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=22359 |

**4d. Loan default dataset (manual)**

1. Go to https://www.kaggle.com/datasets/brycecf/give-me-some-credit
2. Download `cs-training.csv`
3. Save it as `data/raw/loan_default.csv`

**4e. Verify all datasets are present**
```bash
python data/verify_data.py
```

---

### Step 5 — Ingest all documents into ChromaDB

```bash
python src/data_pipeline.py
```

> This step downloads the embedding model (~90MB) on first run. Subsequent runs are instant.

---

### Step 6 — Verify RAG retrieval is working

```bash
python src/rag_pipeline.py
```
---

### Step 7 — Run the three use cases

>  Each script makes LLM calls via Ollama. Runtime: ~15–30 min per script on CPU.

**Use Case 1 — Stock Direction Prediction**
```bash
python src/uc1_stock.py
```

**Use Case 2 — Credit Risk / NPA Assessment**
```bash
python src/uc2_credit.py
```

**Use Case 3 — Loan Default Risk**
```bash
python src/uc3_loan.py
```

---

### Step 8 — Run ablation study

```bash
python src/ablation.py
```

Produces:
- Full ablation accuracy table (no_rag vs top_3 vs top_5)
- Hallucination rate table
- Bar chart saved to `results/ablation_chart.png`

---

### Step 9 — Run XGBoost baseline comparison

```bash
python src/baseline_ml.py
```

Trains XGBoost on the same tasks and compares accuracy vs RAG-LLM.
Results saved to `results/final_comparison.csv`.

---

### Step 10 — Launch the Streamlit web UI

```bash
streamlit run src/app.py
```

Opens at **http://localhost:8501**

Features:
- Select Use Case from sidebar
- Toggle RAG mode: no_rag / top_3 / top_5
- Enter custom inputs (stock prices, bank metrics, borrower profile)
- See prediction, confidence, reasoning, and cited sources in real time
- View ablation results table

---

## Results Summary

After running all scripts, `results/` will contain:

| File | Description |
|---|---|
| `uc1_stock_results.csv` | 150 rows — 5 stocks × 10 dates × 3 modes |
| `uc2_credit_results.csv` | 30 rows — 10 banks × 3 modes |
| `uc3_loan_results.csv` | 30 rows — 10 borrowers × 3 modes |
| `ablation_accuracy.csv` | Accuracy by use case and mode |
| `ablation_hallucination.csv` | Hallucination rate by use case and mode |
| `ablation_chart.png` | Bar chart — accuracy by RAG mode |
| `baseline_uc1_stock.csv` | XGBoost per-ticker accuracy |
| `baseline_uc2_credit.csv` | XGBoost credit risk accuracy |
| `baseline_uc3_loan.csv` | XGBoost loan default accuracy + AUC-ROC |
| `final_comparison.csv` | RAG-LLM vs XGBoost head-to-head |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| `ollama: command not found` | Install Ollama from ollama.com/download |
| `Error: listen tcp 127.0.0.1:11434: bind` | Ollama already running — this is fine, proceed |
| `chromadb collection empty` | Re-run `python src/data_pipeline.py` |
| `[SKIP] Not found: data/raw/...` | Download the missing dataset (Step 4) |
| LLM calls very slow | Normal on CPU — LLaMA3 8B takes ~10s per call |
| Word can't open .docx report | Right-click → Properties → check Unblock |

---

## References

1. Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. https://arxiv.org/abs/2005.11401
2. Ji et al. (2023) — Survey of Hallucination in Natural Language Generation. ACM Computing Surveys. https://arxiv.org/abs/2202.03629
3. Wu et al. (2023) — BloombergGPT: A Large Language Model for Finance. https://arxiv.org/abs/2303.17564
4. Arrieta et al. (2020) — Explainable AI: Concepts, taxonomies, opportunities. Information Fusion. https://arxiv.org/abs/1910.10045
5. RBI Financial Stability Report June 2024. https://www.rbi.org.in
6. RBI Financial Stability Report December 2023. https://www.rbi.org.in
7. Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System. KDD 2016.
8. Reimers & Gurevych (2019) — Sentence-BERT. EMNLP 2019. https://arxiv.org/abs/1908.10084

---

## Project Info

| Field | Details |
|---|---|
| Course | CS5202 — GenAI & LLM |
| Domain | Banking & Finance |
| LLM | LLaMA 3 8B via Ollama (local, free) |
| Vector Store | ChromaDB 1.5.8 (persistent local) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Total LLM calls | 210 (150 UC1 + 30 UC2 + 30 UC3) |
| Hallucination rate | 0% across all modes |

---

*CS5202 - GenAI & LLM · M.Tech(Artificial Intelligence and Data Science) · Domain - Banking & Finance*
