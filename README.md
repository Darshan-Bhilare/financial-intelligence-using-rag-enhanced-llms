# Financial Intelligence using RAG-Enhanced LLMs
## A Unified Explainable AI System for Indian Banking & Finance
### CS5202 — GenAI and LLM · Spring 2026 · Domain D

## Problem
Indian banks and investors face hallucination-prone AI outputs with no cited evidence.
This system grounds Gemini LLM with domain documents via RAG to produce explainable,
source-cited financial intelligence across three use cases.

## Use Cases
- UC1: Stock price direction prediction (TCS, INFY, ICICI, HDFC, Reliance)
- UC2: Credit risk and NPA assessment (RBI policy grounded)
- UC3: Loan default risk prediction (borrower-level explanation)

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add your Gemini API key
```

## Run
```bash
python src/data_pipeline.py   # ingest all documents into ChromaDB
python src/uc1_stock.py       # run stock prediction use case
python src/uc2_credit.py      # run credit risk use case
python src/uc3_loan.py        # run loan default use case
python src/ablation.py        # run full ablation study
```

## Repo Structure

## Team
Bhilare Darshan Parvati (SE25MAID010)