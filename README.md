#Financial Intelligence using RAG-Enhanced LLMs: A Unified Explainable AI System for Indian Banking & Finance
## Stock Price Prediction and Risk Analysis  

---

## 📌 Problem Statement  

Stock price forecasting is one of the most economically significant and technically challenging problems in applied machine learning.  

Traditional models such as ARIMA, GARCH, and LSTMs rely only on historical price and volume data. As a result, they fail to capture:  

- Breaking news  
- Earnings announcements  
- Macroeconomic events  

On the other hand, Large Language Models (LLMs) can process textual data but may produce unreliable or unsupported outputs when not grounded in real data.  

---

## ✅ Proposed Solution  

This project combines:  

- Time-series modeling (Transformer / LSTM)  
- Retrieval-Augmented Generation (RAG)  

This ensures predictions are supported by:  

- Historical price patterns  
- Real financial news data  

---

## 🚀 Project Overview  

This project is a hybrid AI forecasting system built on 22 years of data (2003–2025) for 5 major NSE stocks.  

---

## 🔹 System Architecture  

The system integrates two main streams:  

### 1. Numerical Stream  

- OHLCV stock data  
- Feature engineering (technical indicators)  
- Time-series models (LSTM / Transformer)  

➡️ Output: Pattern-based price prediction  

---

### 2. Textual Stream  

- 50,000 financial news articles  
- Sentiment analysis using NLP models  
- RAG-based retrieval system  

➡️ Output: Context-aware market insights  

---

### 3. Fusion Layer  

Combines numerical + textual signals  

Produces:  

- 📈 Price prediction  
- 📊 Volatility estimation  
- ⚠️ Risk score  

---

## 📊 Dataset  

| Dataset               | Source                   | Size              | Date Range | Stocks / Scope                           | Usage                            |
|----------------------|--------------------------|-------------------|------------|------------------------------------------|----------------------------------|
| NSE Stock Prices     | Yahoo Finance (yfinance) | ~5,703 rows/stock | 2003–2025  | TCS, INFY, ICICIBANK, HDFCBANK, RELIANCE | Prediction & feature engineering |
| Indian Financial News| Kaggle                   | 50,000 articles   | 2003–2020  | Indian financial market                  | Sentiment + RAG retrieval        |

---

## 🔗 Data Sources  

- Stock Data: Yahoo Finance (yfinance)  
- News Dataset: https://www.kaggle.com/datasets/hkapoor/indian-financial-news-articles-20032020  

---

## 🤖 Models  

| Model                         | Type          | Role                   | Status         | Metric                    |
|------------------------------|---------------|------------------------|----------------|---------------------------|
| ARIMA / Exponential Smoothing| Classical     | Baseline forecasting   | ✅ Completed   | ~51% Directional Accuracy |
| LSTM / Transformer           | Deep Learning | Time-series prediction | 🔄 In Progress | Target > 62%              |
| FinBERT (or NLP Model)       | NLP           | Sentiment analysis     | 🔄 In Progress | Sentiment accuracy        |
| RAG (FAISS / ChromaDB)       | Retrieval     | Context grounding      | 🔄 In Progress | Retrieval relevance       |

---

## 🎯 Goals  

| Goal                            | Success Criteria               |
|---------------------------------|------------------------------|
| Improve prediction accuracy     | Directional Accuracy > 62%   |
| Build context-aware forecasting | Combine price + news data    |
| Provide interpretable results   | Each prediction linked to news |
| Deploy real-time system         | API response < 3 seconds     |

---

## 🔄 Pipeline  

### 1. Data Ingestion  

- Load stock data (5 companies)  
- Load news dataset (50,000 articles)  
- Parse and validate timestamps  

📁 Output:
```
data/raw/
```

---

### 2. Data Processing  

- Clean OHLCV data  

Feature engineering:  

- RSI  
- MACD  
- Bollinger Bands  
- ATR  
- Lag features  

Dataset split:  

- Train: 2003–2020  
- Validation: 2021–2022  
- Test: 2023–2025  

📁 Output:
```
data/processed/
├── *_processed.csv
├── *_with_sentiment.csv
├── rag_chunks_all.csv
```

---

### 3. Sentiment Analysis  

- Apply NLP on news headlines  
- Generate daily sentiment scores  
- Merge with stock dataset  

---

### 4. RAG System  

- Convert news into embeddings  
- Store in vector database (FAISS / ChromaDB)  
- Retrieve relevant news during prediction  

---

### 5. Time-Series Modeling  

- Train LSTM / Transformer models  

Input:  

- Historical prices  
- Technical indicators  
- Sentiment features  

---

### 6. Forecast Engine  

Outputs:  

- 📈 Predicted Price  
- 📊 Volatility  
- ⚠️ Risk Score  

---

## 📊 Evaluation Metrics  

- RMSE (Root Mean Square Error)  
- MAE (Mean Absolute Error)  
- Directional Accuracy  
- Risk-adjusted return (Sharpe Ratio)  

---

## 🛠️ Tech Stack  

**Language:**  
- Python  

**Libraries:**  
- pandas, numpy  
- scikit-learn  
- TensorFlow / PyTorch  
- HuggingFace Transformers  
- FAISS / ChromaDB  
- matplotlib / seaborn  

---

## 📁 Project Structure  

```
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
├── outputs/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── rag_pipeline.py
│   ├── forecasting_model.py
│
├── app/
│   ├── api.py
│   ├── dashboard.py
│
└── README.md
```

---

## 🚀 Future Work  

- Real-time data integration  
- Advanced Transformer models  
- Web dashboard (Streamlit / React)  
- Deployment using FastAPI  

---

## 📌 Conclusion  

This project demonstrates how combining:  

- Structured financial data  
- Unstructured news data  

can improve forecasting performance and provide more meaningful financial insights.  

---

## 👨‍💻 Author  

Mohd Raoof Ahmed Siddiqui,  
Jash Kolsawala,  
Bhilare Darshan Parvati,  
Simhudu.  

---

**M.Tech Project — Generative AI**  
**Financial Forecasting System**
