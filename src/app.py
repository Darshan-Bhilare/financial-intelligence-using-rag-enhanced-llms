import sys
import os
import json
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import retrieve, format_context
from llm_client  import predict_stock, assess_credit_risk, assess_loan_default

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRAG — Indian Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark navy background */
.stApp {
    background-color: #0a0e1a;
    color: #e0e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1729;
    border-right: 1px solid #1e2d4a;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #8ba3c7 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Header */
.fin-header {
    background: linear-gradient(135deg, #0f1729 0%, #162040 50%, #0f1729 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.fin-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066ff, #00d4ff);
}
.fin-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #00d4ff;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.fin-header p {
    color: #5a7a9e;
    font-size: 0.85rem;
    margin: 0;
    font-weight: 300;
}
.fin-badge {
    display: inline-block;
    background: #00d4ff18;
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 6px;
    margin-top: 10px;
}

/* Cards */
.fin-card {
    background: #0f1729;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.fin-card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #5a7a9e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* Metric boxes */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-box {
    background: #162040;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    color: #5a7a9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #00d4ff;
}
.metric-value.up    { color: #00e676; }
.metric-value.down  { color: #ff5252; }
.metric-value.high  { color: #ff5252; }
.metric-value.med   { color: #ffab40; }
.metric-value.low   { color: #00e676; }

/* Source chunks */
.source-chunk {
    background: #080d18;
    border-left: 3px solid #00d4ff44;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #8ba3c7;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.5;
}
.source-label {
    font-size: 0.68rem;
    color: #00d4ff88;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* Reasoning box */
.reasoning-box {
    background: #080d18;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #c0cfe8;
    line-height: 1.65;
    font-style: italic;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
}
.risk-HIGH   { background: #ff525222; border: 1px solid #ff5252; color: #ff5252; }
.risk-MEDIUM { background: #ffab4022; border: 1px solid #ffab40; color: #ffab40; }
.risk-LOW    { background: #00e67622; border: 1px solid #00e676; color: #00e676; }
.risk-UP     { background: #00e67622; border: 1px solid #00e676; color: #00e676; }
.risk-DOWN   { background: #ff525222; border: 1px solid #ff5252; color: #ff5252; }

/* Ablation table */
.abl-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.abl-table th {
    background: #162040;
    color: #5a7a9e;
    padding: 8px 14px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #1e2d4a;
}
.abl-table td {
    padding: 8px 14px;
    border-bottom: 1px solid #0f1729;
    color: #c0cfe8;
}
.abl-table tr:hover td { background: #162040; }
.abl-best { color: #00d4ff; font-weight: 600; }

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Input labels */
.stNumberInput label, .stSelectbox label, .stSlider label {
    color: #8ba3c7 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #080d18 !important;
    border: 1px solid #1e2d4a !important;
    color: #e0e6f0 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Divider */
hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### FinRAG")
    st.markdown("---")

    use_case = st.selectbox(
        "Use Case",
        ["UC1 — Stock Prediction", "UC2 — Credit Risk / NPA", "UC3 — Loan Default"],
    )

    rag_mode = st.radio(
        "RAG Mode",
        ["no_rag", "top_3", "top_5"],
        index=1,
        help="no_rag = LLM only (baseline) | top_3/top_5 = RAG grounded"
    )

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("🦙 LLaMA 3 via Ollama")
    st.markdown("**Vector Store**")
    st.markdown("🗄️ ChromaDB")
    st.markdown("**Embeddings**")
    st.markdown("🔢 all-MiniLM-L6-v2")
    st.markdown("---")

    show_ablation = st.checkbox("Show Ablation Results", value=False)
    show_context  = st.checkbox("Show Retrieved Chunks",  value=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fin-header">
    <h1>FinRAG · Indian Financial Intelligence</h1>
    <p>Financial Intelligence using RAG-Enhanced LLMs — A Unified Explainable AI System for Indian Banking &amp; Finance</p>
    <span class="fin-badge">CS5202 Spring 2026</span>
    <span class="fin-badge">Domain D</span>
    <span class="fin-badge">RAG + LLM</span>
    <span class="fin-badge">Ollama LLaMA3</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE (optional)
# ══════════════════════════════════════════════════════════════════════════════
if show_ablation:
    st.markdown("### 📊 Ablation Study Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">UC1 · Stock Prediction</div>
            <table class="abl-table">
                <tr><th>Mode</th><th>Accuracy</th></tr>
                <tr><td>no_rag</td><td class="abl-best">58.0%</td></tr>
                <tr><td>top_3</td><td>52.0%</td></tr>
                <tr><td>top_5</td><td>44.0%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">UC2 · Credit Risk / NPA</div>
            <table class="abl-table">
                <tr><th>Mode</th><th>Accuracy</th></tr>
                <tr><td>no_rag</td><td>50.0%</td></tr>
                <tr><td>top_3</td><td class="abl-best">66.67%</td></tr>
                <tr><td>top_5</td><td>60.0%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="fin-card">
            <div class="fin-card-title">UC3 · Loan Default</div>
            <table class="abl-table">
                <tr><th>Mode</th><th>Accuracy</th></tr>
                <tr><td>no_rag</td><td class="abl-best">60.0%</td></tr>
                <tr><td>top_3</td><td>50.0%</td></tr>
                <tr><td>top_5</td><td>50.0%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# UC1 — STOCK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "UC1" in use_case:
    st.markdown("## 📈 UC1 — Stock Direction Prediction")
    st.markdown("The LLM reads retrieved news and OHLCV data, then predicts stock direction with cited evidence.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        ticker = st.selectbox("Stock Ticker", ["TCS", "INFY", "ICICI", "HDFC", "RELIANCE"])
        open_p  = st.number_input("Open Price (₹)",  value=3500.0, step=10.0)
        high_p  = st.number_input("High Price (₹)",  value=3560.0, step=10.0)
        low_p   = st.number_input("Low Price (₹)",   value=3480.0, step=10.0)
        close_p = st.number_input("Close Price (₹)", value=3540.0, step=10.0)
        volume  = st.number_input("Volume",           value=1500000, step=10000)
        run_btn = st.button("🔍 Run Prediction")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if run_btn:
            with st.spinner("Retrieving context and running LLM..."):
                query   = (f"{ticker} stock price Open={open_p} High={high_p} "
                           f"Low={low_p} Close={close_p} Volume={volume}")
                chunks  = retrieve(query, mode=rag_mode)
                context = format_context(chunks)
                result  = predict_stock(context, ticker, query)

            pred  = result.get("prediction",  "N/A")
            conf  = result.get("confidence",  "N/A")
            hrisk = result.get("hallucination_risk", "N/A")
            reasoning = result.get("reasoning", "No reasoning provided.")

            color_cls = "up" if pred == "UP" else "down"

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value {color_cls}">{pred}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{conf}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Hallucination Risk</div>
                    <div class="metric-value">{hrisk}</div>
                </div>
            </div>
            <div class="fin-card">
                <div class="fin-card-title">LLM Reasoning</div>
                <div class="reasoning-box">{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_context and chunks:
                st.markdown('<div class="fin-card">', unsafe_allow_html=True)
                st.markdown('<div class="fin-card-title">Retrieved Context Chunks</div>', unsafe_allow_html=True)
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-label">Source {i} · {chunk['source']} · distance={chunk['distance']}</div>
                        {chunk['text'][:300]}...
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UC2 — CREDIT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif "UC2" in use_case:
    st.markdown("## 🏦 UC2 — Credit Risk / NPA Assessment")
    st.markdown("The LLM reads retrieved RBI policy chunks and bank metrics, then assesses NPA risk level.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        bank_name   = st.text_input("Bank Name", value="Sample Indian Bank")
        npa_ratio   = st.slider("NPA Ratio (%)",        0.0, 15.0, 4.2, 0.1)
        credit_gr   = st.slider("Credit Growth (%)",    0.0, 30.0, 11.0, 0.5)
        cap_adeq    = st.slider("Capital Adequacy (%)", 8.0, 25.0, 14.5, 0.1)
        run_btn     = st.button("🔍 Assess Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if run_btn:
            bank_data = {
                "bank":               bank_name,
                "npa_ratio":          f"{npa_ratio}%",
                "credit_growth":      f"{credit_gr}%",
                "capital_adequacy":   f"{cap_adeq}%",
            }
            with st.spinner("Retrieving RBI policy and running LLM..."):
                query   = (f"NPA risk assessment bank {bank_name} "
                           f"NPA={npa_ratio}% credit_growth={credit_gr}% capital={cap_adeq}%")
                chunks  = retrieve(query, mode=rag_mode)
                context = format_context(chunks)
                result  = assess_credit_risk(context, bank_data, query)

            risk      = result.get("risk_level",          "N/A")
            npa_ass   = result.get("npa_assessment",      "N/A")
            rbi_ref   = result.get("rbi_policy_reference","None")
            reasoning = result.get("reasoning",           "No reasoning provided.")
            hrisk     = result.get("hallucination_risk",  "N/A")

            risk_cls = risk if risk in ["HIGH","MEDIUM","LOW"] else "LOW"

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value {risk_cls.lower()}">{risk}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">NPA Ratio</div>
                    <div class="metric-value">{npa_ratio}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Hallucination Risk</div>
                    <div class="metric-value">{hrisk}</div>
                </div>
            </div>
            <div class="fin-card">
                <div class="fin-card-title">NPA Assessment</div>
                <div class="reasoning-box">{npa_ass}</div>
            </div>
            <div class="fin-card">
                <div class="fin-card-title">RBI Policy Reference</div>
                <div class="reasoning-box">{rbi_ref}</div>
            </div>
            <div class="fin-card">
                <div class="fin-card-title">LLM Reasoning</div>
                <div class="reasoning-box">{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_context and chunks:
                st.markdown('<div class="fin-card">', unsafe_allow_html=True)
                st.markdown('<div class="fin-card-title">Retrieved Context Chunks</div>', unsafe_allow_html=True)
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-label">Source {i} · {chunk['source']} · distance={chunk['distance']}</div>
                        {chunk['text'][:300]}...
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — LOAN DEFAULT
# ══════════════════════════════════════════════════════════════════════════════
elif "UC3" in use_case:
    st.markdown("## 💳 UC3 — Loan Default Risk Assessment")
    st.markdown("The LLM reads retrieved loan profiles and lending rules, then assesses default risk with justification.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        age         = st.number_input("Age",             value=45,   step=1)
        income      = st.number_input("Monthly Income (₹)", value=35000, step=500)
        debt_ratio  = st.slider("Debt Ratio",            0.0, 2.0, 0.65, 0.01)
        rev_util    = st.slider("Revolving Utilization", 0.0, 1.0, 0.82, 0.01)
        late_90     = st.number_input("Times 90 Days Late", value=2, step=1)
        credit_lines = st.number_input("Open Credit Lines",  value=8, step=1)
        run_btn     = st.button("🔍 Assess Default Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if run_btn:
            applicant = {
                "age":             int(age),
                "monthly_income":  float(income),
                "debt_ratio":      float(debt_ratio),
                "revolving_util":  float(rev_util),
                "times_90_late":   int(late_90),
                "open_credit_lines": int(credit_lines),
            }
            with st.spinner("Retrieving loan profiles and running LLM..."):
                query   = (f"loan default risk age={age} income={income} "
                           f"debt_ratio={debt_ratio} revolving={rev_util} "
                           f"times_90_late={late_90}")
                chunks  = retrieve(query, mode=rag_mode)
                context = format_context(chunks)
                result  = assess_loan_default(context, applicant, query)

            risk      = result.get("default_risk",        "N/A")
            prob      = result.get("default_probability", "N/A")
            factors   = result.get("key_risk_factors",    [])
            reasoning = result.get("reasoning",           "No reasoning provided.")
            hrisk     = result.get("hallucination_risk",  "N/A")

            risk_cls = risk if risk in ["HIGH","MEDIUM","LOW"] else "LOW"

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Default Risk</div>
                    <div class="metric-value {risk_cls.lower()}">{risk}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Default Probability</div>
                    <div class="metric-value">{prob}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Hallucination Risk</div>
                    <div class="metric-value">{hrisk}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if factors:
                st.markdown('<div class="fin-card">', unsafe_allow_html=True)
                st.markdown('<div class="fin-card-title">Key Risk Factors</div>', unsafe_allow_html=True)
                for f in factors:
                    st.markdown(f'<span class="risk-badge risk-HIGH">⚠ {f}</span>&nbsp;', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="fin-card">
                <div class="fin-card-title">LLM Reasoning</div>
                <div class="reasoning-box">{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_context and chunks:
                st.markdown('<div class="fin-card">', unsafe_allow_html=True)
                st.markdown('<div class="fin-card-title">Retrieved Context Chunks</div>', unsafe_allow_html=True)
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-label">Source {i} · {chunk['source']} · distance={chunk['distance']}</div>
                        {chunk['text'][:300]}...
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#2a3f5f; font-size:0.75rem; font-family:'IBM Plex Mono',monospace;">
    FinRAG · CS5202 Spring 2026 · Financial Intelligence using RAG-Enhanced LLMs ·
    Domain D — Indian Banking &amp; Finance
</div>
""", unsafe_allow_html=True)