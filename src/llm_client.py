import ollama
import json

# ── config ─────────────────────────────────────────────────────────────────
MODEL      = "llama3"
TIMEOUT    = 120  # seconds

# ── system prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are FinRAG, an expert financial analyst AI for Indian banking and markets.

STRICT RULES you must always follow:
1. Answer ONLY using the provided context chunks.
2. If context is empty, say "No context available" and give a general answer.
3. Always cite your sources using [Source N] notation.
4. Never hallucinate facts, numbers, or dates not present in the context.
5. Always return a valid JSON object — nothing else.
6. Be concise and precise in your reasoning.
"""

# ── prompt templates ────────────────────────────────────────────────────────
def build_stock_prompt(context, ticker, query):
    return f"""
CONTEXT:
{context}

TASK: Stock direction prediction for {ticker}.
QUERY: {query}

Based ONLY on the context above, analyze the stock and respond with this exact JSON:
{{
    "ticker": "{ticker}",
    "prediction": "UP or DOWN",
    "confidence": "HIGH or MEDIUM or LOW",
    "reasoning": "2-3 sentences citing specific numbers from context",
    "sources_used": ["Source 1", "Source 2"],
    "hallucination_risk": "LOW if context used, HIGH if guessed"
}}
"""

def build_credit_prompt(context, bank_data, query):
    return f"""
CONTEXT:
{context}

BANK DATA: {bank_data}
QUERY: {query}

Based ONLY on the context above, assess the credit risk and respond with this exact JSON:
{{
    "risk_level": "HIGH or MEDIUM or LOW",
    "npa_assessment": "1-2 sentences on NPA risk",
    "rbi_policy_reference": "cite specific RBI policy from context or None",
    "reasoning": "2-3 sentences citing context",
    "sources_used": ["Source 1", "Source 2"],
"hallucination_risk": "LOW if context used, HIGH if guessed"
}}

IMPORTANT: Return ONLY the JSON object above. No extra text, no markdown, no explanation.
"""

def build_loan_prompt(context, applicant_data, query):
    return f"""
CONTEXT:
{context}

APPLICANT PROFILE: {applicant_data}
QUERY: {query}

Based ONLY on the context above, assess loan default risk and respond with this exact JSON:
{{
    "default_risk": "HIGH or MEDIUM or LOW",
    "default_probability": "estimated percentage e.g. 35%",
    "key_risk_factors": ["factor 1", "factor 2", "factor 3"],
    "reasoning": "2-3 sentences citing context",
    "sources_used": ["Source 1", "Source 2"],
    "hallucination_risk": "LOW if context used, HIGH if guessed"
}}
"""

# ── core LLM call ───────────────────────────────────────────────────────────
def call_llm(prompt, system=SYSTEM_PROMPT):
    """
    Call Ollama LLaMA3 with a grounded prompt.
    Returns parsed JSON dict or raw text if parsing fails.
    """
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            options={"temperature": 0.1}  # low temp = consistent, less hallucination
        )

        raw = response["message"]["content"].strip()

        # extract JSON from response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # retry with cleaned string
                json_str = json_str.replace("\n", " ").replace("\t", " ").replace("None", "null").replace("True", "true").replace("False", "false")
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return {"raw_response": raw, "parse_error": "Invalid JSON from LLM"}
        else:
            return {"raw_response": raw, "parse_error": "No JSON found"}

    except Exception as e:
        return {"error": str(e)}


# ── use case wrappers ───────────────────────────────────────────────────────
def predict_stock(context, ticker, query):
    prompt = build_stock_prompt(context, ticker, query)
    return call_llm(prompt)

def assess_credit_risk(context, bank_data, query):
    prompt = build_credit_prompt(context, bank_data, query)
    return call_llm(prompt)

def assess_loan_default(context, applicant_data, query):
    prompt = build_loan_prompt(context, applicant_data, query)
    return call_llm(prompt)


# ── main: test all three use cases ─────────────────────────────────────────
if __name__ == "__main__":
    from rag_pipeline import retrieve, format_context

    print("=" * 55)
    print("   FinRAG LLM Client — Testing All Use Cases")
    print("=" * 55)

    # ── UC1: Stock prediction ──────────────────────────────
    print("\n[UC1] Stock Prediction — TCS\n")
    chunks  = retrieve("TCS stock price movement 2024", mode="top_3")
    context = format_context(chunks)
    result  = predict_stock(context, "TCS", "Will TCS stock go UP or DOWN based on recent price data?")
    print(json.dumps(result, indent=2))

    # ── UC2: Credit risk ───────────────────────────────────
    print("\n[UC2] Credit Risk / NPA Assessment\n")
    bank_data = {
        "bank": "Sample Indian Bank",
        "npa_ratio": "4.2%",
        "credit_growth": "11%",
        "capital_adequacy": "14.5%"
    }
    chunks  = retrieve("RBI NPA credit risk banking policy India", mode="top_5")
    context = format_context(chunks)
    result  = assess_credit_risk(context, bank_data, "What is the NPA risk level for this bank based on RBI guidelines?")
    print(json.dumps(result, indent=2))

    # ── UC3: Loan default ──────────────────────────────────
    print("\n[UC3] Loan Default Risk Assessment\n")
    applicant = {
        "age": 45,
        "monthly_income": 35000,
        "debt_ratio": 0.65,
        "revolving_utilization": 0.82,
        "times_90_days_late": 2,
        "open_credit_lines": 8
    }
    chunks  = retrieve("loan default debt ratio income credit lines", mode="top_3")
    context = format_context(chunks)
    result  = assess_loan_default(context, applicant, "What is the loan default risk for this applicant?")
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 55)
    print("   All use cases complete")
    print("=" * 55)