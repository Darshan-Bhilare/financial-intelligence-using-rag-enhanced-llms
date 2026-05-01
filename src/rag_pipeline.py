import chromadb
from chromadb.utils import embedding_functions

# ── config ─────────────────────────────────────────────────────────────────
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "finrag"

# ── modes ──────────────────────────────────────────────────────────────────
RAG_MODES = {
    "no_rag": 0,
    "top_3":  3,
    "top_5":  5,
}

# ── chromadb client ─────────────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name=EMBED_MODEL
             )
    return client.get_or_create_collection(name=COLLECTION, embedding_function=ef)


# ── retrieve ─────────────────────────────────────────────────────────────────
def retrieve(query, mode="top_3", source_filter=None):
    """
    Retrieve relevant chunks from ChromaDB.

    Args:
        query         : the search query string
        mode          : "no_rag" | "top_3" | "top_5"
        source_filter : optional — filter by source e.g. "indian_financial_news"

    Returns:
        list of dicts with keys: text, source, distance
    """
    k = RAG_MODES.get(mode, 3)

    # no-rag mode — return empty context
    if k == 0:
        return []

    collection = get_collection()

    # build query params
    query_params = {
        "query_texts": [query],
        "n_results":    k,
        "include":     ["documents", "metadatas", "distances"],
    }

    # optional source filter
    if source_filter:
        query_params["where"] = {"source": {"$eq": source_filter}}

    results = collection.query(**query_params)

    # format output
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })

    return chunks


# ── format context for LLM prompt ───────────────────────────────────────────
def format_context(chunks):
    """
    Formats retrieved chunks into a clean string for the LLM prompt.
    """
    if not chunks:
        return "No context retrieved. Answer from general knowledge only."

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i} — {chunk['source']}]\n{chunk['text']}\n\n"

    return context.strip()


# ── main: test retrieval ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FinRAG Retrieval Pipeline — Test")
    print("=" * 50 + "\n")

    test_queries = [
        ("TCS stock price movement 2024",          "top_3", None),
        ("RBI NPA credit risk policy",             "top_5", None),
        ("loan default debt ratio income",         "top_3", None),
        ("Reliance quarterly earnings",            "no_rag", None),
    ]

    for query, mode, source in test_queries:
        print(f"Query  : {query}")
        print(f"Mode   : {mode}")
        chunks = retrieve(query, mode=mode, source_filter=source)
        ctx    = format_context(chunks)
        print(f"Chunks : {len(chunks)}")
        print(f"Context preview:\n{ctx[:300]}...")
        print("-" * 50 + "\n")