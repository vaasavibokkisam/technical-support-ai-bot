"""
rag.py — Part B: RAG Pipeline
==============================
This module contains:
  B1. Semantic Retrieval  — retrieve top-K relevant chunks from ChromaDB
  B2. API Integration     — call DeepInfra LLM with retrieved context
                            and a hallucination-guard system prompt

Imported by app.py (the Streamlit UI).
"""

import os
import time
import requests                                          # for HTTP calls to DeepInfra
from sentence_transformers import SentenceTransformer   # same model used in ingest.py
import chromadb

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "upwork_api_docs"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 5          # number of chunks to retrieve (assignment requirement)

# DeepInfra OpenAI-compatible endpoint
# DeepInfra exposes an OpenAI-style /v1/openai/chat/completions endpoint,
# so we can use the same request format as the OpenAI SDK without the SDK.
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
LLM_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# ── System Prompt (B2 requirement) ───────────────────────────────────────────
# This prompt instructs the model to:
#   1. Stay in character as a Senior Upwork API Consultant
#   2. Answer ONLY from the provided CONTEXT (hallucination guard)
#   3. Use the exact required fallback phrase when the answer is absent
#
# {context} is a placeholder filled at query time with the retrieved chunks.
SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with deep expertise in \
the Upwork developer platform, OAuth 2.0 authentication, GraphQL APIs, and REST \
integrations. You help developers implement Upwork APIs correctly and efficiently.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. If the answer to the user's question cannot be found in the CONTEXT, you MUST \
respond with exactly: "I'm sorry, but the provided documentation does not contain \
that information."
3. Never fabricate, guess, or use knowledge outside the provided CONTEXT.
4. Be precise and technical in your answers — developers rely on accuracy.
5. When referencing specific values (e.g., rate limits, token TTLs), quote them \
exactly as they appear in the documentation.

CONTEXT:
{context}"""


# ── Singleton loader — avoids reloading the model on every query ──────────────
# Both variables start as None and are populated on first use (lazy loading).
# This pattern prevents Streamlit from reloading the model on every interaction.
_embed_model       = None
_chroma_collection = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model once and cache it in module scope."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_collection():
    """Lazy-load the ChromaDB collection once and cache it in module scope."""
    global _chroma_collection
    if _chroma_collection is None:
        client             = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# ── B1: Semantic Retrieval ────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    B1 — Semantic Retrieval
    Converts the user's query into a vector using the same embedding model
    used during ingestion, then finds the top_k most similar chunks in
    ChromaDB using cosine similarity.

    The key insight: we embed the QUERY and compare it against embedded CHUNKS.
    Because both use the same model, semantically related text ends up close
    in the 384-dimensional vector space even if it uses different words.

    Returns a list of dicts: [{"text": ..., "id": ..., "distance": ...}, ...]
    where distance is cosine distance (lower = more similar).
    """
    model      = _get_embed_model()
    collection = _get_collection()

    # Encode the query into a 384-dim vector — same space as stored chunks
    query_vector = model.encode([query]).tolist()    # [[float, ...]] — list of 1 vector

    # Query ChromaDB — returns the n_results nearest neighbours by cosine distance
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "distances"],    # get raw text + similarity scores
    )

    # ChromaDB returns nested lists (one per query); we only sent one query so [0]
    chunks = []
    for doc, dist, doc_id in zip(
        results["documents"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        chunks.append({
            "text":     doc,
            "id":       doc_id,
            "distance": round(dist, 4),    # lower cosine distance = more similar
        })

    return chunks


# ── B2: LLM Integration ───────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """
    Joins retrieved chunks into a single formatted context string.
    Numbering each source lets the model reference them clearly in its answer.
    The --- separator makes the boundary between sources visually distinct
    inside the prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Source {i}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def query_llm(user_question: str, chunks: list[dict]) -> tuple[str, float]:
    """
    B2 — API Integration & Prompting
    Sends the retrieved context + user question to the DeepInfra LLM.
    Returns (answer_text, latency_in_seconds).

    The API key is read from the DEEPINFRA_API_KEY environment variable at
    call time — never stored in source code. The dotenv library (called in
    app.py) populates os.environ from the .env file before this runs.
    """
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPINFRA_API_KEY not set. "
            "Copy .env.example → .env and add your key."
        )

    # Inject the actual retrieved text into the system prompt template
    context        = build_context(chunks)
    system_message = SYSTEM_PROMPT.format(context=context)

    # Build the OpenAI-compatible request body
    # DeepInfra accepts the same JSON schema as openai.ChatCompletion
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_question},
        ],
        "temperature": 0.3,    # low temperature → focused, deterministic answers
        "max_tokens":  512,    # enough for a thorough technical answer
    }

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",    # Bearer token auth
    }

    # ── Time the API call (B3 latency display requirement) ───────────────────
    start_time = time.time()

    response = requests.post(
        DEEPINFRA_URL,
        json=payload,
        headers=headers,
        timeout=60,    # raise requests.Timeout if no response within 60 s
    )

    latency = round(time.time() - start_time, 2)    # wall-clock seconds

    # Raise an HTTPError for 4xx / 5xx responses so Streamlit can catch it
    response.raise_for_status()

    data   = response.json()
    answer = data["choices"][0]["message"]["content"].strip()

    return answer, latency


# ── Combined pipeline (called by app.py) ──────────────────────────────────────

def answer_question(user_question: str) -> dict:
    """
    Full RAG pipeline in one call:
      1. Retrieve top-3 chunks semantically closest to the question
      2. Build context string and fill the system prompt
      3. Call the LLM and time the response
      4. Return a structured dict for the Streamlit UI to render
    """
    chunks          = retrieve_chunks(user_question)
    answer, latency = query_llm(user_question, chunks)

    return {
        "answer":  answer,
        "sources": chunks,      # list of {"text", "id", "distance"}
        "latency": latency,     # float seconds (2 decimal places)
    }