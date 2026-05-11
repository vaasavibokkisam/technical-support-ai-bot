"""
app.py — Streamlit UI
=====================

Frontend for the Upwork API Technical Support Bot.

Run:
    streamlit run app.py
"""

import html
import streamlit as st
from dotenv import load_dotenv
import chromadb

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "upwork_api_docs"

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #14a800;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        background: #f0faf0;
        border-left: 4px solid #14a800;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    .source-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🤖 Upwork API Support Bot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    'Powered by RAG + Meta-Llama-3.1 via DeepInfra'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Upwork API Bot")
    st.info(
        "This bot answers questions about the Upwork API using a "
        "Retrieval-Augmented Generation (RAG) pipeline."
    )

# ── Check ChromaDB ───────────────────────────────────────────────────────────
def chroma_ready():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


if not chroma_ready():
    st.error(
        """
❌ ChromaDB collection not found.

Run ingestion first:

python ingest.py --doc API_Documentation.pdf
"""
    )
    st.stop()

# ── Import RAG Pipeline ──────────────────────────────────────────────────────
from rag import answer_question

# ── Input Area ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    user_query = st.text_input(
        "Ask a question about the Upwork API",
        placeholder="e.g. How long is an OAuth access token valid for?",
        key="user_query",
        label_visibility="collapsed",
    )

with col2:
    ask_btn = st.button(
        "🔍 Ask",
        use_container_width=True,
        type="primary",
    )

# ── Response ─────────────────────────────────────────────────────────────────
if ask_btn and user_query.strip():

    with st.spinner("Retrieving documents and querying LLM..."):
        try:
            result = answer_question(user_query.strip())
        except EnvironmentError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    # ── Metrics ──────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ Latency", f"{result['latency']} s")
    m2.metric("📄 Chunks Used", len(result["sources"]))
    m3.metric("🤖 Model", "Llama-3.1-8B")

    st.divider()

    # ── Answer ───────────────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-box">{html.escape(result["answer"])}</div>',
        unsafe_allow_html=True,
    )

    # ── Sources ──────────────────────────────────────────────────────────────
    st.markdown("### 📚 Retrieved Sources")

    for i, source in enumerate(result["sources"], start=1):
        with st.expander(
            f"Source {i} — {source['id']} (distance: {source['distance']})",
            expanded=(i == 1),
        ):
            st.markdown(
                f'<div class="source-box">{html.escape(source["text"])}</div>',
                unsafe_allow_html=True,
            )

elif ask_btn and not user_query.strip():
    st.warning("⚠️ Please enter a question.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #999;">
        <div style="font-size:3rem">🤖</div>
        <div style="font-size:1.1rem; margin-top:0.5rem">
            Type your question above and click Ask.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Answers are generated only from the provided Upwork API documentation."
)