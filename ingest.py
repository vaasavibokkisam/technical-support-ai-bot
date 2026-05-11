"""
ingest.py — Part A: Knowledge Engineering
==========================================
This script handles:
  A1. Loading the Upwork API documentation (PDF)
  A2. Chunking the text (500 chars, 50-char overlap)
  A3. Embedding chunks and storing them in ChromaDB

Run this ONCE before starting the Streamlit app.
Usage:
    python ingest.py --doc path/to/API_Documentation.pdf
"""

import os
import argparse
import pdfplumber                                              # extracts text from PDF files
from langchain_text_splitters import CharacterTextSplitter   # handles chunking
from sentence_transformers import SentenceTransformer         # local embedding model
import chromadb                                               # local vector database

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"            # folder where ChromaDB persists data
COLLECTION_NAME = "upwork_api_docs"        # name of the vector collection
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # free local model
CHUNK_SIZE      = 500                      # characters per chunk (assignment req)
CHUNK_OVERLAP   = 50                       # overlap between consecutive chunks


def load_pdf(pdf_path: str) -> str:
    """
    A1 — Data Ingestion
    Opens a PDF and extracts all text from every page.
    pdfplumber is used because it handles complex PDF layouts better than PyPDF2
    and correctly extracts text from tables and multi-column layouts common in
    API documentation.
    """
    print(f"\n📄 Loading document: {pdf_path}")
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()    # returns None if page has no text
            if page_text:
                all_text += page_text + "\n"   # join pages with newline

    # ── Sanity Check (A1 requirement) ────────────────────────────────────────
    # We print total character count and a sample to verify the file loaded correctly.
    # If all_text is empty, the PDF may be image-based (scanned) and needs OCR.
    print(f"\n✅ Sanity Check:")
    print(f"   Total characters loaded : {len(all_text):,}")
    print(f"   Sample (first 500 chars):\n")
    print("   " + "-" * 60)
    print(all_text[:500])
    print("   " + "-" * 60)

    if len(all_text) == 0:
        raise ValueError("❌ Document appears empty. Check the PDF path or if it is scanned.")

    return all_text


def chunk_text(text: str) -> list[str]:
    """
    A2 — Document Chunking
    Splits the full text into overlapping chunks.

    Why overlap matters (A2 explanation):
    API documentation often contains code snippets, endpoint definitions, and
    parameter tables where a single concept spans multiple lines right at a
    natural split boundary. Without overlap, a chunk might end mid-sentence
    or mid-code-block, leaving both adjacent chunks with incomplete context.
    A 50-character overlap ensures that boundary content appears in BOTH the
    preceding and following chunks, so vector similarity search can still
    retrieve the relevant chunk even when a query keyword lands at the edge.
    """
    splitter = CharacterTextSplitter(
        separator="\n",       # prefer splitting on newlines (natural boundaries)
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # measure length in characters, not tokens
    )

    chunks = splitter.split_text(text)

    print(f"\n✂️  Chunking complete:")
    print(f"   Chunk size    : {CHUNK_SIZE} characters")
    print(f"   Chunk overlap : {CHUNK_OVERLAP} characters")
    print(f"   Total chunks  : {len(chunks)}")
    print(f"   Sample chunk  :\n   {chunks[0][:200]}...")

    return chunks


def embed_and_store(chunks: list[str]):
    """
    A3 — Vector Storage
    1. Loads a local sentence-transformer model to create embeddings.
       No external API call is made — the model runs entirely on this machine.
       all-MiniLM-L6-v2 produces 384-dimensional vectors and is fast enough
       for this document size while still being semantically accurate.
    2. Stores the vectors in ChromaDB, a lightweight local vector database
       that persists to disk so we don't re-embed on every app launch.
    """
    print(f"\n🤖 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"⚙️  Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,    # displays a tqdm progress bar
        batch_size=32,             # process 32 chunks at a time for efficiency
    )
    # embeddings shape: (num_chunks, 384) — 384-dim float vectors for MiniLM

    # ── Set up ChromaDB ───────────────────────────────────────────────────────
    # PersistentClient saves data to disk automatically; survives between runs
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection so we can re-ingest cleanly if re-run
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}' (clean re-ingest)")
    except Exception:
        pass    # collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},    # use cosine similarity for search
    )

    # ── Insert chunks + embeddings ────────────────────────────────────────────
    # ChromaDB requires string IDs for each document
    ids             = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()    # numpy array → plain Python list

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=chunks,           # the raw text is stored alongside the vector
    )

    print(f"\n✅ ChromaDB storage complete:")
    print(f"   Location   : {CHROMA_DB_PATH}/")
    print(f"   Collection : {COLLECTION_NAME}")
    print(f"   Documents  : {collection.count()}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Upwork API docs into ChromaDB")
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Path to the Upwork API documentation PDF",
    )
    args = parser.parse_args()

    if not os.path.exists(args.doc):
        raise FileNotFoundError(f"Document not found: {args.doc}")

    # Run the three-step pipeline: load → chunk → embed & store
    raw_text = load_pdf(args.doc)
    chunks   = chunk_text(raw_text)
    embed_and_store(chunks)

    print("\n🎉 Ingestion complete! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()