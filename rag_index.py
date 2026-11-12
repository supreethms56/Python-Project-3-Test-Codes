"""
Chunk documents, embed them, and store them in ChromaDB.

Install once (inside your activated venv):
    python -m pip install chromadb fastembed numpy

Folder layout this script expects:
    your-project/
        rag_index.py      <- this file
        docs/             <- put your .txt files here
        chroma_db/        <- created automatically (the database)
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import chromadb
from fastembed import TextEmbedding

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DOCS_FOLDER = "docs"                     # where your .txt files are
DB_FOLDER = "chroma_db"                  # where ChromaDB saves its data
COLLECTION_NAME = "my_documents"         # like a table name
MODEL_NAME = "BAAI/bge-small-en-v1.5"    # embedding model

_model = None


def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


# ---------------------------------------------------------------
# 1. Embedding helpers
# ---------------------------------------------------------------
def embed_texts(texts):
    """Embed document text (sentences, windows or chunks)."""
    vecs = np.array(list(get_model().embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(question):
    """Embed a search question. BGE models expect queries to be embedded this way."""
    vec = np.array(list(get_model().query_embed(question)))[0]
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------
# 2. Semantic chunking (same logic as before)
# ---------------------------------------------------------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    parts = re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(text, buffer=1, percentile=75, min_words=15, max_words=300):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    windows = [" ".join(sentences[max(0, i - buffer): i + buffer + 1])
               for i in range(len(sentences))]
    vecs = embed_texts(windows)
    distances = [1 - float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
    cut_at = np.percentile(distances, percentile)

    chunks, current = [], [sentences[0]]
    current_words = len(sentences[0].split())
    for i, sentence in enumerate(sentences[1:]):
        words = len(sentence.split())
        if (distances[i] > cut_at and current_words >= min_words) or current_words + words > max_words:
            chunks.append(" ".join(current))
            current, current_words = [sentence], words
        else:
            current.append(sentence)
            current_words += words
    chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------
# 3. ChromaDB setup
# ---------------------------------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=DB_FOLDER)     # saves to disk
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},                 # search with cosine distance
    )


# ---------------------------------------------------------------
# 4. Index: read files -> chunk -> embed -> store
# ---------------------------------------------------------------
def index_folder(folder=DOCS_FOLDER):
    collection = get_collection()
    files = sorted(Path(folder).glob("*.txt"))
    if not files:
        print(f"No .txt files found in '{folder}'.")
        return

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = semantic_chunk(text)
        if not chunks:
            continue

        # Remove this file's old chunks, so re-running never leaves stale pieces
        collection.delete(where={"source": path.name})

        embeddings = embed_texts(chunks)
        doc_date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")

        collection.add(
            ids=[f"{path.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": path.name, "chunk": i, "date": doc_date}
                       for i in range(len(chunks))],
        )
        print(f"Indexed {path.name}: {len(chunks)} chunks")

    print(f"\nTotal chunks in database: {collection.count()}")


# ---------------------------------------------------------------
# 5. Search: question -> embed -> find closest chunks
# ---------------------------------------------------------------
def search(question, k=3):
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embed_query(question).tolist()],
        n_results=k,
    )

    print(f"\nQuestion: {question}\n")
    for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
        similarity = 1 - dist
        print(f"[{rank}] similarity {similarity:.2f} | {meta['source']} (chunk {meta['chunk']}, {meta['date']})")
        print(f"    {doc}\n")
    return results


# ---------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Create a sample file the first time, so the script works out of the box
    Path(DOCS_FOLDER).mkdir(exist_ok=True)
    sample = Path(DOCS_FOLDER) / "sample.txt"
    if not any(Path(DOCS_FOLDER).glob("*.txt")):
        sample.write_text(
            "Cats are popular pets around the world. They are independent and love to nap "
            "for most of the day. Many cats enjoy sitting in sunny spots near windows. "
            "Owners should take their cats to the vet once a year.\n\n"
            "The stock market fell sharply today. Investors are worried about rising "
            "inflation. Analysts expect interest rates to rise again next month. "
            "Bond prices also dropped as a result.\n\n"
            "Regular exercise improves heart health. Doctors recommend at least thirty "
            "minutes of walking a day. Stretching before a workout helps prevent injuries.",
            encoding="utf-8",
        )

    index_folder()
    search("Why are investors nervous?")
    search("How often should a cat see a vet?")
