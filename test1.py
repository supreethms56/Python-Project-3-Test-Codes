"""
Semantic chunking with embeddings.

Install once (inside your activated venv):
    python -m pip install fastembed numpy
"""

import re
import numpy as np
from fastembed import TextEmbedding

# A strong, small embedding model (~130 MB, downloaded on first run)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None


def get_model():
    """Load the model only once, the first time it is needed."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


# ---------------------------------------------------------------
# 1. Split text into sentences
# ---------------------------------------------------------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()              # tidy whitespace
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)      # fix "month.Cats"
    parts = re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------
# 2. Turn sentences into normalized embeddings
# ---------------------------------------------------------------
def embed(texts):
    vecs = np.array(list(get_model().embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------------------------------------------------------------
# 3. Semantic chunking
# ---------------------------------------------------------------
def semantic_chunk(
    text,
    buffer=1,          # sentences of context on each side when comparing
    percentile=75,     # cut at the biggest jumps in meaning (higher = fewer cuts)
    threshold=None,    # or set a fixed distance (e.g. 0.3) instead of percentile
    min_words=15,      # don't make chunks smaller than this
    max_words=300,     # never let a chunk grow bigger than this
    verbose=False,
):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # Compare small windows of sentences instead of single sentences,
    # so short sentences like "They love to nap." keep their context.
    windows = [
        " ".join(sentences[max(0, i - buffer): i + buffer + 1])
        for i in range(len(sentences))
    ]
    vecs = embed(windows)

    # Distance between each sentence and the next (0 = same meaning)
    distances = [1 - float(np.dot(vecs[i], vecs[i + 1]))
                 for i in range(len(vecs) - 1)]

    # Decide what counts as a "big jump"
    cut_at = threshold if threshold is not None else np.percentile(distances, percentile)

    if verbose:
        print(f"cut-off distance: {cut_at:.3f}")
        for i, d in enumerate(distances):
            mark = "  <-- CUT" if d > cut_at else ""
            print(f"  {i}->{i+1}: {d:.3f}{mark}")

    # Build the chunks
    chunks, current = [], [sentences[0]]
    current_words = len(sentences[0].split())

    for i, sentence in enumerate(sentences[1:]):
        words = len(sentence.split())
        topic_changed = distances[i] > cut_at
        big_enough = current_words >= min_words
        too_big = current_words + words > max_words

        if (topic_changed and big_enough) or too_big:
            chunks.append(" ".join(current))
            current, current_words = [sentence], words
        else:
            current.append(sentence)
            current_words += words

    chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------
# 4. Try it out
# ---------------------------------------------------------------
if __name__ == "__main__":
    text = """
    Analysts expect interest rates to rise again next month. They are independent and love to nap
    for most of the day. Many cats enjoy sitting in sunny spots near windows.
    Owners should take their cats to the vet once a year.

    The stock market fell sharply today. Investors are worried about rising
    inflation. Cats are popular pets around the world. 
     Stretching before a workout helps prevent injuries.

    Regular exercise improves heart health. Doctors recommend at least thirty
    minutes of walking a day. Bond prices also dropped as a result. 
    """

    chunks = semantic_chunk(text, verbose=True)

    print(f"\n{len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk}\n")