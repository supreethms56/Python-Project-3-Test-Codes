# def chunk_text(text, chunk_size=200, overlap=50):
#     # 1. Check the settings
#     if overlap >= chunk_size:
#         raise ValueError("overlap must be smaller than chunk_size")

#     # 2. Break the text into words
#     words = text.split()

#     # 3. Work out how far to move forward each time
#     step = chunk_size - overlap

#     chunks = []
#     # 4. Slide a window across the words
#     for start in range(0, len(words), step):
#         chunk_words = words[start:start + chunk_size]
#         chunks.append(" ".join(chunk_words))

#         # 5. Stop once the window reaches the end
#         if start + chunk_size >= len(words):
#             break

#     return chunks


# # Example
# text = "one two three four five six seven eight nine ten, Eleven "
# for i, c in enumerate(chunk_text(text, chunk_size=4, overlap=1)):
#     print(i, c)



import re
import numpy as np
from fastembed import TextEmbedding
from sklearn.cluster import AgglomerativeClustering

# ---------- 1. Load the embedding model (once) ----------
model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


# ---------- 2. Helper functions ----------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s*(?=[A-Z])', text) if s.strip()]


def embed(sentences):
    vecs = np.array(list(model.embed(sentences)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------- 3. Semantic chunking (keeps original order) ----------
def semantic_chunk(text, threshold=0.3):
    sentences = split_sentences(text)
    if not sentences:
        return []
    vecs = embed(sentences)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        print(f"  similarity {i-1}->{i}: {sim:.2f}")
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(" ".join(current))
    return chunks


# ---------- 4. Topic grouping (ignores order) ----------
def topic_chunks(text, distance_threshold=0.7):
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences
    vecs = embed(sentences)

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(vecs)

    groups = {}
    for sentence, label in zip(sentences, labels):
        groups.setdefault(label, []).append(sentence)
    return [" ".join(g) for g in groups.values()]


# ---------- 5. Run it ----------
if __name__ == "__main__":
    text = (
        "Interest rates may rise again next month. "
        "Cats are popular pets. They are independent and love to nap. "
        "The stock market fell sharply today. Investors are worried about inflation. "
        "Many cats enjoy sitting in sunny spots. "
    )

    print("=== Semantic chunks (in order) ===")
    for i, chunk in enumerate(semantic_chunk(text)):
        print(i, chunk)

    print("\n=== Topic groups ===")
    for i, chunk in enumerate(topic_chunks(text)):
        print(i, chunk)