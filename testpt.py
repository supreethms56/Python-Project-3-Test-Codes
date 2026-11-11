# pip install sentence-transformers numpy
import re
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(text, threshold=0.5, model_name="all-MiniLM-L6-v2"):
    # 1. Split the text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return []

    # 2. Turn each sentence into an embedding
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # 3. Walk through the sentences and group them
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # 4. Compare this sentence with the previous one
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))

        # 5. If the topic changed, close the chunk and start a new one
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # 6. Add the last chunk
    chunks.append(" ".join(current_chunk))
    return chunks


# Example
text = (
    "The stock market fell sharply today. Investors are worried about inflation. "
    "Cats are popular pets. They are independent and love to nap. "
    "Interest rates may rise again next month."
    "Many cats enjoy sitting in sunny spots. "
)

for i, chunk in enumerate(semantic_chunk(text, threshold=0.3)):
    print(i, chunk)