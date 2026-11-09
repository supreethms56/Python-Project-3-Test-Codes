def chunk_text(text, chunk_size=200, overlap=50):
    # 1. Check the settings
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # 2. Break the text into words
    words = text.split()

    # 3. Work out how far to move forward each time
    step = chunk_size - overlap

    chunks = []
    # 4. Slide a window across the words
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

        # 5. Stop once the window reaches the end
        if start + chunk_size >= len(words):
            break

    return chunks


# Example
text = "one two three four five six seven eight nine ten, Eleven "
for i, c in enumerate(chunk_text(text, chunk_size=4, overlap=1)):
    print(i, c)