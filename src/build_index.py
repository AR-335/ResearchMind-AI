from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from chunk_text import chunk_text
from extract_text import extract_text

# 1. Load model (lightweight + good)
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(chunks):
    texts = [c["text"] for c in chunks]

    # 2. Convert text → embeddings
    embeddings = model.encode(texts)

    # 3. Convert to numpy
    embeddings = np.array(embeddings).astype("float32")

    # 4. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # 5. Add vectors to index
    index.add(embeddings)

    return index, embeddings, chunks


if __name__ == "__main__":

    pdf_path = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"

    # Step 1: extract text
    text = extract_text(pdf_path)

    # Step 2: chunk text
    chunks = chunk_text(text)

    print(f"Chunks: {len(chunks)}")

    # Step 3: build index
    index, embeddings, chunks = build_faiss_index(chunks)

    print("\n✅ FAISS index built successfully!")
    print(f"Vector count: {index.ntotal}")