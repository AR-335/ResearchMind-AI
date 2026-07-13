from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from build_index import build_faiss_index
from chunk_text import chunk_text
from extract_text import extract_text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, index, chunks, k=5):
    # 1. Convert query → vector
    query_vec = model.encode([query]).astype("float32")

    # 2. Search FAISS
    distances, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        results.append(chunks[i]["text"])

    return results


if __name__ == "__main__":

    pdf_path = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"

    # rebuild pipeline (simple version for now)
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    index, embeddings, chunks = build_faiss_index(chunks)

    print("\n🤖 AI PDF Search Ready!")

    while True:
        query = input("\nAsk a question: ")

        if query.lower() in ["exit", "quit"]:
            break

        results = search(query, index, chunks)

        print("\n📌 Top Relevant Chunks:\n")

        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(r[:500])