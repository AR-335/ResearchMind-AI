import numpy as np
import faiss
import nltk
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from extract_text import extract_text
from chunk_text import chunk_text

nltk.download("punkt")

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# -----------------------------
# MODELS
# -----------------------------
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# -----------------------------
# LLM CALL
# -----------------------------
def ask_llama(context, question):

    prompt = f"""
You are a strict document QA assistant.

RULES:
- Use ONLY the given context
- Do NOT use outside knowledge
- If answer is not in context, say:
"I couldn't find this in the document."
- Be concise and accurate

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
    )

    return response.json()["response"]


# -----------------------------
# BUILD SYSTEM
# -----------------------------
def build_system(chunks):

    texts = [c["text"] for c in chunks]

    # ---- FAISS (semantic search)
    vectors = embedder.encode(texts)
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # ---- BM25 (keyword search)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return index, bm25


# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
def retrieve_candidates(query, index, bm25, chunks, k=10):

    # semantic search
    q_vec = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)

    _, sem_idx = index.search(q_vec, k)

    # keyword search
    bm25_scores = bm25.get_scores(query.lower().split())
    key_idx = np.argsort(bm25_scores)[-k:]

    # merge candidates
    candidates = set(sem_idx[0]).union(set(key_idx))

    return list(candidates)


# -----------------------------
# RERANK
# -----------------------------
def rerank(query, candidates, chunks, top_k=5):

    pairs = [(query, chunks[i]["text"]) for i in candidates]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    top_chunks = [chunks[i]["text"] for i, _ in ranked[:top_k]]

    return "\n\n".join(top_chunks)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("\n📄 Loading PDF...")
    pdf_path = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    print("🔎 Building system...")
    index, bm25 = build_system(chunks)

    print(f"\n✅ Ready! Total chunks: {len(chunks)}")
    print("\n🤖 ChatPDF Ready!\n")

    while True:

        query = input("Ask: ")

        if query.lower() == "exit":
            break

        # STEP 1: retrieve candidates
        candidates = retrieve_candidates(query, index, bm25, chunks)

        # STEP 2: rerank best context
        context = rerank(query, candidates, chunks)

        # DEBUG (important for learning)
        print("\n🔎 CONTEXT PREVIEW:\n")
        print(context[:1000])

        # STEP 3: generate answer
        answer = ask_llama(context, query)

        print("\n==============================")
        print("ANSWER")
        print("==============================")
        print(answer)
        print()