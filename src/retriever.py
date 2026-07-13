import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --------------------------------------------------
# MODELS
# --------------------------------------------------
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# --------------------------------------------------
# BUILD RETRIEVAL SYSTEM
# --------------------------------------------------
def build_system(chunks):
    """
    Build FAISS index and BM25 index.
    """

    texts = [c["text"] for c in chunks]

    # Sentence embeddings
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # FAISS (cosine similarity using inner product)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return index, bm25


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------
def retrieve_candidates(query, index, bm25, chunks, k=10):
    """
    Retrieve candidates using semantic search + BM25.
    """

    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    _, semantic_ids = index.search(q_vec, k)

    bm25_scores = bm25.get_scores(query.lower().split())
    keyword_ids = np.argsort(bm25_scores)[-k:]

    candidates = sorted(
        set(semantic_ids[0]).union(set(keyword_ids))
    )

    candidates = [
        i for i in candidates
        if 0 <= i < len(chunks)
    ]

    return candidates


# --------------------------------------------------
# RERANK
# --------------------------------------------------
def rerank(query, candidates, chunks, top_k=5):
    """
    Rerank retrieved chunks using CrossEncoder.
    Returns chunk metadata instead of plain text.
    """

    if len(candidates) == 0:
        return []

    pairs = [
        (query, chunks[i]["text"])
        for i in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:

        results.append({
            "chunk_id": chunks[idx]["id"],
            "page": chunks[idx]["page"],
            "score": float(score),
            "text": chunks[idx]["text"]
        })

    return results