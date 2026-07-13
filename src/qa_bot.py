import re
import numpy as np
import faiss
import nltk
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from extract_text import extract_text
from chunk_text import chunk_text
from evidence import extract_evidence

nltk.download("punkt", quiet=True)

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

PDF_PATH = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"

# True  = show admin/debug details
# False = clean user view only
ADMIN_MODE = True


# -----------------------------
# MODELS
# -----------------------------
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# -----------------------------
# QUESTION TYPE DETECTION
# -----------------------------
def detect_question_type(question):
    """
    Detects question type so the LLM can focus on the correct evidence.
    """

    q = question.lower()

    if any(word in q for word in [
        "main argument",
        "central claim",
        "main claim",
        "central argument",
        "main focus",
        "main idea"
    ]):
        return "main_claim"

    if any(word in q for word in [
        "problem",
        "issue",
        "concern",
        "identify",
        "regarding human data",
        "human data and surveillance"
    ]):
        return "problem"

    if any(word in q for word in [
        "1990s",
        "2010s",
        "changes",
        "changed",
        "change",
        "trend",
        "over time",
        "evolution"
    ]):
        return "trend"

    if any(word in q for word in [
        "method",
        "methods",
        "data",
        "dataset",
        "corpus",
        "analysis",
        "how did the authors"
    ]):
        return "method"

    return "general"


def get_focus_instruction(question_type):
    """
    Gives the LLM focused instructions depending on the question type.
    """

    if question_type == "main_claim":
        return """
FOCUS:
- Answer the central claim of the paper.
- Prioritize evidence about what the authors argue, contend, find, or conclude.
- Focus on computer vision, surveillance, and extraction of human data.
- Do NOT include secondary background details unless they are essential.
"""

    if question_type == "problem":
        return """
FOCUS:
- Explain the problem identified by the paper.
- Prioritize evidence about targeting humans, extracting human body data, public concern, privacy, consent, agency, and surveillance.
- Do NOT focus only on one example if broader evidence is available.
"""

    if question_type == "trend":
        return """
FOCUS:
- Explain the observed change over time.
- Prioritize evidence comparing the 1990s and 2010s.
- Include numbers or trends only if they appear in the evidence.
"""

    if question_type == "method":
        return """
FOCUS:
- Explain the method, dataset, corpus, or analysis approach.
- Prioritize evidence about papers, patents, CVPR, content analysis, and lexicon-based analysis.
"""

    return """
FOCUS:
- Answer directly using the most relevant evidence.
- Avoid adding weak or unrelated details.
"""


# -----------------------------
# LLM CALL
# -----------------------------
def ask_llama(evidence_context, question, retry_mode=False):
    """
    Ask Ollama using focused evidence-grounded prompting.
    """

    question_type = detect_question_type(question)
    focus_instruction = get_focus_instruction(question_type)

    if retry_mode:
        retry_instruction = """
The evidence is relevant to the question.
Give the best supported answer using ONLY the evidence.
Do NOT refuse unless the evidence is completely unrelated.
"""
    else:
        retry_instruction = ""

    prompt = f"""
You are a document-grounded research assistant.

RULES:
- Use ONLY the provided evidence.
- Do NOT use outside knowledge.
- Do NOT invent facts.
- Do NOT include weak or unrelated evidence in the answer.
- You may summarize multiple evidence sentences when the question asks for a main idea, problem, finding, trend, or conclusion.
- If the evidence is completely unrelated to the question, say:
"I couldn't find this in the document."

{focus_instruction}

USER-FACING STYLE:
- Give a direct answer.
- Keep the answer concise and factual.
- Do NOT say "Based on the provided evidence".
- Do NOT mention evidence numbers.
- Do NOT mention chunk IDs.
- Do NOT mention page numbers inside the answer.
- Sources will be shown separately.

{retry_instruction}

EVIDENCE:
{evidence_context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.RequestException:
        return "Ollama is not running. Please start Ollama using: ollama serve"


# -----------------------------
# BUILD SYSTEM
# -----------------------------
def build_system(chunks):
    texts = [c["text"] for c in chunks]

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return index, bm25


# -----------------------------
# RETRIEVE CANDIDATES
# -----------------------------
def retrieve_candidates(query, index, bm25, chunks, k=12):
    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q_vec)

    sem_scores, sem_idx = index.search(q_vec, k)

    bm25_scores = bm25.get_scores(query.lower().split())
    key_idx = np.argsort(bm25_scores)[-k:]

    candidates = {}

    # Semantic search contribution
    for score, idx in zip(sem_scores[0], sem_idx[0]):
        if idx == -1:
            continue

        candidates[idx] = candidates.get(idx, 0) + float(score) * 0.7

    # Keyword search contribution
    for idx in key_idx:
        if idx < 0 or idx >= len(chunks):
            continue

        candidates[idx] = candidates.get(idx, 0) + float(bm25_scores[idx]) * 0.3

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [idx for idx, _ in ranked]


# -----------------------------
# RERANK CHUNKS
# -----------------------------
def rerank_chunks(query, candidates, chunks, top_k=5):
    if len(candidates) == 0:
        return []

    valid_candidates = [
        i for i in candidates
        if 0 <= i < len(chunks)
    ]

    if len(valid_candidates) == 0:
        return []

    pairs = [
        (query, chunks[i]["text"])
        for i in valid_candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(valid_candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    retrieved_chunks = []

    for idx, score in ranked[:top_k]:
        retrieved_chunks.append({
            "id": chunks[idx]["id"],
            "chunk_id": chunks[idx]["id"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "chunk_score": float(score)
        })

    return retrieved_chunks


# -----------------------------
# EVIDENCE CONTEXT BUILDER
# -----------------------------
def build_evidence_context(evidence):
    """
    Builds clean evidence text for the LLM.

    Important:
    We do NOT label sentences as Evidence 1, Evidence 2, etc.
    If we give labels, the LLM may repeat those labels in the answer.
    """

    lines = []

    for e in evidence:
        lines.append(e["text"])

    return "\n".join(lines)


# -----------------------------
# USER ANSWER CLEANUP
# -----------------------------
def clean_answer(answer):
    """
    Removes common LLM phrases that are not good for user-facing output.
    """

    cleaned = answer.strip()

    unwanted_prefixes = [
        r"^Based on the provided evidence,\s*",
        r"^Based on the evidence,\s*",
        r"^According to the provided evidence,\s*",
        r"^According to the evidence,\s*",
        r"^From the provided evidence,\s*",
        r"^The evidence suggests that\s*",
        r"^The provided evidence suggests that\s*",
    ]

    for pattern in unwanted_prefixes:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    # Remove evidence-number references if they appear
    cleaned = re.sub(
        r"\bEvidence\s+\d+\b",
        "the document",
        cleaned,
        flags=re.IGNORECASE
    )

    # Remove phrases like "This is evident from..."
    cleaned = re.sub(
        r"\s*This is evident from.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )

    return cleaned.strip()


# -----------------------------
# SIMPLE SUPPORT CHECK FOR ADMIN
# -----------------------------
STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "with", "that", "this", "it", "as", "by", "from",
    "at", "be", "been", "being", "into", "about", "according", "document",
    "paper", "research"
}


def content_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    }


def support_ratio(answer, evidence_text):
    if "couldn't find" in answer.lower():
        return 0.0

    answer_words = content_words(answer)
    evidence_words = content_words(evidence_text)

    if len(answer_words) == 0:
        return 0.0

    overlap = answer_words.intersection(evidence_words)

    return len(overlap) / len(answer_words)


def answer_is_no_answer(answer):
    return "couldn't find" in answer.lower()


# -----------------------------
# DISPLAY HELPERS
# -----------------------------
def show_user_output(answer, evidence):
    pages = sorted(set(e["page"] for e in evidence))

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(answer)

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    for page in pages:
        print(f"Page {page}")


def show_admin_output(retrieved_chunks, evidence, answer, evidence_text):
    ratio = support_ratio(answer, evidence_text)

    print("\n==============================")
    print("ADMIN DEBUG INFO")
    print("==============================")

    if answer_is_no_answer(answer):
        print("Grounding status: NO ANSWER RETURNED")
    elif ratio >= 0.35:
        print("Grounding status: SUPPORTED")
    else:
        print("Grounding status: NEEDS REVIEW")

    print(f"Answer support ratio: {ratio:.2f}")

    print("\nTop retrieved chunks:")
    for chunk in retrieved_chunks:
        print(
            f"Page {chunk['page']} | "
            f"Chunk {chunk['chunk_id']} | "
            f"Chunk Score {chunk['chunk_score']:.2f}"
        )

    print("\nTop evidence sentences:")
    for e in evidence:
        print(
            f"- Page {e['page']} | "
            f"Chunk {e['chunk_id']} | "
            f"Evidence Score {e['score']:.2f}"
        )
        print(f"  {e['text']}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("\n📄 Loading PDF...")

    text = extract_text(PDF_PATH)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("🔎 Building retrieval system...")
    index, bm25 = build_system(chunks)

    print(f"\n✅ Ready! Total chunks: {len(chunks)}")
    print("\n🤖 Focused Evidence-grounded ChatPDF Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() == "exit":
            break

        # Step 1: retrieve possible chunks
        candidates = retrieve_candidates(
            query,
            index,
            bm25,
            chunks,
            k=12
        )

        # Step 2: rerank chunks
        retrieved_chunks = rerank_chunks(
            query,
            candidates,
            chunks,
            top_k=5
        )

        if len(retrieved_chunks) == 0:
            print("\nI couldn't find this in the document.\n")
            continue

        # Step 3: extract exact evidence sentences
        evidence = extract_evidence(
            query,
            retrieved_chunks,
            reranker,
            top_k=8
        )

        if len(evidence) == 0:
            print("\nI couldn't find this in the document.\n")
            continue

        # Step 4: build clean evidence context
        evidence_context = build_evidence_context(evidence)

        # Step 5: ask LLM
        answer = ask_llama(
            evidence_context,
            query,
            retry_mode=False
        )

        # Step 6: retry once if model refused despite evidence existing
        if answer_is_no_answer(answer):
            answer = ask_llama(
                evidence_context,
                query,
                retry_mode=True
            )

        # Step 7: clean user-facing answer
        answer = clean_answer(answer)

        # Step 8: show user output
        show_user_output(answer, evidence)

        # Step 9: show admin/debug output
        if ADMIN_MODE:
            show_admin_output(
                retrieved_chunks,
                evidence,
                answer,
                evidence_context
            )

        print()