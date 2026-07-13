import csv
import json
import re
from pathlib import Path

import faiss
import nltk
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from extract_text import extract_text
from chunk_text import chunk_text
from evidence import extract_evidence


# -----------------------------
# BASIC SETUP
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "data" / "research_papers" / "computer vision.pdf"
TEST_FILE = PROJECT_ROOT / "evaluation" / "test_questions.json"
RESULT_FILE = PROJECT_ROOT / "evaluation" / "results.csv"


# -----------------------------
# MODELS
# -----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# -----------------------------
# QUESTION TYPE DETECTION
# -----------------------------
def detect_question_type(question):
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

    for score, idx in zip(sem_scores[0], sem_idx[0]):
        if idx == -1:
            continue

        candidates[idx] = candidates.get(idx, 0) + float(score) * 0.7

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
# EVIDENCE CONTEXT
# -----------------------------
def build_evidence_context(evidence):
    return "\n".join(e["text"] for e in evidence)


# -----------------------------
# ANSWER CLEANUP
# -----------------------------
def clean_answer(answer):
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

    cleaned = re.sub(
        r"\bEvidence\s+\d+\b",
        "the document",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*This is evident from.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )

    return cleaned.strip()


def answer_is_no_answer(answer):
    return "couldn't find" in answer.lower()


# -----------------------------
# SUPPORT CHECK
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


# -----------------------------
# QA PIPELINE
# -----------------------------
def answer_question(question, chunks, index, bm25):
    candidates = retrieve_candidates(
        question,
        index,
        bm25,
        chunks,
        k=12
    )

    retrieved_chunks = rerank_chunks(
        question,
        candidates,
        chunks,
        top_k=5
    )

    if len(retrieved_chunks) == 0:
        return {
            "answer": "I couldn't find this in the document.",
            "pages": [],
            "evidence": [],
            "retrieved_chunks": [],
            "support_ratio": 0.0
        }

    evidence = extract_evidence(
        question,
        retrieved_chunks,
        reranker,
        top_k=8
    )

    if len(evidence) == 0:
        return {
            "answer": "I couldn't find this in the document.",
            "pages": [],
            "evidence": [],
            "retrieved_chunks": retrieved_chunks,
            "support_ratio": 0.0
        }

    evidence_context = build_evidence_context(evidence)

    answer = ask_llama(
        evidence_context,
        question,
        retry_mode=False
    )

    if answer_is_no_answer(answer):
        answer = ask_llama(
            evidence_context,
            question,
            retry_mode=True
        )

    answer = clean_answer(answer)

    pages = sorted(set(e["page"] for e in evidence))
    ratio = support_ratio(answer, evidence_context)

    return {
        "answer": answer,
        "pages": pages,
        "evidence": evidence,
        "retrieved_chunks": retrieved_chunks,
        "support_ratio": ratio
    }


# -----------------------------
# EVALUATION HELPERS
# -----------------------------
def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def required_term_score(answer, required_terms):
    answer_lower = normalize_text(answer)

    if len(required_terms) == 0:
        return 1.0, []

    matched_terms = []

    for term in required_terms:
        if normalize_text(term) in answer_lower:
            matched_terms.append(term)

    score = len(matched_terms) / len(required_terms)

    return score, matched_terms


def page_overlap_score(actual_pages, expected_pages):
    actual_set = set(actual_pages)
    expected_set = set(expected_pages)

    if len(expected_set) == 0:
        return 1.0

    overlap = actual_set.intersection(expected_set)

    return len(overlap) / len(expected_set)


def decide_pass(term_score, page_score, support):
    """
    Simple rule-based evaluation.

    Pass means:
    - answer contains enough expected key terms
    - retrieved pages overlap with expected source pages
    - answer is reasonably grounded in evidence
    """

    if term_score >= 0.40 and page_score >= 0.25 and support >= 0.35:
        return "PASS"

    return "REVIEW"


def load_tests():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(rows):
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "answer",
        "expected_pages",
        "actual_pages",
        "required_terms",
        "matched_terms",
        "term_score",
        "page_score",
        "support_ratio",
        "result"
    ]

    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_result(row):
    print("\n" + "=" * 80)
    print(f"ID: {row['id']}")
    print(f"Question: {row['question']}")
    print(f"Result: {row['result']}")
    print(f"Term score: {row['term_score']}")
    print(f"Page score: {row['page_score']}")
    print(f"Support ratio: {row['support_ratio']}")
    print(f"Expected pages: {row['expected_pages']}")
    print(f"Actual pages: {row['actual_pages']}")
    print(f"Matched terms: {row['matched_terms']}")
    print("\nAnswer:")
    print(row["answer"])


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("\nStarting ResearchMind-AI evaluation...")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    print(f"\nLoading PDF: {PDF_PATH}")
    text = extract_text(str(PDF_PATH))

    print("Chunking PDF...")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    print("Building retrieval system...")
    index, bm25 = build_system(chunks)

    tests = load_tests()

    print(f"\nTotal evaluation questions: {len(tests)}")

    results = []

    for test in tests:
        question = test["question"]
        expected_pages = test.get("expected_pages", [])
        required_terms = test.get("required_terms", [])

        print("\nRunning:", test["id"])

        result = answer_question(
            question,
            chunks,
            index,
            bm25
        )

        term_score, matched_terms = required_term_score(
            result["answer"],
            required_terms
        )

        page_score = page_overlap_score(
            result["pages"],
            expected_pages
        )

        final_result = decide_pass(
            term_score,
            page_score,
            result["support_ratio"]
        )

        row = {
            "id": test["id"],
            "question": question,
            "answer": result["answer"],
            "expected_pages": expected_pages,
            "actual_pages": result["pages"],
            "required_terms": required_terms,
            "matched_terms": matched_terms,
            "term_score": round(term_score, 2),
            "page_score": round(page_score, 2),
            "support_ratio": round(result["support_ratio"], 2),
            "result": final_result
        }

        results.append(row)

        print_result(row)

    save_results(results)

    passed = sum(1 for r in results if r["result"] == "PASS")
    total = len(results)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total}")
    print(f"Needs review: {total - passed}/{total}")
    print(f"Results saved to: {RESULT_FILE}")