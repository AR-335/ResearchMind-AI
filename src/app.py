import os
import re
import tempfile

import faiss
import nltk
import numpy as np
import requests
import streamlit as st

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
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"
MODEL_NAME = "llama3.2"


# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ResearchMind-AI",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# UI STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .document-box {
        padding: 1rem;
        border-radius: 0.8rem;
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .document-title {
        font-weight: 700;
        font-size: 1rem;
    }

    .document-meta {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .source-badge {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border-radius: 999px;
        background-color: rgba(99, 102, 241, 0.16);
        border: 1px solid rgba(129, 140, 248, 0.55);
        font-size: 0.88rem;
        color: #c7d2fe;
    }

    .hint-box {
        padding: 0.8rem 1rem;
        border-radius: 0.7rem;
        background-color: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.35);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return embedder, reranker


embedder, reranker = load_models()


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
# OLLAMA HEALTH CHECK
# -----------------------------
def check_ollama():
    try:
        response = requests.get(
            OLLAMA_HEALTH_URL,
            timeout=3
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


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
# BUILD VECTOR + BM25 SYSTEM
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
# HYBRID RETRIEVAL
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
# CROSS-ENCODER CHUNK RERANKING
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


# -----------------------------
# PDF PROCESSING
# -----------------------------
def process_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    text = extract_text(temp_path)
    chunks = chunk_text(text)

    try:
        os.remove(temp_path)
    except OSError:
        pass

    index, bm25 = build_system(chunks)

    return {
        "filename": uploaded_file.name,
        "chunks": chunks,
        "index": index,
        "bm25": bm25
    }


# -----------------------------
# FULL QA PIPELINE
# -----------------------------
def answer_question(question, document_data):
    chunks = document_data["chunks"]
    index = document_data["index"]
    bm25 = document_data["bm25"]

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
# UI HELPER FUNCTIONS
# -----------------------------
def render_sources(pages):
    if not pages:
        return

    badge_html = ""

    for page in pages:
        badge_html += f"<span class='source-badge'>Page {page}</span>"

    st.markdown("**Sources**")
    st.markdown(badge_html, unsafe_allow_html=True)


def render_admin_debug(debug):
    with st.expander("🔧 Admin debug details"):
        ratio = debug["support_ratio"]

        if ratio >= 0.35:
            st.success(f"Grounding status: Supported | Support ratio: {ratio:.2f}")
        elif ratio == 0.0:
            st.warning(f"Grounding status: No answer / no support | Support ratio: {ratio:.2f}")
        else:
            st.warning(f"Grounding status: Needs review | Support ratio: {ratio:.2f}")

        st.markdown("### Retrieved chunks")

        for chunk in debug["retrieved_chunks"]:
            st.write(
                f"Page {chunk['page']} | "
                f"Chunk {chunk['chunk_id']} | "
                f"Score {chunk['chunk_score']:.2f}"
            )

        st.markdown("### Evidence sentences")

        for e in debug["evidence"]:
            st.write(
                f"Page {e['page']} | "
                f"Chunk {e['chunk_id']} | "
                f"Score {e['score']:.2f}"
            )
            st.info(e["text"])


def add_user_message(content):
    st.session_state.messages.append({
        "role": "user",
        "content": content
    })


def add_assistant_message(result):
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "pages": result["pages"],
        "debug": {
            "support_ratio": result["support_ratio"],
            "retrieved_chunks": result["retrieved_chunks"],
            "evidence": result["evidence"]
        }
    })


# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📄 ResearchMind-AI")

    st.write(
        "Upload a PDF and ask grounded questions with source-page references."
    )

    st.markdown("---")

    admin_mode = st.toggle(
        "Admin mode",
        value=False,
        help="Show retrieved chunks, evidence sentences, and support ratio."
    )

    with st.expander("System status", expanded=False):
        if check_ollama():
            st.success("Ollama is running")
        else:
            st.error("Ollama is not running")
            st.write("Start Ollama in a terminal:")
            st.code("ollama serve", language="bash")

        st.write(f"Model: `{MODEL_NAME}`")

    st.markdown("---")

    st.subheader("Sample questions")

    sample_questions = [
        "What is the main argument or central claim of this research paper?",
        "What problem does this paper identify regarding human data and surveillance?",
        "What changes were observed from the 1990s to the 2010s?",
        "What methods did the authors use in this study?"
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown(
    "<div class='main-title'>ResearchMind-AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Evidence-grounded PDF question answering with clean answers and source-page references.</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='hint-box'>
        <strong>How it works:</strong>
        upload a PDF, ask a question, and the assistant answers only from retrieved evidence in the document.
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.uploaded_file_key != file_key:
        with st.spinner("Processing and indexing PDF..."):
            st.session_state.document_data = process_uploaded_pdf(uploaded_file)
            st.session_state.uploaded_file_key = file_key
            st.session_state.messages = []
            st.session_state.pending_question = None

        st.success("Document processed successfully.")

else:
    st.info("Upload a PDF to start asking questions.")


# -----------------------------
# DOCUMENT STATUS
# -----------------------------
if st.session_state.document_data is not None:
    filename = st.session_state.document_data["filename"]
    chunk_count = len(st.session_state.document_data["chunks"])

    st.markdown(
        f"""
        <div class='document-box'>
            <div class='document-title'>✅ Document ready</div>
            <div class='document-meta'>{filename} · {chunk_count} chunks indexed</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# CHAT HISTORY
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("pages", []))

            if admin_mode and message.get("debug"):
                render_admin_debug(message["debug"])


# -----------------------------
# INPUT HANDLING
# -----------------------------
chat_question = st.chat_input("Ask a question about the PDF")

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif chat_question:
    question = chat_question


# -----------------------------
# RUN QA
# -----------------------------
if question:
    if st.session_state.document_data is None:
        st.warning("Please upload a PDF first.")

    else:
        add_user_message(question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Reading the document..."):
                result = answer_question(
                    question,
                    st.session_state.document_data
                )

            st.markdown(result["answer"])
            render_sources(result["pages"])

            if admin_mode:
                render_admin_debug({
                    "support_ratio": result["support_ratio"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "evidence": result["evidence"]
                })

        add_assistant_message(result)