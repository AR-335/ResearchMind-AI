# ResearchMind-AI

ResearchMind-AI is a local evidence-grounded PDF question-answering system built with Streamlit, FAISS, BM25, sentence-transformer embeddings, CrossEncoder reranking, sentence-level evidence extraction, and Ollama.

The system allows users to upload a PDF, ask questions, receive grounded answers, and view source-page references.

---

## Features

- Upload and ask questions about PDF documents
- Local LLM answer generation using Ollama
- Hybrid retrieval using FAISS semantic search and BM25 keyword search
- CrossEncoder reranking for better chunk selection
- Sentence-level evidence extraction
- Clean source-page references
- Admin/debug mode with chunks, evidence, scores, and support ratio
- Streamlit web interface
- Automated evaluation framework

---

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- FAISS
- BM25
- SentenceTransformers
- CrossEncoder reranking
- NLTK
- Ollama
- Llama 3.2

---

## Architecture

```text
PDF Upload
   |
   v
Text Extraction
   |
   v
Chunking with Page Tracking
   |
   v
Hybrid Retrieval
   |---- FAISS Semantic Search
   |---- BM25 Keyword Search
   |
   v
CrossEncoder Reranking
   |
   v
Sentence-Level Evidence Extraction
   |
   v
Grounded LLM Answer Generation
   |
   v
Answer + Source Pages
```

---

## Project Structure

```text
ResearchMind-AI/
│
├── src/
│   ├── app.py
│   ├── qa_bot.py
│   ├── evaluate.py
│   ├── extract_text.py
│   ├── chunk_text.py
│   └── evidence.py
│
├── evaluation/
│   ├── test_questions.json
│   └── results.csv
│
├── data/
│   ├── research_papers/
│   ├── business_documents/
│   └── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How It Works

ResearchMind-AI extracts text from a PDF and keeps track of page numbers. The extracted text is split into chunks for retrieval.

For each user question, the system retrieves candidate chunks using:

1. FAISS semantic search
2. BM25 keyword search

The retrieved chunks are reranked using a CrossEncoder model. Then the most relevant evidence sentences are extracted and sent to the local LLM.

This helps reduce hallucination because the LLM answers only from selected document evidence.

---

## User Mode vs Admin Mode

Normal user mode shows:

- Clean answer
- Source pages

Admin mode shows:

- Retrieved chunks
- Chunk scores
- Evidence sentences
- Evidence scores
- Answer support ratio

---

## Evaluation

The project includes an evaluation framework in:

```text
src/evaluate.py
```

The evaluation checks:

- Required answer terms
- Expected source-page overlap
- Evidence support ratio

Current evaluation result:

```text
Passed: 3/3
Needs review: 0/3
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

```bash
ollama pull llama3.2
ollama serve
```

### 4. Run the Streamlit app

```bash
streamlit run src/app.py
```

Then open:

```text
http://localhost:8501
```

---

## Running Evaluation

Place your evaluation PDF locally at:

```text
data/research_papers/computer vision.pdf
```

Then run:

```bash
python src/evaluate.py
```

Results are saved to:

```text
evaluation/results.csv
```

---

## Example Questions

```text
What is the main argument or central claim of this research paper?
```

```text
What problem does this paper identify regarding human data and surveillance?
```

```text
What changes were observed from the 1990s to the 2010s?
```

```text
What methods did the authors use in this study?
```

---

## Limitations

- Processes one PDF at a time
- Requires Ollama to run locally
- Source references are page-level
- Evaluation uses a manually prepared test set
- Large PDFs may take longer to process

---

## Future Improvements

- Multi-PDF retrieval
- Evidence highlighting inside PDF pages
- Persistent vector database
- Larger evaluation test set
- Docker deployment
- Export answers with citations

---

## Portfolio Summary

This project demonstrates a complete local RAG workflow:

- PDF ingestion
- Chunking
- Hybrid retrieval
- Reranking
- Evidence extraction
- Grounded answer generation
- Source attribution
- Streamlit UI
- Evaluation workflow
