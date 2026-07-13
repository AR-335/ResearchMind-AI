import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def chunk_text(text, max_words=180, overlap=2):

    chunks = []

    paragraphs = re.split(r"\n\s*\n", text)

    buffer = []
    chunk_id = 1
    current_page = 1

    def flush():
        nonlocal chunk_id, buffer

        if len(buffer) == 0:
            return

        chunks.append({
            "id": chunk_id,
            "page": current_page,
            "text": " ".join(buffer).strip()
        })

        chunk_id += 1

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        # detect page safely
        page_match = re.search(r"PAGE\s+(\d+)", para)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        sentences = sent_tokenize(para)

        for sent in sentences:

            buffer.append(sent)

            word_count = len(" ".join(buffer).split())

            if word_count >= max_words:

                flush()

                # keep overlap
                buffer = buffer[-overlap:]

    flush()

    return chunks


if __name__ == "__main__":

    from extract_text import extract_text

    pdf_path = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"

    text = extract_text(pdf_path)

    chunks = chunk_text(text)

    print(f"\nTOTAL CHUNKS CREATED: {len(chunks)}\n")

    for c in chunks[:5]:

        print("=" * 50)
        print(f"ID: {c['id']}")
        print(f"Page: {c['page']}")
        print(c["text"][:400])
        print()