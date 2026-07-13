import fitz  # PyMuPDF

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        text += f"\n\n===== PAGE {page_num + 1} =====\n\n"
        text += page_text

    return text


if __name__ == "__main__":
    pdf_path = r"C:\projects\ResearchMind-AI\data\research_papers\computer vision.pdf"
    
    text = extract_text(pdf_path)

    print("\n===== FULL EXTRACTION PREVIEW =====\n")
    print(f"TOTAL CHARACTERS: {len(text)}")

    # page verification (INSIDE main block)
    for i in range(1, 12):
        if f"===== PAGE {i} =====" in text:
            print(f"Page {i} extracted ✔")
        else:
            print(f"Page {i} missing ❌")