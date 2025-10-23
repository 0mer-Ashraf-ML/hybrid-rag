import pdfplumber
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a single PDF file.
    Returns a list of dicts with page number and text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append({
                "pdf_name": Path(pdf_path).name,
                "page": i + 1,
                "text": txt
            })
    return pages


def extract_all_pdfs(pdf_folder: str = "data/pdfs") -> List[Dict]:
    """
    Extract text from all PDFs inside the given folder.
    Returns a list of dicts with {pdf_name, page, text}.
    """
    pdf_dir = Path(pdf_folder)
    all_docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Extracting {pdf_file.name} ...")
        pages = extract_text_from_pdf(pdf_file)
        all_docs.extend(pages)

    print(f" Extracted {len(all_docs)} pages from {len(list(pdf_dir.glob('*.pdf')))} PDFs")
    return all_docs


if __name__ == "__main__":
    docs = extract_all_pdfs("data/pdfs")
    print(f"Sample: {docs[0]['pdf_name']} - page {docs[0]['page']}")
    print(docs[0]["text"][:300])
