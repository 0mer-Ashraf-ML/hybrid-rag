import os
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
PDF_FOLDER = "data/pdfs"
OUT_DIR = "data/index"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 100


def load_pdf_text(pdf_path):
    """Extract text per page from PDF."""
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            texts.append((i + 1, text.strip()))
        except Exception as e:
            print(f"[WARN] Could not read page {i+1} in {pdf_path}: {e}")
    return texts

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_index():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)

    all_docs = []
    embeddings = []

    doc_id = 0
    for fname in os.listdir(PDF_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_FOLDER, fname)
        print(f"Processing {fname} ...")
        pages = load_pdf_text(pdf_path)
        for page_num, text in pages:
            if not text.strip():
                continue
            chunks = chunk_text(text)
            for ci, chunk in enumerate(chunks):
                doc = {
                    "id": f"{doc_id}",
                    "source": fname,
                    "page": page_num,
                    "chunk_index": ci,
                    "text": chunk,
                    "created_at": datetime.utcnow().isoformat()
                }
                all_docs.append(doc)
                emb = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
                embeddings.append(emb)
                doc_id += 1

    
    docs_path = os.path.join(OUT_DIR, "docs.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    
    embeddings = np.array(embeddings, dtype="float32")
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)  
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))

    print(f"[DONE] Indexed {len(all_docs)} chunks from {len(os.listdir(PDF_FOLDER))} PDFs.")
    print(f"Saved to {OUT_DIR}/ (docs.json, embeddings.npy, faiss.index)")

if __name__ == "__main__":
    build_index()
