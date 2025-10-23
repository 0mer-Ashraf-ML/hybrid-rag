import re
from typing import List
from transformers import AutoTokenizer
def chunk_text_simple(text: str, chunk_size: int = 1500, overlap: int = 300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c.strip()]

def chunk_text_tokenaware(text: str, chunk_tokens: int = 400, overlap_tokens: int = 50, model_name="sentence-transformers/all-mpnet-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    toks = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0].tolist()
    chunks = []
    i = 0
    while i < len(toks):
        chunk_toks = toks[i:i+chunk_tokens]
        chunk_text = tokenizer.decode(chunk_toks, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += chunk_tokens - overlap_tokens
    return chunks
