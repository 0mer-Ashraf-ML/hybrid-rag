# src/retriever/faiss_retriever.py
import faiss
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer
from src.utils.config import get_config

class FaissRetriever:
    def __init__(self):
        cfg = get_config()
        index_path = cfg["wiki_index_path"]
        metadata_path = cfg["wiki_metadata_path"]
        docs_path = cfg["wiki_docs_json"]
        model_name = cfg["embedding_model"]
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            self.id_map = checkpoint_data['id_map']
        
        print(f"Loading documents from {docs_path}...")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        print(f"✅ FAISS index ready: {self.index.ntotal:,} vectors")

    def retrieve(self, query: str, top_k: int = 15):
        """
        Returns list of dicts: {id, doc_idx, score, text, meta}
        score: FAISS similarity (inner product)
        """
        qvec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        
        D, I = self.index.search(qvec, top_k)
        
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.docs):
                continue
            
            doc = self.docs[idx]
            results.append({
                "id": doc.get("id", str(idx)),
                "doc_idx": int(idx),
                "score": float(score),
                "text": doc["text"],
                "meta": doc
            })
        return results