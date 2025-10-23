# src/retriever/wiki_faiss_retriever.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import get_config

class WikiFaissRetriever:
    def __init__(self):
        cfg = get_config()
        self.index_path = cfg["wiki_index_path"]
        self.metadata_path = cfg["wiki_metadata_path"]
        self.model_name = cfg["embedding_model"]
        
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            self.id_map = checkpoint_data['id_map']
        
        print(f"Loading embedding model: {self.model_name}...")
        self.embedder = SentenceTransformer(self.model_name)
        
        print(f"✅ Loaded {self.index.ntotal:,} vectors with {len(self.id_map):,} metadata entries")

    def retrieve(self, query: str, top_k: int = 15):
        """
        Returns list of dicts: {id, doc_idx, score, text, meta}
        """
        # Encode query
        qvec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        
        # Search
        D, I = self.index.search(qvec, top_k)
        
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or str(idx) not in self.id_map:
                continue
            
            metadata = self.id_map[str(idx)]
            
            # Combine text and summary for context
            text = metadata.get('text', '')
            summary = metadata.get('summary', '')
            combined_text = f"{summary}\n\n{text}" if summary else text
            
            results.append({
                "id": metadata.get('id', str(idx)),
                "doc_idx": int(idx),
                "score": float(score),
                "text": combined_text[:2000],  # Limit text length
                "meta": {
                    "title": metadata.get('title', ''),
                    "url": metadata.get('url', ''),
                    "categories": metadata.get('categories', ''),
                    "text_length": metadata.get('text_length', 0),
                    "summary_length": metadata.get('summary_length', 0),
                }
            })
        
        return results