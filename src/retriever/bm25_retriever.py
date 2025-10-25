# src/retriever/bm25_retriever.py
import json
from rank_bm25 import BM25Okapi
from src.utils.config import get_config

class BM25Retriever:
    def __init__(self):
        print('BM25Retriever initialized')
        cfg = get_config()
        docs_path = cfg["wiki_docs_json"]
        
        print(f"Loading documents from {docs_path} for BM25...")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        
        print(f"Tokenizing {len(self.docs)} documents for BM25...")
        self.corpus_tokens = [d["text"].lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"✅ BM25 index ready with {len(self.docs)} documents")

    def retrieve(self, query: str, top_k: int = 30):
        """
        Returns list of dicts: {id, doc_idx, score, text, meta}
        score: BM25 raw score (higher = better)
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
   
        ranked_idxs = list(reversed(scores.argsort()))[:top_k]
        results = []
        for idx in ranked_idxs:
            results.append({
                "id": self.docs[idx].get("id", str(idx)),
                "doc_idx": int(idx),
                "score": float(scores[idx]),
                "text": self.docs[idx]["text"],
                "meta": self.docs[idx]
            })
        return results