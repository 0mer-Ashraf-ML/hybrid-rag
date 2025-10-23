# src/retriever/filtered_bm25_retriever.py
import pickle
from rank_bm25 import BM25Okapi
from src.utils.config import get_config
from src.utils.model_cache import get_cached_query_classifier
from pathlib import Path

class FilteredBM25Retriever:
    """
    BM25 retriever with metadata filtering.
    Uses cached query classifier for efficiency.
    Shared between standard and ultra modes.
    """
    
    def __init__(self):
        cfg = get_config()
        # Use ORIGINAL metadata
        metadata_path = cfg.get("wiki_metadata_path")
        
        print(f"Loading metadata for BM25 from {metadata_path}...")
        with open(metadata_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            self.id_map = checkpoint_data['id_map']
        
        # Verify
        sample_key = list(self.id_map.keys())[0]
        sample_doc = self.id_map[sample_key]
        print(f"   Metadata check - sample has text: {'text' in sample_doc}")
        
        # Use cached query classifier (shared across ALL retrievers)
        self.classifier = get_cached_query_classifier()
        
        # Load domain index from enhanced metadata
        enhanced_path = cfg.get("wiki_metadata_enhanced")
        if enhanced_path and Path(enhanced_path).exists():
            with open(enhanced_path, "rb") as f:
                enhanced_data = pickle.load(f)
                self.domain_index = enhanced_data.get('domain_index', {})
                # Merge domain info
                enhanced_id_map = enhanced_data.get('id_map', {})
                for idx, meta in self.id_map.items():
                    if idx in enhanced_id_map:
                        meta['primary_domain'] = enhanced_id_map[idx].get('primary_domain', 'general')
        else:
            self.domain_index = {}
        
        print(f"✅ Filtered BM25 ready with {len(self.id_map):,} documents")
        if self.domain_index:
            print(f"   Domains: {list(self.domain_index.keys())}")
    
    def _build_bm25_for_indices(self, indices: list) -> tuple:
        """Build BM25 index for specific document indices."""
        corpus_tokens = []
        index_map = []
        
        for idx in indices:
            idx_str = str(idx)
            if idx_str in self.id_map:
                meta = self.id_map[idx_str]
                text = meta.get('text', '')
                summary = meta.get('summary', '')
                combined = f"{summary} {text}"
                
                tokens = combined.lower().split()[:400]
                corpus_tokens.append(tokens)
                index_map.append(idx)
        
        bm25 = BM25Okapi(corpus_tokens)
        return bm25, index_map
    
    def retrieve(self, query: str, top_k: int = 30, use_filtering: bool = True):
        """Retrieve with optional domain filtering."""
        if use_filtering and self.domain_index:
            domains = self.classifier.classify_query(query, top_k=2)
            print(f"  🎯 Query domains (BM25): {domains}")
            
            candidate_indices = set()
            for domain, confidence in domains:
                if domain in self.domain_index:
                    indices = self.domain_index[domain]
                    candidate_indices.update(indices)
            
            if "general" in self.domain_index:
                candidate_indices.update(self.domain_index["general"][:1000])
            
            candidate_indices = sorted(list(candidate_indices))
            print(f"  📊 BM25 searching in {len(candidate_indices):,} filtered documents")
        else:
            candidate_indices = [int(idx) for idx in self.id_map.keys()]
        
        # Build BM25
        bm25, index_map = self._build_bm25_for_indices(candidate_indices)
        
        # Search
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        ranked_positions = list(reversed(scores.argsort()))[:top_k]
        
        # Build results with actual text and summary
        results = []
        for pos in ranked_positions:
            doc_idx = index_map[pos]
            idx_str = str(doc_idx)
            
            if idx_str in self.id_map:
                meta = self.id_map[idx_str]
                text = meta.get('text', '')
                summary = meta.get('summary', '')
                
                results.append({
                    "id": idx_str,
                    "doc_idx": doc_idx,
                    "chunk_no": doc_idx,  # Chunk number
                    "page_no": doc_idx,   # Page number
                    "score": float(scores[pos]),
                    "text": text,  # Full text
                    "summary": summary,  # Full summary
                    "meta": {
                        "title": meta.get('title', ''),
                        "url": meta.get('url', ''),
                        "categories": meta.get('categories', ''),
                        "primary_domain": meta.get('primary_domain', 'general'),
                        "text_length": len(text),
                        "summary_length": len(summary),
                        "text": text,
                        "summary": summary,
                        "chunk_no": doc_idx,
                        "page_no": doc_idx
                    }
                })
        
        return results