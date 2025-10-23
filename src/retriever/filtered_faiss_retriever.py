# src/retriever/filtered_faiss_retriever.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import get_config
from src.utils.query_classifier import QueryClassifier
from pathlib import Path

class FilteredFaissRetriever:
    """
    FAISS retriever with metadata filtering.
    Correctly reads text and summary from PKL file.
    """
    
    def __init__(self):
        cfg = get_config()
        index_path = cfg["wiki_index_path"]
        # Use ORIGINAL metadata, not enhanced (if enhanced doesn't have text/summary)
        metadata_path = cfg.get("wiki_metadata_path")  # Original pkl
        model_name = cfg["embedding_model"]
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            self.id_map = checkpoint_data['id_map']
        
        # Verify metadata has text and summary
        sample_key = list(self.id_map.keys())[0]
        sample_doc = self.id_map[sample_key]
        print(f"   Metadata check - sample document has:")
        print(f"   - text: {'text' in sample_doc} (length: {len(sample_doc.get('text', ''))})")
        print(f"   - summary: {'summary' in sample_doc} (length: {len(sample_doc.get('summary', ''))})")
        
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        # Query classifier
        self.classifier = QueryClassifier()
        
        # Load domain index from enhanced metadata if available
        enhanced_path = cfg.get("wiki_metadata_enhanced")
        if enhanced_path and Path(enhanced_path).exists():
            print(f"Loading domain index from {enhanced_path}...")
            with open(enhanced_path, "rb") as f:
                enhanced_data = pickle.load(f)
                self.domain_index = enhanced_data.get('domain_index', {})
                # Merge domain info into id_map
                enhanced_id_map = enhanced_data.get('id_map', {})
                for idx, meta in self.id_map.items():
                    if idx in enhanced_id_map:
                        # Add domain info but keep original text/summary
                        meta['primary_domain'] = enhanced_id_map[idx].get('primary_domain', 'general')
                        meta['secondary_domains'] = enhanced_id_map[idx].get('secondary_domains', [])
        else:
            self.domain_index = {}
            print("   No enhanced metadata found, filtering disabled")
        
        print(f"✅ Filtered FAISS index ready: {self.index.ntotal:,} vectors")
        if self.domain_index:
            print(f"   Domains available: {list(self.domain_index.keys())}")
    
    def retrieve(self, query: str, top_k: int = 15, use_filtering: bool = True):
        """
        Retrieve with optional domain filtering.
        Returns documents with actual text and summary.
        """
        # Classify query to get relevant domains
        if use_filtering and self.domain_index:
            domains = self.classifier.classify_query(query, top_k=2)
            print(f"  🎯 Query domains: {domains}")
            
            # Get candidate indices from relevant domains
            candidate_indices = set()
            for domain, confidence in domains:
                if domain in self.domain_index:
                    indices = self.domain_index[domain]
                    candidate_indices.update(indices)
                    print(f"     {domain}: {len(indices):,} candidates")
            
            # Add "general" domain as fallback
            if "general" in self.domain_index:
                candidate_indices.update(self.domain_index["general"][:1000])
            
            candidate_indices = sorted(list(candidate_indices))
            print(f"  📊 Total candidates after filtering: {len(candidate_indices):,}")
        else:
            candidate_indices = None
        
        # Encode query
        qvec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        
        if candidate_indices and len(candidate_indices) < self.index.ntotal * 0.5:
            # Create filtered index for search
            filtered_embeddings = np.array([
                self.index.reconstruct(int(idx)) 
                for idx in candidate_indices 
                if idx < self.index.ntotal
            ], dtype='float32')
            
            # Create temporary index
            temp_index = faiss.IndexFlatIP(self.index.d)
            temp_index.add(filtered_embeddings)
            
            # Search in filtered space
            D, I = temp_index.search(qvec, min(top_k, len(filtered_embeddings)))
            
            # Map back to original indices
            I_mapped = np.array([[candidate_indices[i] for i in I[0]]])
            D, I = D, I_mapped
        else:
            # Search full index
            D, I = self.index.search(qvec, top_k)
        
        # Build results with actual text and summary
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            
            idx_str = str(idx)
            if idx_str not in self.id_map:
                continue
            
            meta = self.id_map[idx_str]
            
            # Extract text and summary DIRECTLY from metadata
            text = meta.get('text', '')
            summary = meta.get('summary', '')
            
            results.append({
                "id": idx_str,
                "doc_idx": int(idx),
                "score": float(score),
                "text": text,  # Full text
                "summary": summary,  # Full summary
                "meta": {
                    "title": meta.get('title', ''),
                    "url": meta.get('url', ''),
                    "categories": meta.get('categories', ''),
                    "primary_domain": meta.get('primary_domain', 'general'),
                    "text_length": len(text),
                    "summary_length": len(summary),
                    "text": text,  # Keep for backward compatibility
                    "summary": summary
                }
            })
        
        return results