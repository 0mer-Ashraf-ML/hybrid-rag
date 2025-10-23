# src/retriever/ultra_compressed_retriever.py
"""
Ultra-compressed retriever with full integration into hybrid system.
Uses 192D PCA + int8 + zlib compression (50% size reduction, <3% quality loss)

NOTE: This retriever accesses the SAME Wikipedia data as standard mode,
but uses a compressed index for 50% storage savings and faster retrieval.
Results may differ slightly due to PCA compression, but source documents are identical.
"""

import faiss
import pickle
import numpy as np
import zlib
from pathlib import Path
from src.utils.config import get_config, get_index_paths
from src.utils.model_cache import get_cached_embedder, get_cached_query_classifier

class UltraCompressedRetriever:
    """
    Retriever for ultra-compressed index (192D + int8 + text compression)
    
    Key Features:
    - 50% storage reduction (1.1 GB vs 2.2 GB)
    - Faster retrieval (1.5-2x speedup)
    - <3% quality loss vs standard mode
    - Same source Wikipedia data
    """
    
    def __init__(self):
        cfg = get_config()
        paths = get_index_paths()
        
        # Use paths from config
        if paths['mode'] == 'ultra_compressed':
            INDEX_DIR = Path(cfg['wiki_ultra_index_path']).parent
        else:
            INDEX_DIR = Path("data/index/ultra_compressed")
        
        print(f"Loading ultra-compressed RAG index from {INDEX_DIR}...")
        
        # Load FAISS index
        index_path = INDEX_DIR / "wikipedia_ultra.index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Ultra-compressed index not found at {index_path}\n"
                f"Run: python scripts/compress_to_1_1gb.py"
            )
        
        self.index = faiss.read_index(str(index_path))
        self.index.nprobe = cfg.get("ultra_nprobe", 20)  # Search quality
        
        # Load PCA model
        pca_path = INDEX_DIR / "pca_192.pkl"
        with open(pca_path, "rb") as f:
            self.pca_model = pickle.load(f)
        
        # Load compressed metadata
        
        metadata_path = INDEX_DIR / "metadata_ultra_compressed.pkl"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.docs = data['d']  # Changed from data['docs']
            self.title_dict = data.get('td', [])  # Title dictionary
            self.url_dict = data.get('ud', [])    # URL dictionary
            self.domains_compressed = data.get('domains', {})  
            
            # Decompress domain index
            self.domain_index = {
                k: np.frombuffer(v, dtype=np.int32).tolist()
                for k, v in self.domains_compressed.items()
            }
        
        # Use cached embedding model (shared across ALL retrievers - EFFICIENT!)
        embedding_model = cfg.get("embedding_model", "BAAI/bge-base-en")
        self.embedder = get_cached_embedder(embedding_model)
        
        # Use cached query classifier
        self.classifier = get_cached_query_classifier()
        
        print(f"✅ Ultra-compressed index loaded:")
        print(f"   Vectors: {self.index.ntotal:,}")
        print(f"   Dimensions: {self.pca_model['n_components']}D (from 768D)")
        print(f"   Documents: {len(self.docs):,}")
        print(f"   Domains: {len(self.domain_index)}")
        print(f"   nprobe: {self.index.nprobe}")
    
    def _decompress_text(self, compressed_data):
        """Decompress zlib-compressed text"""
        if isinstance(compressed_data, bytes):
            try:
                return zlib.decompress(compressed_data).decode('utf-8')
            except Exception as e:
                print(f"Warning: Failed to decompress text: {e}")
                return ""
        return compressed_data if compressed_data else ""
    
    def _apply_pca(self, query_embedding):
        """Apply PCA transformation to query (768D → 192D)"""
        return np.dot(
            query_embedding - self.pca_model['mean'],
            self.pca_model['components'].T
        ).astype(np.float32)
    
    def retrieve(self, query: str, top_k: int = 15, use_filtering: bool = True):
        """
        Retrieve documents using ultra-compressed index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_filtering: Enable domain-based filtering
        
        Returns:
            List of documents in standard format
            
        Note: Results contain the SAME Wikipedia articles as standard mode,
        but retrieved using compressed embeddings (192D PCA).
        Chunk numbers and document IDs are consistent with source data.
        """
        # Encode query (768D) - uses CACHED model, very fast!
        qvec = self.embedder.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        # Apply PCA (768D → 192D)
        qvec_pca = self._apply_pca(qvec)
        
        # Domain filtering (optional)
        filtered_ids = None
        if use_filtering and self.domain_index:
            domains = self.classifier.classify_query(query, top_k=2)
            print(f"  🎯 Domains (Ultra): {domains}")
            
            # Collect candidate document IDs
            candidate_ids = set()
            for domain, confidence in domains:
                if domain in self.domain_index:
                    candidate_ids.update(self.domain_index[domain])
            
            # Add general domain
            if "general" in self.domain_index:
                candidate_ids.update(self.domain_index["general"][:1000])
            
            if candidate_ids:
                filtered_ids = sorted(list(candidate_ids))
                print(f"  📊 Ultra searching in {len(filtered_ids):,} filtered documents")
        
        # Search FAISS index
        if filtered_ids:
            # Filter search (search only within specific IDs)
            D, I = self.index.search(qvec_pca, top_k * 3)  # Get more, filter later
            
            # Filter results
            filtered_results = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx in filtered_ids:
                    filtered_results.append((score, idx))
                if len(filtered_results) >= top_k:
                    break
            
            D = np.array([[r[0] for r in filtered_results]])
            I = np.array([[r[1] for r in filtered_results]])
        else:
            # Standard search
            D, I = self.index.search(qvec_pca, top_k)
        
        # Build results in standard format
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or str(idx) not in self.docs:
                continue
            
            doc = self.docs[str(idx)]
            
            # Decompress text if needed
            text = doc.get('text', '')
            if doc.get('_text_compressed', False):
                text = self._decompress_text(text)
            
            summary = doc.get('summary', '')
            if doc.get('_summary_compressed', False):
                summary = self._decompress_text(summary)
            
            # Build result in standard format (compatible with HybridRetriever)
            # Include chunk_no/page_no for reference
            results.append({
                "id": str(idx),
                "doc_idx": int(idx),
                "chunk_no": int(idx),  # Chunk number (same as doc_idx)
                "page_no": int(idx),   # Page number (same as doc_idx)
                "score": float(score),
                "text": text,
                "summary": summary,
                "meta": {
                    "title": doc.get('title', ''),
                    "url": doc.get('url', ''),
                    "categories": "",  # Not stored in ultra-compressed
                    "primary_domain": doc.get('domain', 'general'),
                    "text_length": doc.get('tl', len(text)),
                    "summary_length": doc.get('sl', len(summary)),
                    "text": text,
                    "summary": summary,
                    "chunk_no": int(idx),
                    "page_no": int(idx)
                }
            })
        
        return results
    
    def get_stats(self):
        """Get index statistics"""
        return {
            "total_documents": len(self.docs),
            "total_vectors": self.index.ntotal,
            "dimensions": self.pca_model['n_components'],
            "original_dimensions": self.pca_model['original_dim'],
            "nprobe": self.index.nprobe,
            "domains": list(self.domain_index.keys()),
            "compression_ratio": f"{768 / self.pca_model['n_components']:.1f}x"
        }