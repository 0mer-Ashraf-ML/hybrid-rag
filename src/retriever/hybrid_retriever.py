
from .filtered_bm25_retriever import FilteredBM25Retriever
from .filtered_faiss_retriever import FilteredFaissRetriever
from .ultra_compressed_retriever import UltraCompressedRetriever
from src.evaluation.source_evaluator import SourceEvaluator
from src.utils.config import get_config, get_index_paths

# Global retriever cache (loaded once at startup)
_RETRIEVER_CACHE = {}

class UnifiedHybridRetriever:
    def __init__(self, force_mode=None):
        """
        Initialize retriever with automatic mode detection.
        
        Args:
            force_mode: Override config ('standard', 'ultra', or None for auto)
        """
        cfg = get_config()
        
        # Load config weights FIRST (before initializing modes)
        self.W_SEM = cfg.get("w_sem", 0.6)
        self.W_LEX = cfg.get("w_lex", 0.3)
        self.W_RRF = cfg.get("w_rrf", 0.1)
        self.RRF_K = cfg.get("rrf_k", 60)
        
        # Determine mode
        if force_mode:
            self.mode = force_mode
        else:
            self.mode = 'ultra' if cfg.get('use_ultra_compressed', False) else 'standard'
        
        print(f"Initializing Unified Hybrid Retriever...")
        print(f"Mode: {self.mode.upper()}")
        
        # Initialize retrievers based on mode (with caching)
        if self.mode == 'ultra':
            self._init_ultra_compressed()
        else:
            self._init_standard()
        
        # Always initialize source evaluator (works with both modes)
        if 'evaluator' not in _RETRIEVER_CACHE:
            print("Initializing source evaluator (one-time)...")
            _RETRIEVER_CACHE['evaluator'] = SourceEvaluator()
            print(f"✅ Source evaluator initialized")
        
        self.evaluator = _RETRIEVER_CACHE['evaluator']
        
        print(f"✅ Unified Hybrid Retriever ready")
    
    def _init_standard(self):
        """Initialize standard hybrid retrieval (BM25 + FAISS-768D) - uses cache"""
        global _RETRIEVER_CACHE
        
        if 'standard_bm25' not in _RETRIEVER_CACHE:
            print("  Loading standard retrievers (BM25 + FAISS-768D) - one-time initialization...")
            _RETRIEVER_CACHE['standard_bm25'] = FilteredBM25Retriever()
            _RETRIEVER_CACHE['standard_faiss'] = FilteredFaissRetriever()
            print(f"  ✅ Standard retrievers cached")
        else:
            print("  Using cached standard retrievers (fast)")
        
        self.bm = _RETRIEVER_CACHE['standard_bm25']
        self.faiss = _RETRIEVER_CACHE['standard_faiss']
        self.ultra = None
        print(f"  Strategy: BM25 + FAISS-768D + RRF")
        print(f"  Weights: Semantic={self.W_SEM}, Lexical={self.W_LEX}, RRF={self.W_RRF}")
    
    def _init_ultra_compressed(self):
        """Initialize ultra-compressed retrieval (BM25 + FAISS-192D) - uses cache"""
        global _RETRIEVER_CACHE
        
        if 'ultra_faiss' not in _RETRIEVER_CACHE:
            print("  Loading ultra-compressed retrievers (BM25 + FAISS-192D) - one-time initialization...")
            # Ultra mode ALSO uses BM25 (same as standard)
            _RETRIEVER_CACHE['ultra_bm25'] = FilteredBM25Retriever()
            _RETRIEVER_CACHE['ultra_faiss'] = UltraCompressedRetriever()
            print(f"  ✅ Ultra retrievers cached")
        else:
            print("  Using cached ultra-compressed retrievers (fast)")
        
        self.bm = _RETRIEVER_CACHE['ultra_bm25']
        self.ultra = _RETRIEVER_CACHE['ultra_faiss']  # Ultra FAISS (192D PCA)
        self.faiss = None  # Not used in ultra mode
        print(f"  Strategy: BM25 + FAISS-192D + RRF (same as standard)")
        print(f"  Weights: Semantic={self.W_SEM}, Lexical={self.W_LEX}, RRF={self.W_RRF}")
        print(f"  Difference: Only FAISS uses 192D PCA (50% smaller, <3% quality loss)")
    
    def _normalize_scores(self, score_list):
        """Min-max normalize to [0,1]."""
        if not score_list:
            return []
        mn = min(score_list)
        mx = max(score_list)
        if mx == mn:
            return [1.0 for _ in score_list]
        return [(s - mn) / (mx - mn) for s in score_list]
    
    def retrieve(self, query, sem_k=None, lex_k=None, final_k=None, 
                 use_filtering=True, evaluate_sources=True):
        """
        Unified retrieval interface that works for both modes.
        
        BOTH modes use: BM25 (lexical) + FAISS (semantic) + RRF fusion
        
        Args:
            query: Search query
            sem_k: Number of semantic results
            lex_k: Number of lexical results
            final_k: Final number of results to return
            use_filtering: Enable domain-based filtering
            evaluate_sources: Enable source quality evaluation
        
        Returns:
            List of retrieved documents with text, summary, and evaluation
        """
        cfg = get_config()
        sem_k = sem_k or cfg.get("sem_k", 15)
        lex_k = lex_k or cfg.get("lex_k", 30)
        final_k = final_k or cfg.get("final_k", 5)
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Strategy: BM25 + FAISS + RRF (hybrid)")
        print(f"Filtering: {'ON' if use_filtering else 'OFF'}")
        print(f"Evaluation: {'ON' if evaluate_sources else 'OFF'}")
        print(f"{'='*60}")
        
        # Both modes use the SAME hybrid retrieval strategy
        return self._retrieve_hybrid(
            query, sem_k, lex_k, final_k, use_filtering, evaluate_sources
        )
    
    def _retrieve_hybrid(self, query, sem_k, lex_k, final_k, use_filtering, evaluate_sources):
        """
        Hybrid retrieval (BM25 + FAISS + RRF) - used by BOTH modes.
        Only difference: standard uses FAISS-768D, ultra uses FAISS-192D.
        """
        # Semantic retrieval (FAISS)
        print(f"  🔍 Semantic search (top {sem_k})...")
        if self.mode == 'ultra':
            # Ultra mode: Use compressed FAISS (192D PCA)
            sem_results = self.ultra.retrieve(query, top_k=sem_k, use_filtering=use_filtering)
        else:
            # Standard mode: Use full FAISS (768D)
            sem_results = self.faiss.retrieve(query, top_k=sem_k, use_filtering=use_filtering)
        
        # Lexical retrieval (BM25) - SAME for both modes
        print(f"  📝 Lexical search (top {lex_k})...")
        lex_results = self.bm.retrieve(query, top_k=lex_k, use_filtering=use_filtering)
        
        # Combine with RRF - SAME for both modes
        combined = self._combine_hybrid(sem_results, lex_results)
        
        # Get initial results (2x for evaluation)
        initial_k = final_k * 2 if evaluate_sources else final_k
        initial_results = combined[:initial_k]
        
        # Format results
        formatted_results = self._format_results(initial_results)
        
        # Evaluate sources if enabled
        if evaluate_sources:
            formatted_results = self._apply_evaluation(query, formatted_results, final_k)
        
        # Return final_k results
        final_results = formatted_results[:final_k]
        
        print(f"  ✅ Returning top {len(final_results)} results")
        return final_results
    
    def _combine_hybrid(self, sem_results, lex_results):
        """
        Combine semantic and lexical results with RRF.
        This is the SAME for both standard and ultra modes.
        """
        # Create lookup maps
        sem_map = {r["doc_idx"]: r for r in sem_results}
        lex_map = {r["doc_idx"]: r for r in lex_results}
        
        # Normalize scores
        sem_scores = [r["score"] for r in sem_results]
        lex_scores = [r["score"] for r in lex_results]
        sem_norm = self._normalize_scores(sem_scores)
        lex_norm = self._normalize_scores(lex_scores)
        
        for r, ns in zip(sem_results, sem_norm):
            r["norm_score"] = ns
        for r, ns in zip(lex_results, lex_norm):
            r["norm_score"] = ns
        
        # Create rank maps for RRF
        sem_rank = {r["doc_idx"]: rank for rank, r in enumerate(sem_results, start=1)}
        lex_rank = {r["doc_idx"]: rank for rank, r in enumerate(lex_results, start=1)}
        
        # Combine candidates
        candidate_idxs = set(list(sem_map.keys()) + list(lex_map.keys()))
        print(f"  🔄 Combining {len(candidate_idxs)} unique documents...")
        
        combined = []
        for idx in candidate_idxs:
            sem_entry = sem_map.get(idx)
            lex_entry = lex_map.get(idx)
            
            sem_norm_score = sem_entry["norm_score"] if sem_entry else 0.0
            lex_norm_score = lex_entry["norm_score"] if lex_entry else 0.0
            
            # RRF component
            rrf_comp = 0.0
            if idx in sem_rank:
                rrf_comp += 1.0 / (self.RRF_K + sem_rank[idx])
            if idx in lex_rank:
                rrf_comp += 1.0 / (self.RRF_K + lex_rank[idx])
            
            # Combined score
            combined_score = (
                self.W_SEM * sem_norm_score + 
                self.W_LEX * lex_norm_score + 
                self.W_RRF * rrf_comp
            )
            
            # Use metadata from whichever source has it (prefer semantic)
            source_entry = sem_entry if sem_entry else lex_entry
            meta = source_entry["meta"]
            text = source_entry.get("text", meta.get("text", ""))
            summary = source_entry.get("summary", meta.get("summary", ""))
            doc_id = source_entry["id"]
            
            combined.append({
                "doc_idx": idx,
                "id": doc_id,
                "text": text,
                "summary": summary,
                "meta": meta,
                "score": float(combined_score),
                "debug": {
                    "sem_norm": sem_norm_score,
                    "lex_norm": lex_norm_score,
                    "rrf_comp": rrf_comp
                }
            })
        
        # Prioritize intersection (documents in both retrievers)
        intersection = [c for c in combined if (c["doc_idx"] in sem_map and c["doc_idx"] in lex_map)]
        others = [c for c in combined if (c["doc_idx"] not in sem_map or c["doc_idx"] not in lex_map)]
        
        intersection_sorted = sorted(intersection, key=lambda x: x["score"], reverse=True)
        others_sorted = sorted(others, key=lambda x: x["score"], reverse=True)
        
        # Combine with intersection priority
        return intersection_sorted + others_sorted
    
    def _format_results(self, results):
        """Format results into standard structure"""
        formatted = []
        
        for r in results:
            # Get text and summary
            full_text = r.get("text", r.get("meta", {}).get("text", ""))
            summary = r.get("summary", r.get("meta", {}).get("summary", ""))
            
            formatted.append({
                "id": r["id"],
                "doc_idx": int(r["doc_idx"]),
                "chunk_no": int(r["doc_idx"]),  # Add chunk_no
                "page_no": int(r["doc_idx"]),   # Add page_no
                "score": float(r.get("score", 0)),
                "text": full_text,
                "summary": summary,
                "meta": {
                    "title": r["meta"].get("title", ""),
                    "url": r["meta"].get("url", ""),
                    "categories": r["meta"].get("categories", ""),
                    "primary_domain": r["meta"].get("primary_domain", "general"),
                    "text_length": r["meta"].get("text_length", len(full_text)),
                    "summary_length": r["meta"].get("summary_length", len(summary)),
                    "text": full_text,
                    "summary": summary,
                    "chunk_no": int(r["doc_idx"]),
                    "page_no": int(r["doc_idx"])
                },
                "debug": r.get("debug", {})
            })
        
        return formatted
    
    def _apply_evaluation(self, query, formatted_results, final_k):
        """Apply source evaluation and re-ranking"""
        cfg = get_config()
        
        print(f"  🔍 Evaluating source relevance...")
        formatted_results = self.evaluator.evaluate_sources(query, formatted_results)
        
        # Filter by relevance threshold
        threshold = cfg.get("relevance_threshold", 0.45)
        formatted_results = self.evaluator.filter_by_relevance(formatted_results, threshold)
        
        # Re-rank by evaluation score
        retrieval_weight = cfg.get("retrieval_weight", 0.6)
        evaluation_weight = cfg.get("evaluation_weight", 0.4)
        
        for doc in formatted_results:
            retrieval_score = doc.get("score", 0.5)
            eval_score = doc["evaluation"]["final_score"]
            doc["final_score"] = retrieval_weight * retrieval_score + evaluation_weight * eval_score
        
        formatted_results = sorted(
            formatted_results,
            key=lambda x: x.get("final_score", x.get("score", 0)),
            reverse=True
        )
        
        return formatted_results
    
    def switch_mode(self, mode):
        """
        Dynamically switch retrieval mode.
        
        Args:
            mode: 'standard' or 'ultra'
        """
        if mode not in ['standard', 'ultra']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'ultra'")
        
        if mode == self.mode:
            print(f"Already in {mode} mode")
            return
        
        print(f"\nSwitching from {self.mode} to {mode} mode...")
        self.mode = mode
        
        if mode == 'ultra':
            self._init_ultra_compressed()
        else:
            self._init_standard()
        
        print(f"✅ Switched to {mode} mode")


# Backward compatibility - keep HybridRetriever as alias
class HybridRetriever(UnifiedHybridRetriever):
    """
    Backward-compatible HybridRetriever.
    Automatically uses UnifiedHybridRetriever with mode detection.
    """
    def __init__(self):
        super().__init__(force_mode='standard')
    
    def _normalize_scores(self, score_list):
        """Min-max normalize to [0,1]."""
        if not score_list:
            return []
        mn = min(score_list)
        mx = max(score_list)
        if mx == mn:
            return [1.0 for _ in score_list]
        return [(s - mn) / (mx - mn) for s in score_list]
    
    def retrieve(self, query, sem_k=None, lex_k=None, final_k=None, 
                 use_filtering=True, evaluate_sources=True):
        """
        Unified retrieval interface that works for both modes.
        
        Args:
            query: Search query
            sem_k: Number of semantic results
            lex_k: Number of lexical results (ignored in ultra mode)
            final_k: Final number of results to return
            use_filtering: Enable domain-based filtering
            evaluate_sources: Enable source quality evaluation
        
        Returns:
            List of retrieved documents with text, summary, and evaluation
        """
        cfg = get_config()
        sem_k = sem_k or cfg.get("sem_k", 15)
        lex_k = lex_k or cfg.get("lex_k", 30)
        final_k = final_k or cfg.get("final_k", 5)
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Filtering: {'ON' if use_filtering else 'OFF'}")
        print(f"Evaluation: {'ON' if evaluate_sources else 'OFF'}")
        print(f"{'='*60}")
        
        # Route to appropriate retrieval method
        if self.mode == 'ultra':
            return self._retrieve_ultra_compressed(
                query, sem_k, final_k, use_filtering, evaluate_sources
            )
        else:
            return self._retrieve_standard(
                query, sem_k, lex_k, final_k, use_filtering, evaluate_sources
            )
    
    def _retrieve_ultra_compressed(self, query, top_k, final_k, use_filtering, evaluate_sources):
        """Retrieval using ultra-compressed index"""
        print(f"  ⚡ Ultra-compressed search (top {top_k})...")
        
        # Get initial results (2x for evaluation)
        initial_k = top_k * 2 if evaluate_sources else final_k
        results = self.ultra.retrieve(query, top_k=initial_k, use_filtering=use_filtering)
        
        # Normalize scores
        scores = [r["score"] for r in results]
        norm_scores = self._normalize_scores(scores)
        
        for r, ns in zip(results, norm_scores):
            r["score"] = ns
            r["norm_score"] = ns
        
        # Format results
        formatted_results = self._format_results(results)
        
        # Evaluate sources if enabled
        if evaluate_sources:
            formatted_results = self._apply_evaluation(query, formatted_results, final_k)
        
        # Return final_k results
        final_results = formatted_results[:final_k]
        
        print(f"  ✅ Returning top {len(final_results)} results")
        return final_results
    
    def _retrieve_standard(self, query, sem_k, lex_k, final_k, use_filtering, evaluate_sources):
        """Standard hybrid retrieval (BM25 + FAISS + RRF)"""
        # Semantic retrieval
        print(f"  🔍 Semantic search (top {sem_k})...")
        sem_results = self.faiss.retrieve(query, top_k=sem_k, use_filtering=use_filtering)
        
        # Lexical retrieval
        print(f"  📝 Lexical search (top {lex_k})...")
        lex_results = self.bm.retrieve(query, top_k=lex_k, use_filtering=use_filtering)
        
        # Combine with RRF
        combined = self._combine_standard(sem_results, lex_results)
        
        # Get initial results (2x for evaluation)
        initial_k = final_k * 2 if evaluate_sources else final_k
        initial_results = combined[:initial_k]
        
        # Format results
        formatted_results = self._format_results(initial_results)
        
        # Evaluate sources if enabled
        if evaluate_sources:
            formatted_results = self._apply_evaluation(query, formatted_results, final_k)
        
        # Return final_k results
        final_results = formatted_results[:final_k]
        
        print(f"  ✅ Returning top {len(final_results)} evaluated results")
        return final_results
    
    def _combine_standard(self, sem_results, lex_results):
        """Combine semantic and lexical results with RRF"""
        # Create lookup maps
        sem_map = {r["doc_idx"]: r for r in sem_results}
        lex_map = {r["doc_idx"]: r for r in lex_results}
        
        # Normalize scores
        sem_scores = [r["score"] for r in sem_results]
        lex_scores = [r["score"] for r in lex_results]
        sem_norm = self._normalize_scores(sem_scores)
        lex_norm = self._normalize_scores(lex_scores)
        
        for r, ns in zip(sem_results, sem_norm):
            r["norm_score"] = ns
        for r, ns in zip(lex_results, lex_norm):
            r["norm_score"] = ns
        
        # Create rank maps for RRF
        sem_rank = {r["doc_idx"]: rank for rank, r in enumerate(sem_results, start=1)}
        lex_rank = {r["doc_idx"]: rank for rank, r in enumerate(lex_results, start=1)}
        
        # Combine candidates
        candidate_idxs = set(list(sem_map.keys()) + list(lex_map.keys()))
        print(f"  🔄 Combining {len(candidate_idxs)} unique documents...")
        
        combined = []
        for idx in candidate_idxs:
            sem_entry = sem_map.get(idx)
            lex_entry = lex_map.get(idx)
            
            sem_norm_score = sem_entry["norm_score"] if sem_entry else 0.0
            lex_norm_score = lex_entry["norm_score"] if lex_entry else 0.0
            
            # RRF component
            rrf_comp = 0.0
            if idx in sem_rank:
                rrf_comp += 1.0 / (self.RRF_K + sem_rank[idx])
            if idx in lex_rank:
                rrf_comp += 1.0 / (self.RRF_K + lex_rank[idx])
            
            # Combined score
            combined_score = (
                self.W_SEM * sem_norm_score + 
                self.W_LEX * lex_norm_score + 
                self.W_RRF * rrf_comp
            )
            
            # Use metadata from whichever source has it (prefer semantic)
            source_entry = sem_entry if sem_entry else lex_entry
            meta = source_entry["meta"]
            text = source_entry.get("text", meta.get("text", ""))
            summary = source_entry.get("summary", meta.get("summary", ""))
            doc_id = source_entry["id"]
            
            combined.append({
                "doc_idx": idx,
                "id": doc_id,
                "text": text,
                "summary": summary,
                "meta": meta,
                "score": float(combined_score),
                "debug": {
                    "sem_norm": sem_norm_score,
                    "lex_norm": lex_norm_score,
                    "rrf_comp": rrf_comp
                }
            })
        
        # Prioritize intersection (documents in both retrievers)
        intersection = [c for c in combined if (c["doc_idx"] in sem_map and c["doc_idx"] in lex_map)]
        others = [c for c in combined if (c["doc_idx"] not in sem_map or c["doc_idx"] not in lex_map)]
        
        intersection_sorted = sorted(intersection, key=lambda x: x["score"], reverse=True)
        others_sorted = sorted(others, key=lambda x: x["score"], reverse=True)
        
        # Combine with intersection priority
        return intersection_sorted + others_sorted
    
    def _format_results(self, results):
        """Format results into standard structure"""
        formatted = []
        
        for r in results:
            # Get text and summary
            full_text = r.get("text", r.get("meta", {}).get("text", ""))
            summary = r.get("summary", r.get("meta", {}).get("summary", ""))
            
            formatted.append({
                "id": r["id"],
                "doc_idx": int(r["doc_idx"]),
                "score": float(r.get("score", 0)),
                "text": full_text,
                "summary": summary,
                "meta": {
                    "title": r["meta"].get("title", ""),
                    "url": r["meta"].get("url", ""),
                    "categories": r["meta"].get("categories", ""),
                    "primary_domain": r["meta"].get("primary_domain", "general"),
                    "text_length": r["meta"].get("text_length", len(full_text)),
                    "summary_length": r["meta"].get("summary_length", len(summary)),
                    "text": full_text,
                    "summary": summary
                },
                "debug": r.get("debug", {})
            })
        
        return formatted
    
    def _apply_evaluation(self, query, formatted_results, final_k):
        """Apply source evaluation and re-ranking"""
        cfg = get_config()
        
        print(f"  🔍 Evaluating source relevance...")
        formatted_results = self.evaluator.evaluate_sources(query, formatted_results)
        
        # Filter by relevance threshold
        threshold = cfg.get("relevance_threshold", 0.45)
        formatted_results = self.evaluator.filter_by_relevance(formatted_results, threshold)
        
        # Re-rank by evaluation score
        retrieval_weight = cfg.get("retrieval_weight", 0.6)
        evaluation_weight = cfg.get("evaluation_weight", 0.4)
        
        for doc in formatted_results:
            retrieval_score = doc.get("score", 0.5)
            eval_score = doc["evaluation"]["final_score"]
            doc["final_score"] = retrieval_weight * retrieval_score + evaluation_weight * eval_score
        
        formatted_results = sorted(
            formatted_results,
            key=lambda x: x.get("final_score", x.get("score", 0)),
            reverse=True
        )
        
        return formatted_results
    
    def switch_mode(self, mode):
        """
        Dynamically switch retrieval mode.
        
        Args:
            mode: 'standard' or 'ultra'
        """
        if mode not in ['standard', 'ultra']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'ultra'")
        
        if mode == self.mode:
            print(f"Already in {mode} mode")
            return
        
        print(f"\nSwitching from {self.mode} to {mode} mode...")
        self.mode = mode
        
        if mode == 'ultra':
            self._init_ultra_compressed()
        else:
            self._init_standard()
        
        print(f"✅ Switched to {mode} mode")


# Backward compatibility - keep HybridRetriever as alias
class HybridRetriever(UnifiedHybridRetriever):
    """
    Backward-compatible HybridRetriever.
    Automatically uses UnifiedHybridRetriever with mode detection.
    """
    def __init__(self):
        super().__init__(force_mode='standard')