# src/utils/model_cache.py
"""
Global model cache for efficient model loading across all retrievers.
Models are loaded ONCE at startup and shared across all instances.
"""

from sentence_transformers import SentenceTransformer
from typing import Optional

# Global caches
_EMBEDDER_CACHE: Optional[SentenceTransformer] = None
_QUERY_CLASSIFIER_CACHE = None

def get_cached_embedder(model_name: str = "BAAI/bge-base-en") -> SentenceTransformer:
    """
    Get or create cached embedding model.
    
    This model is loaded ONCE at startup and shared across:
    - FilteredFaissRetriever
    - UltraCompressedRetriever
    - Any other component needing embeddings
    
    Args:
        model_name: Name of the sentence transformer model
    
    Returns:
        Cached SentenceTransformer instance
    """
    global _EMBEDDER_CACHE
    
    if _EMBEDDER_CACHE is None:
        print(f"🔄 Loading embedding model: {model_name} (one-time initialization)...")
        print(f"   This model will be shared across all retrievers for efficiency")
        _EMBEDDER_CACHE = SentenceTransformer(model_name)
        print(f"✅ Embedding model loaded and cached globally")
    
    return _EMBEDDER_CACHE

def get_cached_query_classifier():
    """
    Get or create cached query classifier.
    
    Returns:
        Cached QueryClassifier instance
    """
    global _QUERY_CLASSIFIER_CACHE
    
    if _QUERY_CLASSIFIER_CACHE is None:
        from src.utils.query_classifier import QueryClassifier
        print(f"🔄 Loading query classifier (one-time initialization)...")
        _QUERY_CLASSIFIER_CACHE = QueryClassifier()
        print(f"✅ Query classifier loaded and cached globally")
    
    return _QUERY_CLASSIFIER_CACHE

def clear_model_cache():
    """Clear all cached models (useful for testing or reloading)"""
    global _EMBEDDER_CACHE, _QUERY_CLASSIFIER_CACHE
    
    _EMBEDDER_CACHE = None
    _QUERY_CLASSIFIER_CACHE = None
    
    print("✓ Model cache cleared")

def get_cache_info() -> dict:
    """Get information about cached models"""
    return {
        "embedder_loaded": _EMBEDDER_CACHE is not None,
        "query_classifier_loaded": _QUERY_CLASSIFIER_CACHE is not None,
    }