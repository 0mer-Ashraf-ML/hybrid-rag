# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WIKI_DB_PATH = DATA_DIR / "wikipedia" / "wikipedia_comprehensive.db"

# Wikipedia Index paths
WIKI_INDEX_DIR = DATA_DIR / "index"
WIKI_FAISS_INDEX = WIKI_INDEX_DIR / "wikipedia.index"
WIKI_METADATA = WIKI_INDEX_DIR / "wikipedia.pkl"  # Original - has text/summary
WIKI_METADATA_ENHANCED = WIKI_INDEX_DIR / "wikipedia_enhanced.pkl"  # Optional - has domains
WIKI_DOCS_JSON = WIKI_INDEX_DIR / "docs.json"
WIKI_PROGRESS = WIKI_INDEX_DIR / "progress.json"

# Ultra-compressed index paths (50% size, <3% quality loss)
WIKI_ULTRA_INDEX_DIR = Path("/Users/omarashraf/Downloads/hybrid-rag/ultra_compressed_208k_p6")
WIKI_ULTRA_FAISS_INDEX = WIKI_ULTRA_INDEX_DIR / "wikipedia_ultra.index"
WIKI_ULTRA_METADATA = WIKI_ULTRA_INDEX_DIR / "metadata_ultra_enhanced.pkl"
WIKI_ULTRA_PCA = WIKI_ULTRA_INDEX_DIR / "pca_192.pkl"
WIKI_ULTRA_QUANT = WIKI_ULTRA_INDEX_DIR / "quant_params.pkl"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding model (must match what you used for indexing)
EMBEDDING_MODEL = "BAAI/bge-base-en"

# Local LLM settings
LOCAL_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# OpenAI model for source evaluation and generation
OPENAI_MODEL = "gpt-4o-mini"

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Retriever mode selection
USE_ULTRA_COMPRESSED = False  # Set to True to use ultra-compressed index (50% smaller)

# Retrieval counts
FINAL_K = 5  # Final number of documents to return
SEM_K = 15   # Number of semantic (FAISS) results
LEX_K = 30   # Number of lexical (BM25) results

# Hybrid RAG weights
W_SEM = 0.6  # Semantic (FAISS) weight
W_LEX = 0.3  # Lexical (BM25) weight
W_RRF = 0.1  # Reciprocal Rank Fusion weight
RRF_K = 60   # RRF constant

# Domain filtering
USE_DOMAIN_FILTERING = True

# Ultra-compressed index settings
ULTRA_NPROBE = 20  # Number of clusters to search (higher = better quality, slower)
ULTRA_DIMENSIONS = 192  # PCA-reduced dimensions

# ============================================================================
# SOURCE EVALUATION CONFIGURATION
# ============================================================================

# Enable/disable source evaluation
EVALUATE_SOURCES = True

# Relevance threshold (0.0 - 1.0)
# Documents below this score will be filtered out
RELEVANCE_THRESHOLD = 0.45

# Use LLM for evaluation (more accurate but slower)
# If False, uses rule-based heuristics only
USE_LLM_EVALUATION = False

# Adaptive weights for combining retrieval and evaluation scores
ADAPTIVE_WEIGHTS = True

# Retrieval vs Evaluation weight ratio
RETRIEVAL_WEIGHT = 0.6  # Weight for initial retrieval score
EVALUATION_WEIGHT = 0.4  # Weight for evaluation score

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

# Temperature for generation
TEMPERATURE = 0.2

# Max tokens for generation
MAX_TOKENS = 1024

# ============================================================================
# MAIN CONFIG FUNCTION
# ============================================================================

def get_config():
    """
    Returns a dictionary with all configuration parameters.
    Can be extended to support environment-specific overrides.
    """
    print('Getting config')
    return {
        # Paths - Standard Index
        "wiki_index_path": str(WIKI_FAISS_INDEX),
        "wiki_metadata_path": str(WIKI_METADATA),
        "wiki_metadata_enhanced": str(WIKI_METADATA_ENHANCED),
        "wiki_db_path": str(WIKI_DB_PATH),
        "wiki_docs_json": str(WIKI_DOCS_JSON),
        "wiki_progress": str(WIKI_PROGRESS),
        
        # Paths - Ultra-Compressed Index
        "wiki_ultra_index_path": str(WIKI_ULTRA_FAISS_INDEX),
        "wiki_ultra_metadata_path": str(WIKI_ULTRA_METADATA),
        "wiki_ultra_pca_path": str(WIKI_ULTRA_PCA),
        "wiki_ultra_quant_path": str(WIKI_ULTRA_QUANT),
        "use_ultra_compressed": USE_ULTRA_COMPRESSED,
        "ultra_nprobe": ULTRA_NPROBE,
        "ultra_dimensions": ULTRA_DIMENSIONS,
        
        # Models
        "embedding_model": EMBEDDING_MODEL,
        "local_llm_model": LOCAL_LLM_MODEL,
        "openai_model": OPENAI_MODEL,
        
        # Retrieval parameters
        "final_k": FINAL_K,
        "sem_k": SEM_K,
        "lex_k": LEX_K,
        "w_sem": W_SEM,
        "w_lex": W_LEX,
        "w_rrf": W_RRF,
        "rrf_k": RRF_K,
        "use_domain_filtering": USE_DOMAIN_FILTERING,
        
        # Source evaluation
        "evaluate_sources": EVALUATE_SOURCES,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "use_llm_evaluation": USE_LLM_EVALUATION,
        "adaptive_weights": ADAPTIVE_WEIGHTS,
        "retrieval_weight": RETRIEVAL_WEIGHT,
        "evaluation_weight": EVALUATION_WEIGHT,
        
        # Generation
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }


def update_config(**kwargs):
    """
    Update configuration values dynamically.
    
    Example:
        update_config(final_k=10, temperature=0.5)
    """
    config = get_config()
    config.update(kwargs)
    return config


def get_evaluation_config():
    """
    Get configuration specific to source evaluation.
    Useful for the SourceEvaluator class.
    """
    return {
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "use_llm_evaluation": USE_LLM_EVALUATION,
        "adaptive_weights": ADAPTIVE_WEIGHTS,
        "openai_model": OPENAI_MODEL,
        "temperature": TEMPERATURE,
    }


def get_retrieval_config():
    """
    Get configuration specific to retrieval.
    Useful for retriever classes.
    """
    return {
        "final_k": FINAL_K,
        "sem_k": SEM_K,
        "lex_k": LEX_K,
        "w_sem": W_SEM,
        "w_lex": W_LEX,
        "w_rrf": W_RRF,
        "rrf_k": RRF_K,
        "use_domain_filtering": USE_DOMAIN_FILTERING,
        "use_ultra_compressed": USE_ULTRA_COMPRESSED,
        "ultra_nprobe": ULTRA_NPROBE,
    }


def get_index_paths():
    """
    Get the correct index paths based on compression mode.
    Returns paths for either standard or ultra-compressed index.
    """
    if USE_ULTRA_COMPRESSED:
        return {
            "index_path": str(WIKI_ULTRA_FAISS_INDEX),
            "metadata_path": str(WIKI_ULTRA_METADATA),
            "pca_path": str(WIKI_ULTRA_PCA),
            "quant_path": str(WIKI_ULTRA_QUANT),
            "mode": "ultra_compressed",
            "dimensions": ULTRA_DIMENSIONS,
        }
    else:
        return {
            "index_path": str(WIKI_FAISS_INDEX),
            "metadata_path": str(WIKI_METADATA),
            "metadata_enhanced_path": str(WIKI_METADATA_ENHANCED),
            "mode": "standard",
            "dimensions": 768,
        }


def set_compression_mode(use_ultra: bool = False):
    """
    Dynamically switch between standard and ultra-compressed modes.
    
    Args:
        use_ultra: If True, use ultra-compressed index (50% smaller, <3% quality loss)
    
    Example:
        set_compression_mode(use_ultra=True)
        config = get_config()  # Will now use ultra-compressed paths
    """
    global USE_ULTRA_COMPRESSED
    USE_ULTRA_COMPRESSED = use_ultra
    
    mode = "ultra-compressed" if use_ultra else "standard"
    print(f"✓ Compression mode set to: {mode}")
    
    return get_config()