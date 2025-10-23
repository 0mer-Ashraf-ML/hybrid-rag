# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import os
from src.service.rag_service import run_rag_pipeline
from src.utils.config import get_config, set_compression_mode

# Initialize FastAPI
app = FastAPI(
    title="Intelligent Wikipedia RAG API",
    description="Hybrid RAG with domain filtering, source evaluation, and LLM synthesis. "
                "Supports standard and ultra-compressed (50% smaller) retrieval modes.",
    version="2.0.0",
)

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    use_filtering: Optional[bool] = Field(True, description="Enable domain-based filtering")
    evaluate_sources: Optional[bool] = Field(True, description="Enable source quality evaluation")
    retrieval_mode: Optional[str] = Field(None, description="Force retrieval mode: 'standard', 'ultra', or None for auto")

class SourceInfo(BaseModel):
    title: str
    url: str
    domain: str

class EvaluationScores(BaseModel):
    keyword_relevance: float
    domain_relevance: float
    content_quality: float
    retrieval_confidence: float

class SourceEvaluation(BaseModel):
    scores: EvaluationScores
    final_score: float
    relevance_level: str
    should_use: bool

class RetrievedDoc(BaseModel):
    id: str
    title: str
    text: str  # Full text
    summary: str  # Full summary
    text_length: int
    summary_length: int
    url: str
    score: float
    final_score: float
    domain: str
    evaluation: Dict = Field(default_factory=dict)
    debug: Dict = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: str
    reasoning: str
    retrieved: List[RetrievedDoc]
    metadata: Dict

class ConfigResponse(BaseModel):
    retrieval_mode: str
    use_ultra_compressed: bool
    evaluate_sources: bool
    relevance_threshold: float
    final_k: int
    features: List[str]

class HealthResponse(BaseModel):
    status: str
    retrieval_mode: str
    index_available: Dict[str, bool]
    config: Dict

# ============================================================================
# Startup Configuration
# ============================================================================

# Global retriever instances (both modes pre-loaded)
_GLOBAL_RETRIEVERS = {
    'standard': None,
    'ultra': None
}

@app.on_event("startup")
async def startup_event():
    """Initialize retrieval system on startup - PRE-LOAD BOTH MODES"""
    global _GLOBAL_RETRIEVERS
    
    cfg = get_config()
    
    # Check environment variable for default mode
    env_mode = os.getenv("RAG_USE_ULTRA", "").lower()
    if env_mode in ["true", "1", "yes"]:
        set_compression_mode(use_ultra=True)
        default_mode = "ultra-compressed"
    elif env_mode in ["false", "0", "no"]:
        set_compression_mode(use_ultra=False)
        default_mode = "standard"
    else:
        default_mode = "ultra-compressed" if cfg.get("use_ultra_compressed", False) else "standard"
    
    print("="*70)
    print(f"🚀 Starting Intelligent Wikipedia RAG API")
    print(f"   Default Mode: {default_mode.upper()}")
    print(f"   Source Evaluation: {'ON' if cfg.get('evaluate_sources', True) else 'OFF'}")
    print(f"   Domain Filtering: {'ON' if cfg.get('use_domain_filtering', True) else 'OFF'}")
    print("="*70)
    
    # PRE-LOAD BOTH MODES at startup for instant switching!
    print("\n📦 Pre-loading BOTH retrieval modes at startup...")
    print("   This allows instant mode switching without reloading models!\n")
    
    from src.retriever.hybrid_retriever import UnifiedHybridRetriever
    
    try:
        # Load standard mode
        print("1️⃣  Loading STANDARD mode (BM25 + FAISS-768D + RRF)...")
        _GLOBAL_RETRIEVERS['standard'] = UnifiedHybridRetriever(force_mode='standard')
        print("   ✅ Standard mode ready\n")
    except Exception as e:
        print(f"   ⚠️  Standard mode failed: {e}\n")
        _GLOBAL_RETRIEVERS['standard'] = None
    
    try:
        # Load ultra mode
        print("2️⃣  Loading ULTRA mode (BM25 + FAISS-192D + RRF)...")
        _GLOBAL_RETRIEVERS['ultra'] = UnifiedHybridRetriever(force_mode='ultra')
        print("   ✅ Ultra mode ready\n")
    except FileNotFoundError as e:
        print(f"   ⚠️  Ultra mode not available: {e}")
        print(f"   Run: python scripts/compress_to_1_1gb.py\n")
        _GLOBAL_RETRIEVERS['ultra'] = None
    except Exception as e:
        print(f"   ⚠️  Ultra mode failed: {e}\n")
        _GLOBAL_RETRIEVERS['ultra'] = None
    
    # Summary
    print("="*70)
    modes_loaded = [m for m, r in _GLOBAL_RETRIEVERS.items() if r is not None]
    print(f"✅ Loaded modes: {', '.join(modes_loaded)}")
    print(f"🔄 Mode switching: {'Instant' if len(modes_loaded) > 1 else 'Not available'}")
    print(f"⚡ Ready for fast queries!")
    print("="*70 + "\n")

def get_retriever(mode: str = None):
    """
    Get pre-loaded retriever instance.
    
    Args:
        mode: 'standard' or 'ultra', or None for default
    
    Returns:
        Pre-loaded retriever instance (instant!)
    """
    global _GLOBAL_RETRIEVERS
    
    # Determine mode
    if mode is None:
        cfg = get_config()
        mode = 'ultra' if cfg.get('use_ultra_compressed', False) else 'standard'
    
    # Get pre-loaded retriever
    retriever = _GLOBAL_RETRIEVERS.get(mode)
    
    if retriever is None:
        # Fallback: try to load if not pre-loaded
        print(f"⚠️  {mode} mode not pre-loaded, loading now...")
        from src.retriever.hybrid_retriever import UnifiedHybridRetriever
        try:
            retriever = UnifiedHybridRetriever(force_mode=mode)
            _GLOBAL_RETRIEVERS[mode] = retriever
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"{mode} mode not available: {str(e)}"
            )
    
    return retriever

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def query_docs(payload: QueryRequest):
    """
    Query the Wikipedia knowledge base with intelligent filtering and evaluation.
    
    **Features:**
    - **Hybrid retrieval**: BM25 + FAISS + RRF fusion (BOTH modes)
    - Domain-based filtering for faster, more relevant results
    - Source quality evaluation with relevance scoring
    - LLM-powered answer synthesis
    - **Models pre-loaded at startup for fast queries**
    
    **Retrieval Modes:**
    - `standard`: BM25 + FAISS-768D + RRF (2.2 GB, full quality)
    - `ultra`: BM25 + FAISS-192D + RRF (1.1 GB, 50% smaller, <3% quality loss)
    - `None`: Auto-detect from config/environment
    
    **IMPORTANT:** Both modes use the SAME strategy (BM25 + FAISS + RRF).
    The only difference is FAISS compression (768D vs 192D PCA).
    This ensures CONSISTENT results with 50% storage savings!
    
    **Example:**
    ```json
    {
        "query": "What is quantum entanglement?",
        "top_k": 5,
        "use_filtering": true,
        "evaluate_sources": true,
        "retrieval_mode": null
    }
    ```
    """
    try:
        # Validate retrieval mode
        if payload.retrieval_mode and payload.retrieval_mode not in ['standard', 'ultra']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid retrieval_mode: {payload.retrieval_mode}. Use 'standard', 'ultra', or null."
            )
        
        # Use service layer with mode parameter
        response = run_rag_pipeline(
            query=payload.query,
            top_k=payload.top_k,
            use_filtering=payload.use_filtering,
            evaluate_sources=payload.evaluate_sources,
            retrieval_mode=payload.retrieval_mode  # Pass mode to service
        )
        
        return response
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index not available: {str(e)}. "
                   f"Run compression script or switch to standard mode."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/", response_model=Dict)
async def root():
    """API information and available features"""
    cfg = get_config()
    mode = "ultra-compressed" if cfg.get("use_ultra_compressed", False) else "standard"
    
    return {
        "message": "Intelligent Wikipedia RAG API v2.0 - Unified Hybrid Retrieval",
        "retrieval_mode": mode,
        "strategy": "BM25 + FAISS + RRF (both modes)",
        "features": [
            "Hybrid retrieval (BM25 + FAISS + RRF)",
            "Domain-based filtering",
            "Source relevance evaluation",
            "LLM answer synthesis",
            "Dynamic mode switching",
            "Model caching for fast queries",
            "RESTful API"
        ],
        "endpoints": {
            "/query": "POST - Query the knowledge base",
            "/config": "GET - View current configuration",
            "/config/mode": "POST - Switch retrieval mode",
            "/health": "GET - Health check with detailed status",
            "/docs": "GET - Interactive API documentation"
        },
        "modes": {
            "standard": {
                "size": "2.2 GB",
                "quality": "100%",
                "methods": "BM25 + FAISS-768D + RRF",
                "best_for": "Production, maximum quality"
            },
            "ultra": {
                "size": "1.1 GB (50% smaller)",
                "quality": "97%+ (<3% loss)",
                "methods": "BM25 + FAISS-192D + RRF (SAME strategy)",
                "best_for": "Edge, mobile, cost optimization"
            }
        },
        "note": "Both modes use identical retrieval strategy. Only difference: FAISS compression (768D vs 192D)."
    }


@app.get("/config", response_model=ConfigResponse)
async def get_current_config():
    """Get current system configuration"""
    global _GLOBAL_RETRIEVERS
    
    cfg = get_config()
    mode = "ultra-compressed" if cfg.get("use_ultra_compressed", False) else "standard"
    
    features = [
        "Domain filtering" if cfg.get("use_domain_filtering", True) else None,
        "Source evaluation" if cfg.get("evaluate_sources", True) else None,
        "Adaptive weights" if cfg.get("adaptive_weights", True) else None,
        f"{mode.upper()} retrieval (default)"
    ]
    
    # Add info about pre-loaded modes
    loaded_modes = [m for m, r in _GLOBAL_RETRIEVERS.items() if r is not None]
    if len(loaded_modes) > 1:
        features.append(f"Pre-loaded modes: {', '.join(loaded_modes)}")
    
    features = [f for f in features if f]  # Remove None values
    
    return ConfigResponse(
        retrieval_mode=mode,
        use_ultra_compressed=cfg.get("use_ultra_compressed", False),
        evaluate_sources=cfg.get("evaluate_sources", True),
        relevance_threshold=cfg.get("relevance_threshold", 0.45),
        final_k=cfg.get("final_k", 5),
        features=features
    )


@app.post("/config/mode")
async def switch_retrieval_mode(mode: str):
    """
    Dynamically switch retrieval mode.
    
    **Modes:**
    - `standard`: Full quality (2.2 GB)
    - `ultra`: 50% smaller (1.1 GB)
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/config/mode?mode=ultra"
    ```
    """
    if mode not in ['standard', 'ultra']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use 'standard' or 'ultra'."
        )
    
    try:
        # Update config
        use_ultra = (mode == 'ultra')
        new_config = set_compression_mode(use_ultra=use_ultra)
        
        return {
            "status": "success",
            "message": f"Switched to {mode} mode",
            "retrieval_mode": mode,
            "use_ultra_compressed": use_ultra,
            "note": "Existing retriever instances will continue using their original mode. "
                    "New requests will use the updated mode."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch mode: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check with index availability.
    
    Returns system status, retrieval mode, and index availability.
    """
    global _GLOBAL_RETRIEVERS
    
    import os
    from pathlib import Path
    
    cfg = get_config()
    mode = "ultra-compressed" if cfg.get("use_ultra_compressed", False) else "standard"
    
    # Check index availability
    standard_index = Path(cfg["wiki_index_path"])
    ultra_index = Path(cfg["wiki_ultra_index_path"]) if "wiki_ultra_index_path" in cfg else None
    
    index_available = {
        "standard": standard_index.exists() if standard_index else False,
        "ultra": ultra_index.exists() if ultra_index else False
    }
    
    # Check which modes are pre-loaded
    modes_loaded = {
        "standard": _GLOBAL_RETRIEVERS.get('standard') is not None,
        "ultra": _GLOBAL_RETRIEVERS.get('ultra') is not None
    }
    
    # Overall status
    current_index_available = (
        index_available["ultra"] if mode == "ultra-compressed" 
        else index_available["standard"]
    )
    status = "healthy" if current_index_available else "degraded"
    
    # Add pre-loaded info to config
    config_info = {
        "evaluate_sources": cfg.get("evaluate_sources", True),
        "use_domain_filtering": cfg.get("use_domain_filtering", True),
        "relevance_threshold": cfg.get("relevance_threshold", 0.45),
        "final_k": cfg.get("final_k", 5),
        "sem_k": cfg.get("sem_k", 15),
        "lex_k": cfg.get("lex_k", 30),
        "modes_pre_loaded": [m for m, loaded in modes_loaded.items() if loaded],
        "instant_switching": sum(modes_loaded.values()) > 1
    }
    
    return HealthResponse(
        status=status,
        retrieval_mode=mode,
        index_available=index_available,
        config=config_info
    )


@app.get("/stats")
async def get_stats():
    """
    Get retrieval statistics and index information.
    """
    try:
        from src.retriever.hybrid_retriever import UnifiedHybridRetriever
        
        cfg = get_config()
        mode = "ultra-compressed" if cfg.get("use_ultra_compressed", False) else "standard"
        
        stats = {
            "current_mode": mode,
            "index_stats": {}
        }
        
        # Try to get stats from ultra-compressed retriever if available
        if mode == "ultra-compressed":
            try:
                retriever = UnifiedHybridRetriever(force_mode='ultra')
                if hasattr(retriever.ultra, 'get_stats'):
                    stats["index_stats"] = retriever.ultra.get_stats()
            except Exception as e:
                stats["index_stats"] = {"error": str(e)}
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    # Read port from environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "true").lower() in ["true", "1", "yes"]
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload
    )