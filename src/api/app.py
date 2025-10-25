# src/api/app.py
"""
Clean Wikipedia RAG API.

Much simpler than before:
- Semantic-first retrieval (fast)
- Optional lexical boost (for keywords)
- Lightweight relevance filtering
- LLM answer generation
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn

from src.models import (
    QueryRequest, QueryResponse, HealthResponse, ConfigResponse,
    Source, RetrievedDocument
)
from src.config import get_config, DEFAULT_MODE, FINAL_K
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.evaluation.relevance_filter import get_relevance_filter
from src.generation.answer_generator import get_answer_generator


# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Clean Wikipedia RAG API",
    description=(
        "Fast semantic search over 7M Wikipedia articles. "
        "Dual-mode support: standard (full quality) or ultra (50% smaller)."
    ),
    version="2.0.0"
)


# ============================================================================
# GLOBAL INSTANCES (loaded once at startup)
# ============================================================================

_STANDARD_RETRIEVER = None
_ULTRA_RETRIEVER = None
_RELEVANCE_FILTER = None
_ANSWER_GENERATOR = None


@app.on_event("startup")
async def startup():
    """Pre-load models at startup for fast queries."""
    global _STANDARD_RETRIEVER, _ULTRA_RETRIEVER, _RELEVANCE_FILTER, _ANSWER_GENERATOR
    
    print("="*70)
    print("🚀 Starting Clean Wikipedia RAG API")
    print("="*70)
    
    # Pre-load both modes (if available)
    print("\n📦 Pre-loading retrieval modes...")
    
    # Try standard
    try:
        print("\n1️⃣  Loading STANDARD mode (768D, full quality)...")
        _STANDARD_RETRIEVER = get_hybrid_retriever('standard')
    except FileNotFoundError as e:
        print(f"   ⚠️  Standard mode not available: {e}")
        _STANDARD_RETRIEVER = None
    
    # Try ultra
    try:
        print("\n2️⃣  Loading ULTRA mode (192D, 50% smaller)...")
        _ULTRA_RETRIEVER = get_hybrid_retriever('ultra')
    except FileNotFoundError as e:
        print(f"   ⚠️  Ultra mode not available: {e}")
        _ULTRA_RETRIEVER = None
    
    # Check if at least one mode is available
    if _STANDARD_RETRIEVER is None and _ULTRA_RETRIEVER is None:
        raise RuntimeError("No retrieval mode available! Check your index paths.")
    
    # Load filter and generator
    print("\n3️⃣  Loading relevance filter...")
    _RELEVANCE_FILTER = get_relevance_filter()
    
    print("\n4️⃣  Loading answer generator...")
    _ANSWER_GENERATOR = get_answer_generator()
    
    print("\n" + "="*70)
    print("✅ API Ready!")
    modes = []
    if _STANDARD_RETRIEVER: modes.append("standard")
    if _ULTRA_RETRIEVER: modes.append("ultra")
    print(f"   Available modes: {', '.join(modes)}")
    print(f"   Default mode: {DEFAULT_MODE}")
    print("="*70 + "\n")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Redirect to docs."""
    return RedirectResponse(url="/docs")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the Wikipedia knowledge base.
    
    **Modes:**
    - `standard`: Full 768D embeddings (2.2 GB, 100% quality)
    - `ultra`: 192D PCA compressed (1.1 GB, 97%+ quality)
    
    **Retrieval:**
    - Default: Fast semantic search only (FAISS)
    - Optional: Add lexical boost for keyword-heavy queries (slower)
    
    **Example:**
    ```json
    {
        "query": "What is photosynthesis?",
        "mode": "ultra",
        "top_k": 5,
        "use_lexical": false
    }
    ```
    """
    # Get retriever
    mode = request.mode or DEFAULT_MODE
    
    if mode == 'standard':
        if _STANDARD_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Standard mode not available. Use 'ultra' mode or check index paths."
            )
        retriever = _STANDARD_RETRIEVER
    else:
        if _ULTRA_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Ultra mode not available. Use 'standard' mode or check index paths."
            )
        retriever = _ULTRA_RETRIEVER
    
    print(f"\n{'='*60}")
    print(f"Query: {request.query}")
    print(f"Mode: {mode.upper()}")
    print(f"Use lexical: {request.use_lexical}")
    print(f"{'='*60}")
    
    try:
        # 1. Retrieve documents
        documents = retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,  # Get more for filtering
            use_lexical=request.use_lexical
        )
        
        if not documents:
            return QueryResponse(
                answer="No relevant information found.",
                sources=[],
                confidence="low",
                retrieved=[],
                metadata={
                    'mode': mode,
                    'num_retrieved': 0,
                    'use_lexical': request.use_lexical
                }
            )
        
        print(f"  📥 Retrieved {len(documents)} documents")
        
        # 2. Filter by relevance
        filtered = _RELEVANCE_FILTER.filter_results(request.query, documents)
        filtered = filtered[:FINAL_K]  # Keep top K
        
        print(f"  ✅ Filtered to {len(filtered)} relevant documents")
        
        # 3. Generate answer
        print(f"  🤖 Generating answer...")
        result = _ANSWER_GENERATOR.generate(request.query, filtered)
        
        # 4. Format response
        retrieved_docs = [
            RetrievedDocument(
                id=doc['id'],
                title=doc['title'],
                text=doc['text'],
                summary=doc['summary'],
                url=doc['url'],
                score=doc.get('final_score', doc.get('score', 0)),
                relevance=doc.get('relevance', 'unknown')
            )
            for doc in filtered
        ]
        
        response = QueryResponse(
            answer=result['answer'],
            sources=[
                Source(
                    title=s['title'],
                    url=s['url'],
                    score=s['score']
                )
                for s in result['sources']
            ],
            confidence=result['confidence'],
            retrieved=retrieved_docs,
            metadata={
                'mode': mode,
                'num_retrieved': len(documents),
                'num_filtered': len(filtered),
                'use_lexical': request.use_lexical,
                'reasoning': result['reasoning']
            }
        )
        
        print(f"  ✅ Response ready (confidence: {result['confidence']})")
        print(f"{'='*60}\n")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    # Count total documents
    total_docs = 0
    if _STANDARD_RETRIEVER:
        total_docs = _STANDARD_RETRIEVER.semantic.index.ntotal
    elif _ULTRA_RETRIEVER:
        total_docs = _ULTRA_RETRIEVER.semantic.index.ntotal
    
    return HealthResponse(
        status="healthy",
        mode=DEFAULT_MODE,
        index_available={
            'standard': _STANDARD_RETRIEVER is not None,
            'ultra': _ULTRA_RETRIEVER is not None
        },
        total_documents=total_docs
    )


@app.get("/config", response_model=ConfigResponse)
async def config():
    """Get current configuration."""
    cfg = get_config(DEFAULT_MODE)
    
    return ConfigResponse(
        mode=cfg['mode'],
        dimensions=cfg['dimensions'],
        compressed=cfg['compressed'],
        top_k=FINAL_K,
        use_lexical_default=False
    )


@app.get("/stats")
async def stats():
    """Get detailed statistics."""
    stats = {
        'default_mode': DEFAULT_MODE,
        'modes': {}
    }
    
    if _STANDARD_RETRIEVER:
        stats['modes']['standard'] = _STANDARD_RETRIEVER.get_stats()
    
    if _ULTRA_RETRIEVER:
        stats['modes']['ultra'] = _ULTRA_RETRIEVER.get_stats()
    
    return stats


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload
    )
