"""
RAG Service with support for both standard and ultra-compressed retrieval modes.
Enhanced to provide comprehensive context and intelligent answer generation.
"""

from typing import Optional, Dict, List
from src.retriever.hybrid_retriever import UnifiedHybridRetriever
from src.utils.config import get_config
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global retriever instance (initialized on first use)
_retriever_cache = {}

def get_retriever(mode: Optional[str] = None) -> UnifiedHybridRetriever:
    """
    Get or create retriever instance with caching.
    
    Args:
        mode: 'standard', 'ultra', or None for auto-detect
    
    Returns:
        UnifiedHybridRetriever instance
    """
    global _retriever_cache
    
    # Determine mode
    if mode is None:
        cfg = get_config()
        mode = 'ultra' if cfg.get('use_ultra_compressed', False) else 'standard'
    
    # Return cached retriever if available
    if mode in _retriever_cache:
        return _retriever_cache[mode]
    
    # Create new retriever
    print(f"Initializing retriever in {mode} mode...")
    retriever = UnifiedHybridRetriever(force_mode=mode)
    _retriever_cache[mode] = retriever
    
    return retriever


def format_context_for_llm(docs: List[Dict]) -> str:
    """
    Format retrieved documents into comprehensive context for LLM.
    Includes both summaries and full text content to maximize information availability.
    
    Args:
        docs: List of retrieved documents
    
    Returns:
        Formatted context string with both summaries and full content
    """
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Combine both summary and text for comprehensive context
        summary = doc.get('summary', '').strip()
        text = doc.get('text', '').strip()
        
        # Build comprehensive content
        content_pieces = []
        if summary:
            content_pieces.append(f"Summary: {summary}")
        if text:
            # Use full text or reasonable excerpt
            text_excerpt = text if len(text) <= 1000 else text[:1000] + "..."
            content_pieces.append(f"Full Content: {text_excerpt}")
        
        content = "\n\n".join(content_pieces) if content_pieces else "No content available"
        
        context_parts.append(
            f"[Source {i}: {doc['meta']['title']}]\n"
            f"{content}\n"
            f"URL: {doc['meta']['url']}\n"
        )
    
    return "\n---\n".join(context_parts)


def extract_sources(docs: List[Dict]) -> List[Dict]:
    """
    Extract source information from retrieved documents.
    
    Args:
        docs: List of retrieved documents
    
    Returns:
        List of source info dictionaries
    """
    sources = []
    for doc in docs:
        sources.append({
            "title": doc['meta']['title'],
            "url": doc['meta']['url'],
            "domain": doc['meta'].get('primary_domain', 'general')
        })
    return sources


def generate_answer(query: str, context: str, cfg: Dict) -> Dict:
    """
    Generate answer using LLM with enhanced prompting for comprehensive responses.
    
    Args:
        query: User query
        context: Formatted context from retrieved documents
        cfg: Configuration dictionary
    
    Returns:
        Dictionary with answer, confidence, and reasoning
    """
    system_prompt = """You are an intelligent Wikipedia assistant. Your task is to:
1. Answer questions comprehensively using ALL available information from the provided context
2. Synthesize information from both summaries and full content sections
3. Cite sources by referring to [Source N] when using information
4. Make intelligent inferences and connections from the available information
5. Only state that information is unavailable if you've thoroughly reviewed all context and found nothing relevant
6. Be concise but comprehensive in your answers
7. Provide a confidence level (High/Medium/Low) based on the completeness and relevance of available information

IMPORTANT: The context includes both summaries and full content. Always review both sections carefully before concluding information is unavailable.

Format your response as:
ANSWER: [Your detailed answer with citations, synthesizing all available information]
CONFIDENCE: [High/Medium/Low]
REASONING: [Brief explanation of your confidence level and what information you used]"""
    
    user_prompt = f"""Question: {query}

Context from Wikipedia:
{context}

Please answer the question using the context above."""
    
    try:
        response = client.chat.completions.create(
            model=cfg.get("openai_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 1024)
        )
        
        content = response.choices[0].message.content
        
        # Parse response
        answer = ""
        confidence = "Medium"
        reasoning = ""
        
        for line in content.split('\n'):
            if line.startswith('ANSWER:'):
                answer = line.replace('ANSWER:', '').strip()
                # Capture multi-line answer
                answer_lines = [answer]
                idx = content.index(line) + len(line)
                for next_line in content[idx:].split('\n'):
                    if next_line.startswith('CONFIDENCE:') or next_line.startswith('REASONING:'):
                        break
                    if next_line.strip():
                        answer_lines.append(next_line.strip())
                answer = ' '.join(answer_lines)
            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip()
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
                # Capture multi-line reasoning
                reasoning_lines = [reasoning]
                idx = content.index(line) + len(line)
                for next_line in content[idx:].split('\n'):
                    if next_line.strip():
                        reasoning_lines.append(next_line.strip())
                reasoning = ' '.join(reasoning_lines)
        
        # Fallback if parsing fails
        if not answer:
            answer = content
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "answer": "I apologize, but I encountered an error generating an answer.",
            "confidence": "Low",
            "reasoning": f"Error: {str(e)}"
        }


def run_rag_pipeline(
    query: str,
    top_k: int = 5,
    use_filtering: bool = True,
    evaluate_sources: bool = True,
    retrieval_mode: Optional[str] = None
) -> Dict:
    """
    Run the complete RAG pipeline.
    
    Args:
        query: User query
        top_k: Number of documents to retrieve
        use_filtering: Enable domain-based filtering
        evaluate_sources: Enable source quality evaluation
        retrieval_mode: 'standard', 'ultra', or None for auto-detect
    
    Returns:
        Complete response with answer, sources, and metadata
    """
    cfg = get_config()
    
    # Get retriever (cached)
    retriever = get_retriever(mode=retrieval_mode)
    
    # Retrieve documents
    print(f"\n{'='*60}")
    print(f"RAG Pipeline - Mode: {retriever.mode.upper()}")
    print(f"{'='*60}")
    
    docs = retriever.retrieve(
        query=query,
        final_k=top_k,
        use_filtering=use_filtering,
        evaluate_sources=evaluate_sources
    )
    
    if not docs:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": [],
            "confidence": "Low",
            "reasoning": "No relevant documents were found.",
            "retrieved": [],
            "metadata": {
                "retrieval_mode": retriever.mode,
                "num_retrieved": 0,
                "filtering_enabled": use_filtering,
                "evaluation_enabled": evaluate_sources
            }
        }
    
    # Format context and extract sources
    context = format_context_for_llm(docs)
    sources = extract_sources(docs)
    
    # Generate answer
    llm_response = generate_answer(query, context, cfg)
    
    # Format retrieved documents for response
    retrieved_docs = []
    for doc in docs:
        retrieved_docs.append({
            "id": doc["id"],
            "title": doc["meta"]["title"],
            "text": doc.get("text", ""),
            "summary": doc.get("summary", ""),
            "text_length": doc["meta"].get("text_length", 0),
            "summary_length": doc["meta"].get("summary_length", 0),
            "url": doc["meta"]["url"],
            "score": doc.get("score", 0),
            "final_score": doc.get("final_score", doc.get("score", 0)),
            "domain": doc["meta"].get("primary_domain", "general"),
            "evaluation": doc.get("evaluation", {}),
            "debug": doc.get("debug", {})
        })
    
    # Build metadata
    metadata = {
        "retrieval_mode": retriever.mode,
        "num_retrieved": len(docs),
        "filtering_enabled": use_filtering,
        "evaluation_enabled": evaluate_sources,
        "model": cfg.get("openai_model", "gpt-4o-mini"),
        "avg_relevance": sum(d.get("final_score", d.get("score", 0)) for d in docs) / len(docs) if docs else 0,
    }
    
    # Add mode-specific metadata
    if retriever.mode == "ultra":
        if hasattr(retriever.ultra, 'get_stats'):
            metadata["ultra_stats"] = {
                "dimensions": retriever.ultra.pca_model['n_components'],
                "nprobe": retriever.ultra.index.nprobe
            }
    
    return {
        "answer": llm_response["answer"],
        "sources": sources,
        "confidence": llm_response["confidence"],
        "reasoning": llm_response["reasoning"],
        "retrieved": retrieved_docs,
        "metadata": metadata
    }


def clear_retriever_cache():
    """Clear cached retriever instances (useful for testing)"""
    global _retriever_cache
    _retriever_cache.clear()
    print("✓ Retriever cache cleared")