# src/utils/llm_generator.py
import os
import json
from openai import OpenAI
from src.utils.config import get_config
def generate_answer_with_llm(query: str, retrieved: list) -> dict:
    """
    Generate answer using OpenAI API with retrieved documents.
    Now properly uses both text and summary fields.
    """
    if not retrieved:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": [],
            "reasoning": "No documents were retrieved for this query."
        }
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
            "sources": [doc['meta'].get('title', 'Unknown') for doc in retrieved[:5]],
            "reasoning": "API key missing"
        }
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved[:5], 1):
            title = doc['meta'].get('title', 'Unknown Document')
            url = doc['meta'].get('url', '')
            domain = doc['meta'].get('primary_domain', 'general')
            
            # Get text and summary directly from top-level fields
            full_text = doc.get('text', '')
            summary = doc.get('summary', '')
            
            # Build context intelligently
            if summary and full_text:
                # Use summary + first part of text
                context_text = f"Summary: {summary}\n\nDetailed Content: {full_text[:2000]}"
            elif summary:
                context_text = f"Summary: {summary}"
            elif full_text:
                context_text = full_text[:2500]
            else:
                context_text = "No content available"
            
            # Add evaluation info if available
            evaluation = doc.get('evaluation', {})
            relevance = evaluation.get('relevance_level', 'Unknown')
            eval_score = evaluation.get('final_score', 0)
            
            context_parts.append(
                f"[Source {i}] {title}\n"
                f"Domain: {domain} | Relevance: {relevance} ({eval_score:.2f})\n"
                f"URL: {url}\n"
                f"Content:\n{context_text}\n"
            )
            
            sources.append({
                "title": title,
                "url": url,
                "domain": domain
            })
        
        context = "\n" + "="*80 + "\n\n".join(context_parts)
        
        # System message
        system_msg = {
            "role": "system",
            "content": (
                "You are an intelligent research assistant that synthesizes information from Wikipedia sources.\n\n"
                "Your responsibilities:\n"
                "1. Carefully read all provided sources (summaries and detailed content)\n"
                "2. Synthesize information from multiple sources when they complement each other\n"
                "3. Prioritize sources with higher relevance scores\n"
                "4. Provide accurate, comprehensive, and well-structured answers\n"
                "5. Cite which source numbers you used in your answer\n"
                "6. If information is incomplete or contradictory, acknowledge this\n"
                "7. Write in a clear, informative style suitable for educational purposes\n\n"
                "Response format (JSON):\n"
                "{\n"
                '  "answer": "Your comprehensive answer (2-4 paragraphs)",\n'
                '  "sources_used": [1, 2, 3],  // Source numbers you referenced\n'
                '  "confidence": "low/medium/high",\n'
                '  "reasoning": "Brief explanation of how you derived the answer"\n'
                "}"
            )
        }
        
        # User message
        user_msg = {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Available Sources (with relevance scores):\n"
                f"{context}\n\n"
                f"Please provide a comprehensive answer based on these sources."
            )
        }
        
        # Call OpenAI API
        cfg = get_config()
        response = client.chat.completions.create(
            model=cfg.get("openai_model", "gpt-4o-mini"),
            messages=[system_msg, user_msg],
            temperature=cfg.get("temperature", 0.2),
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        # Map source numbers to actual sources
        sources_used_indices = result.get("sources_used", [])
        sources_used = []
        for i in sources_used_indices:
            if 0 < i <= len(sources):
                sources_used.append(sources[i-1])
        
        # Fallback to top 3 if no sources specified
        if not sources_used:
            sources_used = sources[:3]
        
        return {
            "answer": result.get("answer", "Unable to generate answer."),
            "sources": sources_used,
            "confidence": result.get("confidence", "medium"),
            "reasoning": result.get("reasoning", "Answer synthesized from provided sources."),
            "all_retrieved_sources": sources
        }
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return {
            "answer": content if 'content' in locals() else "Error generating answer.",
            "sources": [doc['meta'].get('title', 'Unknown') for doc in retrieved[:3]],
            "confidence": "low",
            "reasoning": "JSON parsing error"
        }
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [doc['meta'].get('title', 'Unknown') for doc in retrieved[:3]],
            "confidence": "low",
            "reasoning": f"API error: {str(e)}"
        }