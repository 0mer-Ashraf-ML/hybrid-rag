# src/evaluation/source_evaluator.py
import re
from typing import Dict, List, Tuple
from openai import OpenAI
import os
import json
from collections import Counter
import math

class AdvancedSourceEvaluator:
    
    def __init__(self, use_llm_evaluation: bool = False):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_llm = use_llm_evaluation and self.api_key is not None
        
        # Stop words for better keyword matching
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with',
            'tell', 'me', 'about', 'what', 'who', 'when', 'where',
            'why', 'how', 'which', 'can', 'could', 'would', 'should',
            'do', 'does', 'did', 'have', 'been', 'being'
        }
        
        # Query type patterns for adaptive evaluation
        self.query_patterns = {
            'definition': r'\b(what is|define|meaning of|definition)\b',
            'how_to': r'\b(how to|how do|how does|steps to)\b',
            'comparison': r'\b(difference between|compare|versus|vs)\b',
            'causes': r'\b(why|cause|reason|what causes)\b',
            'list': r'\b(list|types of|kinds of|examples of)\b',
            'history': r'\b(history|when|origin|evolution)\b',
            'process': r'\b(process|procedure|mechanism|works)\b',
            'factual': r'\b(who|when|where)\b'
        }
    
    def evaluate_sources(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Comprehensive evaluation combining original and advanced methods.
        """
        # Identify query type and extract key concepts
        query_type = self._identify_query_type(query)
        key_concepts = self._extract_key_concepts(query)
        
        print(f"  📋 Query analysis:")
        print(f"     Type: {query_type}")
        print(f"     Key concepts: {key_concepts[:5]}")  # Show first 5
        
        evaluated = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Calculate all evaluation scores
            scores = {
                # Original scores (kept from first implementation)
                "keyword_relevance": self._keyword_relevance_score(query, doc),
                "domain_relevance": self._domain_relevance_score(query, doc, query_type),
                "content_quality": self._content_quality_score(doc),
                "retrieval_confidence": self._normalize_retrieval_score(doc.get("score", 0)),
                
                # Advanced scores (new)
                "semantic_relevance": self._semantic_relevance_score(query, doc, key_concepts),
                "content_coverage": self._content_coverage_score(query_type, doc, key_concepts),
                "source_authority": self._source_authority_score(doc),
                "information_density": self._information_density_score(doc),
                "freshness": self._freshness_score(doc)
            }
            
            # Adaptive weighting based on query type
            weights = self._get_adaptive_weights(query_type)
            
            # Calculate base final score
            final_score = sum(scores[k] * weights[k] for k in scores)
            
            # Optional LLM-based relevance boost
            llm_evaluation = None
            if self.use_llm and i <= 5:  # Only evaluate top 5 with LLM
                print(f"     🤖 LLM evaluating source {i}...")
                llm_evaluation = self._evaluate_with_llm_advanced(query, doc, query_type)
                # Blend LLM score (15% weight to not override heuristics)
                if llm_evaluation:
                    final_score = 0.85 * final_score + 0.15 * llm_evaluation['llm_relevance']
            
            # Add evaluation to document
            doc["evaluation"] = {
                "scores": scores,
                "weights": weights,
                "base_score": sum(scores[k] * weights[k] for k in scores),
                "final_score": final_score,
                "relevance_level": self._get_relevance_level(final_score),
                "should_use": final_score >= 0.4,
                "query_type": query_type,
                "key_concepts_found": self._count_concepts_in_doc(doc, key_concepts),
                "llm_evaluation": llm_evaluation,
                "evaluation_reason": self._generate_evaluation_reason(scores, query_type, doc)
            }
            
            evaluated.append(doc)
        
        return evaluated
    
    def _identify_query_type(self, query: str) -> str:
        """
        Identify the type of query to adapt evaluation criteria.
        """
        query_lower = query.lower()
        
        for qtype, pattern in self.query_patterns.items():
            if re.search(pattern, query_lower):
                return qtype
        
        return 'general'
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query (excluding stop words).
        """
        # Tokenize and clean
        words = re.findall(r'\b[a-z]+\b', query.lower())
        
        # Remove stop words
        key_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Extract phrases (bigrams)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        return key_words + phrases
    
    # ===================================================================
    # ORIGINAL EVALUATION METHODS (from first implementation)
    # ===================================================================
    
    def _keyword_relevance_score(self, query: str, doc: Dict) -> float:
        """
        Original: Calculate keyword overlap between query and document.
        Checks term overlap with title bonus.
        """
        query_terms = set(query.lower().split())
        
        # Remove stop words from query terms
        query_terms = {term for term in query_terms if term not in self.stop_words}
        
        # Check in title, text, and summary
        title = doc['meta'].get('title', '').lower()
        text = doc.get('text', '').lower()
        summary = doc.get('summary', '').lower()
        
        combined = f"{title} {summary} {text[:500]}"
        doc_terms = set(combined.split())
        
        if not query_terms:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_terms & doc_terms) / len(query_terms)
        
        # Bonus for query terms in title
        title_matches = sum(1 for term in query_terms if term in title)
        title_bonus = (title_matches / len(query_terms)) * 0.3
        
        return min(overlap + title_bonus, 1.0)
    
    def _domain_relevance_score(self, query: str, doc: Dict, query_type: str = None) -> float:
        """
        Original: Score based on domain match between query and document.
        Enhanced with query type awareness.
        """
        doc_domain = doc['meta'].get('primary_domain', 'general')
        
        # Domain-specific boost
        if doc_domain != 'general':
            score = 0.8
        else:
            score = 0.5
        
        # Query type alignment with domain
        type_domain_match = {
            'technology': ['definition', 'how_to', 'process'],
            'science': ['definition', 'causes', 'process'],
            'history': ['history', 'factual'],
            'medicine': ['definition', 'causes', 'process']
        }
        
        if query_type and doc_domain in type_domain_match:
            if query_type in type_domain_match[doc_domain]:
                score += 0.1
        
        return min(score, 1.0)
    
    def _content_quality_score(self, doc: Dict) -> float:
        """
        Original: Assess the quality of the document content.
        Checks length, structure, URL presence.
        """
        score = 0.5  # Base score
        
        text_length = doc['meta'].get('text_length', 0)
        summary_length = doc['meta'].get('summary_length', 0)
        
        # Ideal text length: 500-5000 characters
        if 500 <= text_length <= 5000:
            score += 0.3
        elif 200 <= text_length < 500:
            score += 0.2
        elif text_length > 5000:
            score += 0.2
        
        # Has summary (indicates structured content)
        if summary_length > 100:
            score += 0.2
        elif summary_length > 50:
            score += 0.1
        
        # Has URL (verifiable)
        if doc['meta'].get('url'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _normalize_retrieval_score(self, score: float) -> float:
        """
        Original: Normalize retrieval score to 0-1 range.
        Enhanced with sigmoid transformation.
        """
        # Apply sigmoid-like transformation to spread scores better
        normalized = 1 / (1 + math.exp(-5 * (score - 0.5)))
        return min(max(normalized, 0.0), 1.0)
    
    # ===================================================================
    # ADVANCED EVALUATION METHODS (new additions)
    # ===================================================================
    
    def _semantic_relevance_score(self, query: str, doc: Dict, key_concepts: List[str]) -> float:
        """
        Advanced: TF-IDF-like semantic relevance scoring.
        Position-aware with frequency consideration.
        """
        title = doc['meta'].get('title', '').lower()
        summary = doc.get('summary', '').lower()
        text_preview = doc.get('text', '')[:1000].lower()
        
        # Combine with different weights
        title_score = self._calculate_concept_match(title, key_concepts, weight=3.0)
        summary_score = self._calculate_concept_match(summary, key_concepts, weight=2.0)
        text_score = self._calculate_concept_match(text_preview, key_concepts, weight=1.0)
        
        # Weighted combination
        total_score = (title_score + summary_score + text_score) / 6.0
        
        return min(total_score, 1.0)
    
    def _calculate_concept_match(self, text: str, concepts: List[str], weight: float = 1.0) -> float:
        """
        Calculate how well concepts appear in text with position weighting.
        """
        if not text or not concepts:
            return 0.0
        
        score = 0.0
        text_length = len(text)
        
        for concept in concepts:
            if concept in text:
                # Find position (earlier = better)
                position = text.find(concept)
                position_score = 1.0 - (position / max(text_length, 1))
                
                # Count frequency
                frequency = text.count(concept)
                frequency_score = min(frequency / 3.0, 1.0)  # Cap at 3 occurrences
                
                score += (position_score + frequency_score) / 2.0
        
        normalized_score = score / max(len(concepts), 1)
        return min(normalized_score * weight, 1.0)
    
    def _content_coverage_score(self, query_type: str, doc: Dict, key_concepts: List[str]) -> float:
        
        text = doc.get('text', '')
        summary = doc.get('summary', '')
        combined = f"{summary} {text[:2000]}"
        
        score = 0.5  # Base score
        
        # Type-specific checks
        if query_type == 'definition':
            definition_patterns = [
                r'\bis\s+(a|an|the)\s+',
                r'refers to',
                r'defined as',
                r'means',
                r'represents',
                r'known as'
            ]
            matches = sum(1 for pattern in definition_patterns 
                         if re.search(pattern, combined[:500], re.IGNORECASE))
            score += min(matches / 4.0, 0.3)
        
        elif query_type == 'how_to':
            procedural_indicators = ['step', 'first', 'then', 'next', 'finally', 
                                   'process', 'method', 'procedure', 'guide']
            found = sum(1 for ind in procedural_indicators if ind in combined.lower())
            score += min(found / 6.0, 0.3)
        
        elif query_type == 'comparison':
            comparison_words = ['difference', 'unlike', 'whereas', 'compared', 
                              'versus', 'both', 'similar', 'contrast']
            found = sum(1 for word in comparison_words if word in combined.lower())
            score += min(found / 5.0, 0.3)
        
        elif query_type == 'causes':
            causal_words = ['because', 'due to', 'caused by', 'reason', 'result', 
                          'leads to', 'consequence']
            found = sum(1 for word in causal_words if word in combined.lower())
            score += min(found / 4.0, 0.3)
        
        elif query_type == 'list':
            # Look for list structures
            has_lists = bool(re.search(r'[\n•\-\*]\s*', text[:1000]))
            enumeration = bool(re.search(r'\d+\.', text[:1000]))
            if has_lists or enumeration:
                score += 0.3
        
        elif query_type == 'history':
            temporal_words = ['year', 'century', 'historical', 'originated', 
                            'founded', 'began', 'ancient', 'era', 'period']
            found = sum(1 for word in temporal_words if word in combined.lower())
            score += min(found / 5.0, 0.3)
        
        elif query_type == 'process':
            process_words = ['mechanism', 'works', 'function', 'operate', 
                           'system', 'cycle', 'sequence']
            found = sum(1 for word in process_words if word in combined.lower())
            score += min(found / 4.0, 0.3)
        
        # Concept coverage (universal across all query types)
        concepts_found = self._count_concepts_in_doc(doc, key_concepts)
        coverage_ratio = concepts_found / max(len(key_concepts), 1)
        score += coverage_ratio * 0.2
        
        return min(score, 1.0)
    
    def _source_authority_score(self, doc: Dict) -> float:
        """
        Evaluate source authority based on Wikipedia quality signals.
        """
        score = 0.5
        
        title = doc['meta'].get('title', '')
        categories = doc['meta'].get('categories', '').lower()
        url = doc['meta'].get('url', '')
        
        # Wikipedia specific authority signals
        authority_indicators = {
            'featured': 0.2,
            'good article': 0.15,
            'peer reviewed': 0.15,
            'verified': 0.1,
            'reliable': 0.1
        }
        
        for indicator, boost in authority_indicators.items():
            if indicator in categories:
                score += boost
        
        # Check for problematic indicators
        quality_issues = {
            'stub': -0.2,
            'cleanup': -0.15,
            'disputed': -0.25,
            'unreferenced': -0.2,
            'citation needed': -0.15,
            'neutrality': -0.1
        }
        
        for issue, penalty in quality_issues.items():
            if issue in categories:
                score += penalty
        
        # URL presence and validity
        if url and 'wikipedia.org' in url:
            score += 0.1
        
        # Title specificity (more specific titles often indicate better articles)
        title_words = len(title.split())
        if 2 <= title_words <= 5:
            score += 0.1
        elif title_words > 5:
            score += 0.05
        
        return max(0.0, min(score, 1.0))
    
    def _information_density_score(self, doc: Dict) -> float:
        """
        Measure information density and structure quality.
        Not too sparse, not too verbose.
        """
        text = doc.get('text', '')
        summary = doc.get('summary', '')
        
        if not text:
            return 0.0
        
        text_length = len(text)
        summary_length = len(summary)
        
        score = 0.5
        
        # Ideal text length: 1000-10000 characters (sweet spot)
        if 1000 <= text_length <= 10000:
            score += 0.3
        elif 500 <= text_length < 1000:
            score += 0.2
        elif text_length > 10000:
            score += 0.15
        elif text_length < 500:
            score += 0.05
        
        # Good summary indicates structure
        if 100 <= summary_length <= 500:
            score += 0.2
        elif summary_length > 50:
            score += 0.1
        
        # Sentence length distribution (readability check)
        sentences = re.split(r'[.!?]+', text[:2000])
        sentences = [s for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Ideal: 10-30 words per sentence
            if 10 <= avg_sentence_length <= 30:
                score += 0.1
            elif 8 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 35:
                score += 0.05
        
        return min(score, 1.0)
    
    def _freshness_score(self, doc: Dict) -> float:
        """
        Evaluate content freshness based on domain.
        Some domains need fresh info, others don't.
        """
        domain = doc['meta'].get('primary_domain', 'general')
        
        # Domains where freshness matters more
        time_sensitive = ['technology', 'medicine', 'politics', 'business']
        time_neutral = ['history', 'mathematics', 'philosophy', 'arts']
        
        if domain in time_sensitive:
            # Without timestamp, give neutral-low score
            return 0.6
        elif domain in time_neutral:
            # Historical/timeless topics don't need freshness
            return 0.9
        else:
            return 0.7
    
    def _get_adaptive_weights(self, query_type: str) -> Dict[str, float]:
        """
        Adaptive weights based on query type.
        Different query types prioritize different evaluation criteria.
        """
        # Default weights (balanced)
        default_weights = {
        "keyword_relevance": 0.08,       # Reduced from 0.12
        "domain_relevance": 0.10,
        "content_quality": 0.10,
        "retrieval_confidence": 0.15,    # Boosted from 0.13
        "semantic_relevance": 0.25,      # Boosted from 0.20
        "content_coverage": 0.12,        # Reduced from 0.15
        "source_authority": 0.10,
        "information_density": 0.05,
        "freshness": 0.05
    }
        if query_type == 'general' or query_type == 'factual':
            
            return {
                "keyword_relevance": 0.10,
                "domain_relevance": 0.08,
                "content_quality": 0.12,
                "retrieval_confidence": 0.18,    # Trust the retrieval!
                "semantic_relevance": 0.28,      # Semantic is key
                "content_coverage": 0.08,        # De-emphasize phrasing
                "source_authority": 0.10,
                "information_density": 0.04,
                "freshness": 0.02
            }   
        # Type-specific adjustments
        adjustments = {
            'definition': {
                "keyword_relevance": 0.15,
                "domain_relevance": 0.10,
                "content_quality": 0.12,
                "retrieval_confidence": 0.10,
                "semantic_relevance": 0.20,
                "content_coverage": 0.18,
                "source_authority": 0.10,
                "information_density": 0.03,
                "freshness": 0.02
            },
            'how_to': {
                "keyword_relevance": 0.10,
                "domain_relevance": 0.08,
                "content_quality": 0.10,
                "retrieval_confidence": 0.08,
                "semantic_relevance": 0.15,
                "content_coverage": 0.30,
                "source_authority": 0.08,
                "information_density": 0.08,
                "freshness": 0.03
            },
            'comparison': {
                "keyword_relevance": 0.12,
                "domain_relevance": 0.10,
                "content_quality": 0.12,
                "retrieval_confidence": 0.10,
                "semantic_relevance": 0.18,
                "content_coverage": 0.22,
                "source_authority": 0.08,
                "information_density": 0.06,
                "freshness": 0.02
            },
            'history': {
                "keyword_relevance": 0.12,
                "domain_relevance": 0.12,
                "content_quality": 0.12,
                "retrieval_confidence": 0.12,
                "semantic_relevance": 0.18,
                "content_coverage": 0.15,
                "source_authority": 0.15,
                "information_density": 0.04,
                "freshness": 0.00
            },
            'causes': {
                "keyword_relevance": 0.12,
                "domain_relevance": 0.10,
                "content_quality": 0.10,
                "retrieval_confidence": 0.10,
                "semantic_relevance": 0.18,
                "content_coverage": 0.25,
                "source_authority": 0.08,
                "information_density": 0.05,
                "freshness": 0.02
            },
            'list': {
                "keyword_relevance": 0.15,
                "domain_relevance": 0.08,
                "content_quality": 0.10,
                "retrieval_confidence": 0.10,
                "semantic_relevance": 0.15,
                "content_coverage": 0.20,
                "source_authority": 0.08,
                "information_density": 0.12,
                "freshness": 0.02
            }
        }
        
        return adjustments.get(query_type, default_weights)
    
    def _count_concepts_in_doc(self, doc: Dict, concepts: List[str]) -> int:
        """
        Count how many key concepts appear in document.
        """
        text = f"{doc.get('summary', '')} {doc.get('text', '')[:1000]}".lower()
        return sum(1 for concept in concepts if concept in text)
    
    def _generate_evaluation_reason(self, scores: Dict, query_type: str, doc: Dict) -> str:
        """
        Generate human-readable explanation of evaluation.
        """
        strengths = []
        weaknesses = []
        
        for metric, score in scores.items():
            if score >= 0.75:
                strengths.append(metric.replace('_', ' '))
            elif score < 0.35:
                weaknesses.append(metric.replace('_', ' '))
        
        reason = f"Query type: {query_type}. "
        
        if strengths:
            reason += f"Strong: {', '.join(strengths[:3])}. "
        
        if weaknesses:
            reason += f"Weak: {', '.join(weaknesses[:2])}."
        else:
            reason += "Balanced quality across metrics."
        
        return reason
    
    def _get_relevance_level(self, score: float) -> str:
        """
        Enhanced relevance levels with more granularity.
        """
        if score >= 0.80:
            return "Excellent"
        elif score >= 0.65:
            return "High"
        elif score >= 0.50:
            return "Medium"
        elif score >= 0.35:
            return "Low"
        else:
            return "Very Low"
    
    def filter_by_relevance(self, evaluated_docs: List[Dict], threshold: float = 0.4) -> List[Dict]:
        """
        Filter with enhanced logging showing rejection reasons.
        """
        filtered = [
            doc for doc in evaluated_docs 
            if doc["evaluation"]["final_score"] >= threshold
        ]
        
        if len(filtered) < len(evaluated_docs):
            rejected = len(evaluated_docs) - len(filtered)
            print(f"  🎯 Source evaluation: {len(filtered)}/{len(evaluated_docs)} passed threshold ({threshold})")
            
            # Show why docs were rejected (only if verbose)
            for doc in evaluated_docs:
                if doc["evaluation"]["final_score"] < threshold:
                    title = doc['meta'].get('title', 'Unknown')
                    score = doc["evaluation"]["final_score"]
                    level = doc["evaluation"]["relevance_level"]
                    print(f"     ❌ '{title[:45]}...' - Score: {score:.3f} ({level})")
        else:
            print(f"  ✅ All {len(filtered)} sources passed evaluation")
        
        return filtered
    
    def rank_by_relevance(self, evaluated_docs: List[Dict]) -> List[Dict]:
        """
        Enhanced ranking with multiple tie-breakers.
        """
        return sorted(
            evaluated_docs,
            key=lambda x: (
                x["evaluation"]["final_score"],  # Primary sort
                x.get("score", 0),  # Tie-breaker: retrieval score
                x["evaluation"]["scores"].get("semantic_relevance", 0),  # Second tie-breaker
                x["evaluation"]["scores"].get("source_authority", 0)  # Third tie-breaker
            ),
            reverse=True
        )
    
    def _evaluate_with_llm_advanced(self, query: str, doc: Dict, query_type: str) -> Dict:
        """
        Advanced LLM-based evaluation with query context.
        Optional but powerful for complex queries.
        """
        if not self.api_key:
            return None
        
        try:
            client = OpenAI(api_key=self.api_key)
            
            title = doc['meta'].get('title', 'Unknown')
            summary = doc.get('summary', '')[:600]
            text_preview = doc.get('text', '')[:400]
            domain = doc['meta'].get('primary_domain', 'general')
            
            prompt = f"""You are an expert information retrieval evaluator. Assess how well this source answers the query.

Query: "{query}"
Query Type: {query_type}
Domain: {domain}

Source:
Title: {title}
Summary: {summary}
Content Preview: {text_preview}

Evaluate on a scale of 0.0 (irrelevant) to 1.0 (perfect match):
1. Does it directly address the query intent?
2. Is the information comprehensive for this query type?
3. Does it provide authoritative, well-structured information?

Return JSON with:
- relevance_score (float 0-1): overall relevance
- addresses_query (bool): directly answers the question
- comprehensiveness (float 0-1): how complete the answer is
- reasoning (string, 1-2 sentences): brief explanation"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=250
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "llm_relevance": result.get("relevance_score", 0.5),
                "llm_addresses_query": result.get("addresses_query", False),
                "llm_comprehensiveness": result.get("comprehensiveness", 0.5),
                "llm_reasoning": result.get("reasoning", "")
            }
        
        except Exception as e:
            print(f"     ⚠️  LLM evaluation error: {e}")
            return None

# Backward compatibility alias
SourceEvaluator = AdvancedSourceEvaluator