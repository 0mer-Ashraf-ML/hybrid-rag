# src/evaluation/criteria.py
from typing import Dict, List
import re
from datetime import datetime

class SourceEvaluator:
    """
    Evaluates retrieved documents on multiple quality dimensions.
    """
    
    def __init__(self):
        # Weights for different criteria (sum to 1.0)
        self.weights = {
            "relevance": 0.30,      # How relevant to the query
            "completeness": 0.20,   # How complete/detailed
            "credibility": 0.25,    # How credible (categories, length, etc.)
            "freshness": 0.10,      # How recent (if available)
            "clarity": 0.15,        # How clear/readable
        }
    
    def evaluate_document(self, doc: Dict, query: str) -> Dict:
        """
        Evaluate a single document and return scores.
        
        Args:
            doc: Document dict with 'text', 'meta', 'score'
            query: Original search query
        
        Returns:
            Dict with individual scores and final evaluation score
        """
        scores = {}
        
        # 1. Relevance Score (based on retrieval score + keyword overlap)
        scores["relevance"] = self._score_relevance(doc, query)
        
        # 2. Completeness Score (based on length and structure)
        scores["completeness"] = self._score_completeness(doc)
        
        # 3. Credibility Score (based on metadata indicators)
        scores["credibility"] = self._score_credibility(doc)
        
        # 4. Freshness Score (if timestamp available)
        scores["freshness"] = self._score_freshness(doc)
        
        # 5. Clarity Score (readability)
        scores["clarity"] = self._score_clarity(doc)
        
        # Calculate weighted final score
        final_score = sum(
            scores[criterion] * self.weights[criterion]
            for criterion in self.weights
        )
        
        return {
            "individual_scores": scores,
            "final_score": final_score,
            "evaluation_details": self._get_evaluation_summary(scores, final_score)
        }
    
    def _score_relevance(self, doc: Dict, query: str) -> float:
        """
        Score based on:
        - Retrieval score (from hybrid retriever)
        - Query term overlap
        - Title relevance
        """
        score = 0.0
        
        # Use retrieval score (normalized to 0-1)
        retrieval_score = doc.get("score", 0.0)
        score += min(retrieval_score, 1.0) * 0.5
        
        # Query term overlap
        query_terms = set(query.lower().split())
        text_terms = set(doc["text"].lower().split())
        title = doc["meta"].get("title", "").lower()
        
        if query_terms:
            # Term overlap in text
            text_overlap = len(query_terms & text_terms) / len(query_terms)
            score += text_overlap * 0.3
            
            # Query terms in title (bonus)
            title_overlap = sum(1 for term in query_terms if term in title) / len(query_terms)
            score += title_overlap * 0.2
        
        return min(score, 1.0)
    
    def _score_completeness(self, doc: Dict) -> float:
        """
        Score based on:
        - Text length (not too short, not too long)
        - Presence of structured information
        - Summary quality
        """
        score = 0.0
        
        text_length = len(doc["text"])
        
        # Optimal length range: 500-3000 characters
        if text_length < 200:
            score += 0.3  # Too short
        elif 200 <= text_length < 500:
            score += 0.5
        elif 500 <= text_length <= 3000:
            score += 1.0  # Ideal
        elif 3000 < text_length <= 5000:
            score += 0.8
        else:
            score += 0.6  # Very long
        
        # Check for structured content (lists, sections)
        text = doc["text"]
        has_structure = (
            text.count('\n') > 2 or  # Multiple paragraphs
            bool(re.search(r'\d+\.', text)) or  # Numbered lists
            bool(re.search(r'[•\-\*]', text))  # Bullet points
        )
        
        if has_structure:
            score += 0.2
        
        # Summary presence
        summary_length = doc["meta"].get("summary_length", 0)
        if summary_length > 50:
            score += 0.3
        
        return min(score / 1.5, 1.0)
    
    def _score_credibility(self, doc: Dict) -> float:
        """
        Score based on:
        - Article categories (quality indicators)
        - Text length (more comprehensive = more credible)
        - URL presence
        """
        score = 0.5  # Base score
        
        categories = doc["meta"].get("categories", "").lower()
        
        # High-quality indicators
        quality_indicators = [
            "featured", "good article", "verified", "peer reviewed"
        ]
        if any(indicator in categories for indicator in quality_indicators):
            score += 0.3
        
        # Low-quality indicators
        low_quality = ["stub", "cleanup", "disputed", "unreferenced"]
        if any(indicator in categories for indicator in low_quality):
            score -= 0.3
        
        # Text length as credibility proxy
        text_length = doc["meta"].get("text_length", 0)
        if text_length > 2000:
            score += 0.2
        elif text_length > 5000:
            score += 0.3
        
        # Has URL (verifiable)
        if doc["meta"].get("url"):
            score += 0.1
        
        return max(0.0, min(score, 1.0))
    
    def _score_freshness(self, doc: Dict) -> float:
        """
        Score based on recency (if available).
        For Wikipedia, we may not have this, so return neutral.
        """
        # If you add timestamp metadata in the future, implement here
        # For now, return neutral score
        return 0.5
    
    def _score_clarity(self, doc: Dict) -> float:
        """
        Score based on readability:
        - Sentence length
        - Word complexity
        - Paragraph structure
        """
        text = doc["text"]
        
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Average sentence length (optimal: 15-25 words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
            
            if 15 <= avg_words_per_sentence <= 25:
                score += 0.3
            elif 10 <= avg_words_per_sentence < 15 or 25 < avg_words_per_sentence <= 30:
                score += 0.2
            else:
                score += 0.1
        
        # Check for excessive jargon (very long words)
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 6:  # Simple language
                score += 0.2
            elif avg_word_length > 8:  # Complex language
                score -= 0.1
        
        return max(0.0, min(score, 1.0))
    
    def _get_evaluation_summary(self, scores: Dict, final_score: float) -> str:
        """Generate human-readable evaluation summary."""
        if final_score >= 0.8:
            quality = "Excellent"
        elif final_score >= 0.6:
            quality = "Good"
        elif final_score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Find strongest and weakest criteria
        strongest = max(scores.items(), key=lambda x: x[1])
        weakest = min(scores.items(), key=lambda x: x[1])
        
        return (
            f"Quality: {quality} ({final_score:.2f}). "
            f"Strongest: {strongest[0]} ({strongest[1]:.2f}), "
            f"Weakest: {weakest[0]} ({weakest[1]:.2f})"
        )
    
    def filter_by_threshold(self, evaluated_docs: List[Dict], threshold: float = 0.4) -> List[Dict]:
        """
        Filter documents below quality threshold.
        
        Args:
            evaluated_docs: List of docs with evaluation scores
            threshold: Minimum score to keep (0-1)
        
        Returns:
            Filtered list of documents
        """
        return [
            doc for doc in evaluated_docs
            if doc["evaluation"]["final_score"] >= threshold
        ]
    
    def rank_by_evaluation(self, evaluated_docs: List[Dict]) -> List[Dict]:
        """
        Re-rank documents by evaluation score.
        """
        return sorted(
            evaluated_docs,
            key=lambda x: x["evaluation"]["final_score"],
            reverse=True
        )