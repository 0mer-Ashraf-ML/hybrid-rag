# src/utils/query_classifier.py
import re
from typing import List, Tuple

class QueryClassifier:
    """
    Classify queries into domains to filter search space.
    """
    
    def __init__(self):
        self.domain_keywords = {
            "science": ["science", "physics", "chemistry", "biology", "scientific", 
                       "atom", "molecule", "cell", "experiment", "theory"],
            "technology": ["technology", "computer", "software", "app", "algorithm",
                          "ai", "machine learning", "code", "programming", "internet"],
            "medicine": ["medicine", "health", "disease", "treatment", "symptom",
                        "doctor", "hospital", "drug", "medical", "cure"],
            "history": ["history", "historical", "ancient", "war", "century",
                       "when did", "who was", "battle", "empire"],
            "geography": ["country", "city", "capital", "continent", "where is",
                         "located", "population", "geography", "region"],
            "arts": ["art", "music", "painting", "artist", "movie", "film",
                    "book", "novel", "literature", "poetry"],
            "sports": ["sport", "game", "team", "player", "championship",
                      "football", "basketball", "cricket", "olympic"],
            "politics": ["politics", "government", "president", "election",
                        "law", "democracy", "parliament", "minister"],
            "business": ["business", "company", "market", "economy", "finance",
                        "trade", "industry", "stock", "startup"],
            "mathematics": ["math", "calculate", "equation", "formula", "theorem",
                           "algebra", "geometry", "statistics", "number"],
            "philosophy": ["philosophy", "philosopher", "ethics", "meaning of",
                          "existential", "moral", "belief"],
            "entertainment": ["celebrity", "actor", "show", "series", "entertainment",
                             "star", "famous"]
        }
    
    def classify_query(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify query into domains.
        
        Returns:
            List of (domain, confidence_score) tuples
        """
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Exact word match gets higher score
                    if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        # If no matches, return general
        if not domain_scores:
            return [("general", 0.5)]
        
        # Normalize scores
        max_score = max(domain_scores.values())
        normalized = [(d, s / max_score) for d, s in domain_scores.items()]
        
        # Sort by score and return top-k
        sorted_domains = sorted(normalized, key=lambda x: x[1], reverse=True)
        return sorted_domains[:top_k]