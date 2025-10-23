# scripts/enhance_metadata.py
import pickle
import re
from pathlib import Path
from collections import defaultdict

def categorize_article(title: str, categories: str, text: str) -> dict:
    """
    Categorize Wikipedia article into domain and extract metadata.
    """
    categories_lower = categories.lower()
    text_lower = text.lower()
    title_lower = title.lower()
    
    # Define domain keywords
    domains = {
        "science": ["science", "physics", "chemistry", "biology", "astronomy", 
                   "scientific", "scientist", "research", "laboratory"],
        "technology": ["technology", "computer", "software", "internet", "programming",
                      "algorithm", "artificial intelligence", "machine learning", "engineering"],
        "medicine": ["medicine", "medical", "health", "disease", "treatment", "hospital",
                    "doctor", "patient", "clinical", "pharmaceutical"],
        "history": ["history", "historical", "ancient", "medieval", "war", "century",
                   "historian", "archaeological"],
        "geography": ["geography", "country", "city", "continent", "ocean", "mountain",
                     "river", "region", "geographic"],
        "arts": ["art", "artist", "painting", "music", "literature", "poetry", "novel",
                "theater", "film", "cinema"],
        "sports": ["sport", "athlete", "game", "championship", "olympic", "football",
                  "basketball", "cricket", "tennis"],
        "politics": ["politic", "government", "president", "minister", "parliament",
                    "election", "democracy", "law"],
        "business": ["business", "company", "corporation", "economy", "finance", "market",
                    "trade", "industry", "commerce"],
        "mathematics": ["mathematics", "mathematical", "theorem", "equation", "algebra",
                       "geometry", "calculus", "statistics"],
        "philosophy": ["philosophy", "philosopher", "ethics", "metaphysics", "logic",
                      "existential", "moral"],
        "entertainment": ["entertainment", "celebrity", "television", "show", "series",
                         "actor", "actress", "entertainment industry"]
    }
    
    # Score each domain
    domain_scores = defaultdict(int)
    
    # Check categories (highest weight)
    for domain, keywords in domains.items():
        for keyword in keywords:
            if keyword in categories_lower:
                domain_scores[domain] += 3
            if keyword in title_lower:
                domain_scores[domain] += 2
            # Check first 500 chars of text (lighter weight)
            if keyword in text_lower[:500]:
                domain_scores[domain] += 1
    
    # Get primary and secondary domains
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_domain = sorted_domains[0][0] if sorted_domains else "general"
    secondary_domains = [d[0] for d in sorted_domains[1:3] if d[1] > 0]
    
    return {
        "primary_domain": primary_domain,
        "secondary_domains": secondary_domains,
        "domain_confidence": sorted_domains[0][1] if sorted_domains else 0
    }

def enhance_metadata():
    """
    Enhance existing metadata with domain categorization and filters.
    """
    PROJECT_ROOT = Path(__file__).parent.parent
    INDEX_DIR = PROJECT_ROOT / "data" / "index"
    
    metadata_path = INDEX_DIR / "wikipedia.pkl"
    enhanced_path = INDEX_DIR / "wikipedia_enhanced.pkl"
    
    print("Loading existing metadata...")
    with open(metadata_path, "rb") as f:
        checkpoint_data = pickle.load(f)
        id_map = checkpoint_data['id_map']
        next_id = checkpoint_data['next_id']
        total_vectors = checkpoint_data.get('total_vectors', len(id_map))
    
    print(f"Found {len(id_map):,} documents")
    print("\nEnhancing metadata with domain categorization...")
    
    # Build domain index
    domain_index = defaultdict(list)
    enhanced_id_map = {}
    
    for idx, meta in id_map.items():
        title = meta.get('title', '')
        categories = meta.get('categories', '')
        text = meta.get('text', '')
        
        # Categorize
        domain_info = categorize_article(title, categories, text)
        
        # Enhance metadata
        enhanced_meta = {
            **meta,
            **domain_info
        }
        
        enhanced_id_map[idx] = enhanced_meta
        
        # Add to domain index
        domain_index[domain_info['primary_domain']].append(int(idx))
        for sec_domain in domain_info['secondary_domains']:
            domain_index[sec_domain].append(int(idx))
        
        if int(idx) % 1000 == 0:
            print(f"  Processed {idx} documents...")
    
    print("\n✅ Enhancement complete!")
    print("\nDomain distribution:")
    for domain, indices in sorted(domain_index.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {domain}: {len(indices):,} articles")
    
    # Save enhanced metadata
    enhanced_data = {
        'id_map': enhanced_id_map,
        'next_id': next_id,
        'total_vectors': total_vectors,
        'domain_index': dict(domain_index)
    }
    
    with open(enhanced_path, "wb") as f:
        pickle.dump(enhanced_data, f)
    
    print(f"\n✅ Enhanced metadata saved to: {enhanced_path}")
    print(f"   Size: {enhanced_path.stat().st_size / (1024**2):.2f} MB")

if __name__ == "__main__":
    enhance_metadata()