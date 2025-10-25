import pickle
import bz2
import lzma
import zlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def decompress_text(compressed_data, method):
    if not compressed_data:
        return ""
    
    if not isinstance(compressed_data, bytes):
        return compressed_data
    
    try:
        if method == 1:  
            return zlib.decompress(compressed_data).decode('utf-8')
        elif method == 2:
            return lzma.decompress(compressed_data).decode('utf-8')
        else:
            return compressed_data.decode('utf-8') if isinstance(compressed_data, bytes) else compressed_data
    except Exception as e:
        print(f"Warning: Failed to decompress: {e}")
        return ""

def categorize_article(title: str, text: str, summary: str) -> dict:
    
    text_lower = text.lower()[:1000]  # Use first 1000 chars for efficiency
    summary_lower = summary.lower()
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
    
    domain_scores = defaultdict(int)
    
    for domain, keywords in domains.items():
        for keyword in keywords:
            if keyword in title_lower:
                domain_scores[domain] += 3
            if keyword in summary_lower:
                domain_scores[domain] += 2
            if keyword in text_lower:
                domain_scores[domain] += 1
    
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_domain = sorted_domains[0][0] if sorted_domains else "general"
    secondary_domains = [d[0] for d in sorted_domains[1:3] if d[1] > 0]
    
    return {
        "primary_domain": primary_domain,
        "secondary_domains": secondary_domains,
        "domain_confidence": sorted_domains[0][1] if sorted_domains else 0
    }

def enhance_ultra_metadata():
    
    PROJECT_ROOT = Path(__file__).parent.parent
    ULTRA_DIR = Path("/Users/omarashraf/Downloads/hybrid-rag/ultra_compressed_208k_p6")
    
    input_path = ULTRA_DIR / "metadata_ultra_compressed.pkl"
    output_path = ULTRA_DIR / "metadata_ultra_enhanced.pkl"
    
    if not input_path.exists():
        print(f"❌ Error: Ultra-compressed metadata not found at {input_path}")
        print(f"\nPlease run: python scripts/compress_to_1_1gb.py")
        return
    
    print("="*70)
    print("🔧 ENHANCING ULTRA-COMPRESSED METADATA WITH DOMAINS")
    print("="*70)
    
    print(f"\nLoading ultra-compressed metadata from {input_path}...")
    print(f"   Size: {input_path.stat().st_size / (1024**2):.2f} MB")
    
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    docs = data['d']  
    title_dict = data.get('td', [])  
    url_dict = data.get('ud', [])    
    
    print(f"   Documents: {len(docs):,}")
    print(f"   Title dictionary: {len(title_dict):,} unique titles")
    print(f"   URL dictionary: {len(url_dict):,} unique URLs")
    
    print("\n🔍 Analyzing and categorizing documents...")
    
    
    domain_index = defaultdict(list)
    enhanced_docs = {}
    
    
    processed = 0
    for idx, doc in docs.items():
    
        title_idx = doc.get('ti', -1)
        if title_idx >= 0 and title_idx < len(title_dict):
            title = title_dict[title_idx]
        else:
            title = doc.get('t', '')  
        
        # Decompress text
        text = doc.get('txt', '')
        text_compressed = doc.get('_tc', False)
        if text_compressed:
            text = decompress_text(text, text_compressed)
        
        summary = doc.get('s', '')
        summary_compressed = doc.get('_sc', False)
        if summary_compressed:
            summary = decompress_text(summary, summary_compressed)
        
        domain_info = categorize_article(title, text, summary)
        
        enhanced_doc = {
            **doc,  
            'd': domain_info['primary_domain'],  
            'sd': domain_info['secondary_domains'][:2] if domain_info['secondary_domains'] else []  
        }
        
        enhanced_docs[idx] = enhanced_doc
        
        primary = domain_info['primary_domain']
        domain_index[primary].append(int(idx))
        
        for sec_domain in domain_info['secondary_domains'][:2]:
            domain_index[sec_domain].append(int(idx))
        
        processed += 1
        if processed % 10000 == 0:
            print(f"   Processed {processed:,} documents...")
    
    print(f"\n✅ Categorization complete! Processed {processed:,} documents")
    
    print("\n📦 Compressing domain index...")
    compressed_domain_index = {
        domain: np.array(sorted(set(indices)), dtype=np.int32).tobytes()
        for domain, indices in domain_index.items()
    }
    
    # Domain distribution
    print("\n📊 Domain Distribution:")
    sorted_domains = sorted(domain_index.items(), key=lambda x: len(x[1]), reverse=True)
    for domain, indices in sorted_domains[:15]:  # Top 15
        print(f"   {domain:20s}: {len(set(indices)):,} documents")
    
    if len(sorted_domains) > 15:
        print(f"   ... and {len(sorted_domains) - 15} more domains")
    
    enhanced_data = {
        'd': enhanced_docs,  
        'td': title_dict,    
        'ud': url_dict,      
        'domains': compressed_domain_index,  
        'm': {  
            'n': len(enhanced_docs),
            'total_domains': len(domain_index),
            'enhanced': True,
            'version': '2.0'
        }
    }
    
    print(f"\n💾 Saving enhanced metadata...")
    with bz2.open(output_path, "wb") as f:
        pickle.dump(enhanced_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    enhanced_size = output_path.stat().st_size / (1024**2)
    original_size = input_path.stat().st_size / (1024**2)
    
    print(f"\n✅ Enhanced metadata saved!")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Enhanced size: {enhanced_size:.2f} MB")
    print(f"   Size increase: {enhanced_size - original_size:.2f} MB ({(enhanced_size/original_size - 1)*100:.1f}%)")
    print(f"   Location: {output_path}")
    
    print(f"\n🔍 Verifying enhanced metadata...")
    try:
        with bz2.open(output_path, "rb") as f:
            verify_data = pickle.load(f)
        
        print(f"   ✅ File loads successfully")
        print(f"   ✅ Documents: {len(verify_data['d']):,}")
        print(f"   ✅ Domains: {len(verify_data['domains'])}")
        
        test_domain = list(verify_data['domains'].keys())[0]
        test_indices = np.frombuffer(verify_data['domains'][test_domain], dtype=np.int32)
        print(f"   ✅ Domain index decompression works: {len(test_indices):,} docs in '{test_domain}'")
        
        test_idx = list(verify_data['d'].keys())[0]
        test_doc = verify_data['d'][test_idx]
        print(f"   ✅ Document access works: doc {test_idx} has domain '{test_doc.get('d', 'unknown')}'")
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return
    
    print("\n" + "="*70)
    print("🎉 ENHANCEMENT COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Update ultra_compressed_retriever.py to use 'metadata_ultra_enhanced.pkl'")
    print(f"2. Test retrieval with domain filtering enabled")
    print(f"3. Optionally backup/remove old metadata_ultra_compressed.pkl")
    
    print(f"\n💡 Usage in code:")
    print(f"   # In src/utils/config.py, update:")
    print(f"   WIKI_ULTRA_METADATA = WIKI_ULTRA_INDEX_DIR / 'metadata_ultra_enhanced.pkl'")

if __name__ == "__main__":
    enhance_ultra_metadata()