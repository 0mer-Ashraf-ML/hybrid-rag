# create_docs_json.py
import pickle
import json
from pathlib import Path

def create_docs_json():
    """
    Convert Wikipedia metadata (pickle) to docs.json format for BM25.
    """
    # Use the actual path where your files are
    INDEX_DIR = Path(r"G:\fyp relevant\work\New folder (2)\hybrid-rag\data\index")
    
    metadata_path = INDEX_DIR / "wikipedia.pkl"
    output_path = INDEX_DIR / "docs.json"
    
    print(f"Loading metadata from {metadata_path}...")
    
    # Check if file exists
    if not metadata_path.exists():
        print(f"❌ Error: File not found at {metadata_path}")
        print(f"\nPlease ensure wikipedia.pkl is in: {INDEX_DIR}")
        return
    
    with open(metadata_path, "rb") as f:
        checkpoint_data = pickle.load(f)
        id_map = checkpoint_data['id_map']
    
    print(f"Found {len(id_map)} documents")
    
    # Convert to docs.json format
    docs = []
    for idx, metadata in id_map.items():
        # Combine text and summary for better retrieval
        text = metadata.get('text', '')
        summary = metadata.get('summary', '')
        combined_text = f"{summary} {text}".strip()
        
        doc = {
            "id": str(idx),
            "source": metadata.get('title', 'Unknown'),
            "url": metadata.get('url', ''),
            "categories": metadata.get('categories', ''),
            "text": combined_text,
            "title": metadata.get('title', ''),
            "text_length": len(text),
            "summary_length": len(summary),
        }
        docs.append(doc)
    
    # Save to JSON
    print(f"Saving {len(docs)} documents to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created docs.json with {len(docs)} documents")
    print(f"   File location: {output_path}")
    print(f"   Total size: {output_path.stat().st_size / (1024**2):.2f} MB")

if __name__ == "__main__":
    create_docs_json()