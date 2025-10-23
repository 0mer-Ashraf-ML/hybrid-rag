import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

def load_or_download_model(model_name="BAAI/bge-base-en", local_dir="G:/huggingface_models/bge-base-en"):
    """
    Load SentenceTransformer model with offline caching.
    If not found locally, it will download and store permanently.
    """

    # Step 1: Check if the model already exists locally
    if os.path.exists(local_dir):
        print(f"✅ Loading model from local cache: {local_dir}")
        model = SentenceTransformer(local_dir)
    else:
        print(f"⬇️ Model not found locally. Downloading {model_name} ...")
        os.makedirs(local_dir, exist_ok=True)

        # Step 2: Download full repository and save to local_dir
        snapshot_download(repo_id=model_name, local_dir=local_dir, resume_download=True)

        print(f"✅ Download completed. Model saved at: {local_dir}")
        model = SentenceTransformer(local_dir)

    return model


# ✅ Example usage
if __name__ == "__main__":
    model = load_or_download_model()
    embedding = model.encode("This is an offline-friendly embedding example.")
    print("🧠 Embedding vector size:", len(embedding))
