# indexing.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_index(documents, model_name="all-mpnet-base-v2"):
    """Creates a FAISS index for efficient similarity search."""
    model = SentenceTransformer(model_name)
    sentences = list(documents.values())  # or split into smaller chunks
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Convert embeddings to numpy array
    embeddings = embeddings.cpu().numpy()

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)

    return index, model