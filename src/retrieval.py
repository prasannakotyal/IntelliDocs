# retrieval.py
import faiss
from sentence_transformers import util

def retrieve_relevant_documents(query, index, model, documents, top_k=3):
    """Retrieves the top_k most relevant documents for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Convert query embedding to numpy array and normalize
    query_embedding = query_embedding.cpu().numpy()
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    # FAISS search
    D, I = index.search(query_embedding.reshape(1, -1), top_k)

    # Retrieve document names and texts
    relevant_docs = [list(documents.keys())[i] for i in I[0]]
    relevant_texts = [list(documents.values())[i] for i in I[0]]

    return relevant_docs, relevant_texts, D[0]  # Return distances as well