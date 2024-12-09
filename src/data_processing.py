# data_processing.py
import os

def load_documents(directory):
    """Loads documents from a specified directory."""
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents

def preprocess_document(text):
    """Performs basic text preprocessing (e.g., lowercasing, removing punctuation)."""
    text = text.lower()
    # Add more preprocessing steps as needed
    return text