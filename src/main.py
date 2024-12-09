# main.py
from data_processing import load_documents, preprocess_document
from indexing import create_index
from retrieval import retrieve_relevant_documents
from summarization import generate_summary
import time

if __name__ == "__main__":
    # 1. Load and preprocess documents
    documents = load_documents("data/sample_documents")
    for doc_name, doc_text in documents.items():
        documents[doc_name] = preprocess_document(doc_text)

    # 2. Create an index
    index, model = create_index(documents)

    # 3. Get a query from the user
    query = input("Enter your query to search the documents: ")

    # Measure retrieval time
    start_time = time.time()

    # 4. Retrieve relevant documents
    relevant_docs, relevant_texts, distances = retrieve_relevant_documents(query, index, model, documents)

    # Measure retrieval time
    end_time = time.time()
    retrieval_time = end_time - start_time

    # 5. Combine the text of the retrieved documents into a single context
    context = " ".join(relevant_texts)

    # Measure summarization time
    start_time = time.time()

    # 6. Generate a summary
    summary = generate_summary(context)

    # Measure summarization time
    end_time = time.time()
    summarization_time = end_time - start_time

    # 7. Print the results
    print("\nRelevant documents:", relevant_docs)
    print("\nGenerated summary:", summary)
    print(f"\nRetrieval time: {retrieval_time:.4f} seconds")
    print(f"Summarization time: {summarization_time:.4f} seconds")