# summarization.py
from transformers import pipeline

def generate_summary(context, model_name="facebook/bart-large-cnn", max_length=150, min_length=50):
    """Generates a summary using a pre-trained summarization model."""
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(context, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary