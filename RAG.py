import os
import numpy as np
import PyPDF2
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

nltk.download('punkt')

# Load Open-Source Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    """Generate embeddings using SentenceTransformer."""
    return embedding_model.encode(texts, convert_to_tensor=True).cpu().numpy()

# Load Financial Data from PDF with Sentence Tokenization
def load_financial_data(filepath, chunk_size=1):
    texts = []
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                sentences = sent_tokenize(text)
                texts.extend([" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)])
    return texts

# Implement Guardrail (Filtering Non-Financial Queries)
def filter_query(query):
    financial_keywords = [
        "revenue", "net income", "gross profit", "operating profit", "earnings", 
        "cash flow", "assets", "liabilities", "equity", "balance sheet", "income statement", 
        "financial position", "operating expenses", "depreciation", "amortization", "net earnings",
        "stock price", "market capitalization", "shareholder equity", "earnings per share", "dividend",
        "stock buyback", "trading symbol", "New York Stock Exchange", "risk factors", 
        "regulatory compliance", "SEC filings", "Sarbanes-Oxley Act", "audit report", "debt maturity",
        "loan covenants", "leverage ratio", "credit facility", "advertising revenue", "digital revenue",
        "podcast revenue", "sponsorship revenue", "event revenue", "network revenue", "operating margin",
        "advertising expenses", "marketing budget", "loss"
    ]
    
    return any(word in query.lower() for word in financial_keywords)

# Preprocessing (Tokenization for BM25)
def preprocess(texts):
    return [text.lower().split() for text in texts]

# Multi-Stage Retrieval: BM25 + Embeddings + Re-ranking
def multi_stage_retrieval(query, bm25_corpus, bm25, embeddings, original_texts):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-10:][::-1]
    query_embedding = embed_text([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_embedding_indices = np.argsort(similarities)[-5:][::-1]
    retrieved_indices = list(set(top_bm25_indices) | set(top_embedding_indices))
    ranked_results = sorted(retrieved_indices, key=lambda i: (bm25_scores[i] + similarities[i]), reverse=True)
    return [original_texts[i] for i in ranked_results[:3]]

# Confidence Calculation with Weighted Scores
def calculate_confidence(query, bm25, bm25_corpus, embeddings):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = embed_text([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_bm25_scores = np.sort(bm25_scores)[-5:]
    top_embedding_scores = np.sort(similarities)[-5:]
    if np.max(bm25_scores) > 0:
        top_bm25_scores /= np.max(bm25_scores)
    if np.max(similarities) > 0:
        top_embedding_scores /= np.max(similarities)
    confidence_score = (0.7 * np.mean(top_bm25_scores)) + (0.3 * np.mean(top_embedding_scores))
    if confidence_score > 0.7:
        return "High Confidence"
    elif confidence_score > 0.4:
        return "Low Confidence"
    else:
        return "Irrelevant"

# Streamlit UI
def main():
    st.set_page_config(page_title="Financial QA System", layout="wide")
    st.title("📊 Financial Document Q&A")
    st.markdown("A professional AI-powered financial question-answering system for investors.")
    
    uploaded_file = st.file_uploader("Upload a financial statement (PDF)", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            texts = load_financial_data(uploaded_file, chunk_size=1)
            tokenized_corpus = preprocess(texts)
            bm25 = BM25Okapi(tokenized_corpus)
            embeddings = embed_text(texts)
            st.success("Document processed successfully!")
    
        query = st.text_input("Enter your financial question:")
        if st.button("Get Answer") and query:
            if filter_query(query):
                confidence = calculate_confidence(query, bm25, tokenized_corpus, embeddings)
                retrieved_texts = multi_stage_retrieval(query, tokenized_corpus, bm25, embeddings, texts)
                response = "\n\n".join(retrieved_texts)
                st.markdown(f"### Confidence: {confidence}")
                st.info(response)
            else:
                st.error("Invalid query. Please ask about financial topics.")

if __name__ == "__main__":
    main()
