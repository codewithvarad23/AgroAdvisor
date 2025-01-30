"""Embedding data file and faiss vector database connect"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunking import text_split, txt_data_load
from langchain.schema import Document

def embedding_text():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def faiss_store_db(docs, embeddings):
    store = FAISS.from_documents(docs, embeddings)
    store.save_local('faiss_db')

# Load and process data
data = txt_data_load('text.txt')  # Text-file-Loader only
split_text = text_split(data)
docs = [Document(page_content=text) for text in split_text]

# Generate embeddings and store in FAISS
embedding = embedding_text()
faiss_store_db(docs, embedding)