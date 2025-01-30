from langchain_community.vectorstores import FAISS
from embedding import faiss_store_db, embedding_text, embedding
from langchain_community.llms import Ollama
from langchain.chains import retrieval_qa

# vector db load
def faiss_db():
    store_db=FAISS.load_local(embedding.embedding)