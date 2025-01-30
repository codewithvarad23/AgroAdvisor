"""Embedding data file and faiss vector database connect"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunking import text_split, txt_data_load
from langchain.schema import Document


def embedding_text():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = model.encode(text_data)
    # print("Embeddings:\n", embeddings)
    # return embeddings


def faiss_store_db(docs,embeddings):
    store=FAISS.from_documents(docs,embeddings)
    store.save_local('faiss_db')


data = txt_data_load('text.txt') # Text-file-Loader only
split_text = text_split(data)
docs = [Document(page_content=text) for text in split_text]

embedding=embedding_text()

faiss_store_db(docs, embedding)

# RAG (Retrieve and Generate) code

# def store_db():
#     try:
#         faiss_vector = FAISS.load_local('faiss_db', embedding, allow_dangerous_deserialization=True)
#         retrieve = faiss_vector.as_retriever(search_type='similarity', search_kwargs={'k': 3})
#     except Exception as e:
#         print(f"Error loading FAISS database: {e}")
#         return None
# retriever = store_db()
# if retriever:
#     results = retriever.get_relevant_documents("your search query here")
#     print(results)
