from langchain_community.vectorstores import FAISS
from embedding import faiss_store_db, embedding_text
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def store_db():
    try:
        faiss_vector = FAISS.load_local('faiss_db', embedding_text(), allow_dangerous_deserialization=True)
        retrieve = faiss_vector.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        return retrieve
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None

retriever = store_db()

if retriever:
    llm = Ollama(model="tinyllama")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = "History of Mahabaleshwar strawberry"
    result = qa_chain.invoke(query)

    print("Answer from Q&A:", result)
