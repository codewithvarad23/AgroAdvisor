"""Chatbot code"""
from rag_file import retrieve_data
import streamlit as st
from embedding import embedding_text
from langchain_community.vectorstores import FAISS

def initialize_session_state():
    """Initialize chat history and other session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm AgroAdvisor AI. How can I help you today?"}
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def load_vector_store():
    """Load FAISS vector store with error handling"""
    try:
        embeddings = embedding_text()
        return FAISS.load_local(
            'faiss_db',
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def handle_user_query(user_query, vector_store):
    """Process user query and return response"""
    try:
        with st.spinner("Analyzing your query..."):
            # Directly get response from retrieve_data
            response = retrieve_data(user_query, vector_store)
            return response
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def chat_interface():
    """Main chat interface"""
    st.title("AgroAdvisor AI Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_query := st.chat_input("Type your agricultural question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process query and get response
        vector_store = st.session_state.vector_store or load_vector_store()
        if vector_store:
            response = handle_user_query(user_query, vector_store)
        else:
            response = "Failed to load knowledge base. Please try again later."

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def api_page():
    """API documentation page"""
    st.title("API Documentation")
    st.markdown("""
    ## AgroAdvisor API Documentation
    ### Endpoints
    - `/predict`: POST endpoint for crop predictions
    - `/weather`: GET endpoint for weather data
    """)

def main():
    initialize_session_state()

    st.sidebar.title("AgroAdvisor")
    page = st.sidebar.selectbox("Navigation", ["Chat", "API"])

    # Load vector store once and cache in session state
    if not st.session_state.vector_store:
        st.session_state.vector_store = load_vector_store()

    if page == "Chat":
        chat_interface()
    elif page == "API":
        api_page()

if __name__ == "__main__":
    main()