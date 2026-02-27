import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.document_processor import load_and_split_documents
from src.vector_store import create_vector_store, get_retriever
from src.rag_engine import build_rag_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🤖")
st.title("🤖 Enterprise RAG Chatbot")

# Initialize session state for memory and chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for document processing
with st.sidebar:
    st.header("Document Processing")
    st.info("Ensure PDFs are placed in the `data/` folder before initializing.")
    if st.button("Initialize/Update Knowledge Base"):
        with st.spinner("Processing documents..."):
            chunks = load_and_split_documents()
            vector_store = create_vector_store(chunks)
            retriever = get_retriever(vector_store)
            st.session_state.rag_chain = build_rag_chain(retriever)
            st.success("Knowledge Base Updated!")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if st.session_state.rag_chain is None:
        st.error("Please initialize the knowledge base first from the sidebar!")
    else:
        # Append and display user message
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get response from RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(
                    {"chat_history": st.session_state.chat_history, "input": user_query}
                )
                answer = response["answer"]
                st.markdown(answer)

        # Append AI response to history
        st.session_state.chat_history.append(AIMessage(content=answer))
