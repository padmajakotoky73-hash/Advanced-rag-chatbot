from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    """Embeds document chunks and creates a local FAISS vector store."""
    # FIXED: Updated to Google's current supported embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(vector_store):
    """Returns a retriever interface from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": 3})
