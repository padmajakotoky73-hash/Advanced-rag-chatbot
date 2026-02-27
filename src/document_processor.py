from langchain_community.document_loaders import PyPDFDirectoryLoader

# FIXED: Updated import path for LangChain 0.3.0+
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(data_dir: str = "data/"):
    """Loads PDFs from the data directory and splits them into smaller chunks."""
    # Load documents
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
