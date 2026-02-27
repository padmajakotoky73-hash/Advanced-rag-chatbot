# 🤖 Advanced RAG Chatbot

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.3.0+-green)
![Google Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini-blue)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-lightgrey)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 Overview

Advanced RAG Chatbot is a production-style Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and interact with them through a conversational AI interface. The system ensures responses are grounded strictly in user-provided documents by combining semantic search, local vector indexing, and context-aware generation using Google Gemini, significantly reducing hallucinations and improving reliability.

---

## 🎯 Key Highlights

- End-to-end RAG pipeline implementation  
- LLM orchestration with LangChain  
- Local vector database integration using FAISS  
- Secure API key handling via environment variables  
- Modular backend architecture  
- Interactive frontend built with Streamlit  

---

## 🏗️ Architecture

User Query → Semantic Retrieval (FAISS) → Relevant Context Chunks → Google Gemini (LLM) → Grounded Response  

### RAG Pipeline

1. Load – Extract text from PDFs using `pypdf`  
2. Split – Token-based chunking using `tiktoken`  
3. Embed – Convert chunks into embeddings using Google GenAI  
4. Store – Save embeddings in a FAISS vector index  
5. Retrieve – Perform similarity search for relevant context  
6. Generate – Send retrieved context + query to Gemini  
7. Respond – Display final answer via Streamlit  

---

## 🛠️ Tech Stack

| Layer | Technology |
|--------|------------|
| Language | Python 3.9+ |
| LLM Orchestration | LangChain |
| LLM & Embeddings | Google Gemini (`langchain-google-genai`) |
| Vector Database | FAISS (CPU) |
| Tokenization | tiktoken |
| Document Parsing | PyPDF |
| Frontend | Streamlit |

---

## 📂 Project Structure

Advanced-rag-chatbot/
│
├── data/                      # Uploaded PDF files  
├── src/  
│   ├── __init__.py  
│   ├── document_processor.py  
│   ├── rag_engine.py  
│   └── vector_store.py  
│
├── app.py                     # Streamlit entry point  
├── requirements.txt  
├── .env.example  
└── README.md  

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher  
- Google Gemini API Key: https://aistudio.google.com/

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/advanced-rag-chatbot.git
cd advanced-rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key_here
```

⚠️ Never commit your `.env` file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Open in browser:
http://localhost:8501

---

## 🚀 Features

- Multi-PDF document ingestion  
- Semantic similarity search  
- Fully local vector storage (FAISS)  
- Conversational memory  
- Clean modular backend architecture  
- Responsive Streamlit UI  
- Secure API key management  

---

## 📊 Production Considerations

For production deployment, consider implementing:

- FAISS GPU acceleration  
- Persistent vector storage  
- Docker containerization  
- Cloud deployment (AWS / GCP / Render)  
- Authentication and user management  
- Rate limiting  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue to discuss proposed improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Padmaja Kotoky  
GitHub: https://github.com/padmajakotoky73-hash