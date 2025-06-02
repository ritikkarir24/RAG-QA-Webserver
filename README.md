# RAG PDF/HTML QA System

A Retrieval-Augmented Generation (RAG) web server for question answering over your own PDF documents. Built with FastAPI, LangChain, and Ollama, this project enables users to upload documents, process them into vector databases, and interactively query their content using advanced LLMs.


## High-Level Architecture

```
User (Browser)
   |
   |  (HTTP/REST, SSE)
   v
FastAPI Backend (main.py)
   |
   |-- Document Service (background processing, chunking, embedding, vector DB)
   |-- Query Service (retrieval, streaming answers)
   |-- Log Service (in-memory logs for UI)
   |
   |-- File System (temp/, saved_dbs/, processing_status/)
   |
   |-- LangChain + Ollama (deepseek-r1:14b for embeddings & LLM)
   v
Vector DB (FAISS), BM25, Hybrid Retriever
```

---

## 🚀 Features
- **Document Upload**: Supports PDF and HTML files for ingestion.
- **Automated Chunking & Embedding**: Documents are split and embedded using the DeepSeek LLM via Ollama.
- **Hybrid Retrieval**: Combines semantic (vector) and keyword (BM25) search for robust QA.
- **Streaming Answers**: Real-time, word-by-word answer streaming to the UI.
- **Multiple Search Modes**: Switch between hybrid, semantic, and keyword retrieval.
- **Persistent Vector DBs**: Save and reload processed document databases.
- **Server Logs**: Live server log panel for transparency and debugging.


## 📝 Usage
- **Upload** a PDF or HTML file using the web interface.
- Wait for processing to complete (progress and status are shown).
- **Ask questions** about the document content in natural language.
- Switch retrieval modes as needed for best results.
- View logs and manage saved databases from the UI.

## 🧩 Tech Stack
- **Backend**: FastAPI, LangChain, FAISS, BM25, Ollama, PyMuPDF
- **Frontend**: Simple HTML/CSS/JS (Jinja2 templates)
- **LLM**: DeepSeek via Ollama (local inference)

## ⚠️ Limitations & Notes
- **UI/UX**: The user interface is intentionally minimal and functional. Due to time constraints, the focus was on backend robustness and retrieval quality rather than UI polish. PRs for UI improvements are welcome!
- **Model Requirements**: Requires Ollama and the DeepSeek model running locally.
- **File Types**: Only PDF supported.

