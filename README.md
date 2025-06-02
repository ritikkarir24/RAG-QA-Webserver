# RAG QA System

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


## 🧪 API Endpoints

| Endpoint                | Method | Description                                             |
|-------------------------|--------|---------------------------------------------------------|
| `/upload`               | POST   | Upload PDF/HTML                                         |
| `/query`                | GET    | Ask question (via query param, SSE streaming)           |
| `/query`                | POST   | Ask question (JSON, legacy/alt)                         |
| `/set_search_mode`      | POST   | Set retrieval mode (semantic, keyword, hybrid)          |
| `/processing_status/{base_name}` | GET | Get processing status for uploaded file             |
| `/saved_dbs`            | GET    | List all saved vector databases                         |
| `/load_db/{db_name}`    | GET    | Load a specific saved vector database                   |
| `/logs`                 | GET    | Fetch server logs for UI panel                          |


## ⚠️ Disclaimer
This repo is a conceptual demo for technical evaluation only. To use this in production or extend further, please reach out.

## 📄 License

This project is licensed under a custom MIT License (viewing only, no execution or deployment allowed without written consent). See the [LICENSE](./LICENSE) file for full terms.

Permission is granted to view and review the code for evaluation purposes only. Running, modifying, distributing, or deploying this code in any environment (local or production) is strictly prohibited without prior written consent from the copyright holder.

For licensing or usage inquiries, contact: rkarir@gmail.com
