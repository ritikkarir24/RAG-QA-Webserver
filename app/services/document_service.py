import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
import pickle
from app.services.log_service import log_event
from concurrent.futures import ThreadPoolExecutor
import threading
import json

PROCESSING_STATUS_DIR = Path("processing_status")
PROCESSING_STATUS_DIR.mkdir(exist_ok=True)
_executor = ThreadPoolExecutor(max_workers=2)

# Helper to write status
def set_processing_status(base_name, status, error=None):
    status_file = PROCESSING_STATUS_DIR / f"{base_name}.json"
    with open(status_file, 'w') as f:
        json.dump({"status": status, "error": error}, f)

def get_processing_status(base_name):
    status_file = PROCESSING_STATUS_DIR / f"{base_name}.json"
    if not status_file.exists():
        return {"status": "not_found"}
    with open(status_file, 'r') as f:
        return json.load(f)

def _process_document_sync(filepath, embedding_model=None):
    try:
        ext = os.path.splitext(filepath)[1].lower()
        base_name = Path(filepath).stem
        set_processing_status(base_name, "processing")
        if ext == '.pdf':
            loader = PyMuPDFLoader(filepath)
        elif ext == '.html':
            loader = UnstructuredHTMLLoader(filepath)
        else:
            log_event(f"Unsupported file format: {ext}")
            set_processing_status(base_name, "error", f"Unsupported file format: {ext}")
            return None
        documents = loader.load()
        # Option 1: Larger chunk size, less overlap
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        # Option 2: Parallel embedding (if supported)
        if embedding_model is None:
            embedding_model = OllamaEmbeddings(model="deepseek-r1:14b")
        # If embedding_model supports batch, use it directly; else, parallelize
        try:
            embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
            vectordb = FAISS.from_embeddings(embeddings, chunks)
        except Exception:
            # Fallback: sequential
            vectordb = FAISS.from_documents(chunks, embedding_model)
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 4
        hybrid_retriever = EnsembleRetriever(retrievers=[vectordb.as_retriever(), bm25], weights=[0.7, 0.3])
        save_vector_db(vectordb, chunks, filepath)
        set_processing_status(base_name, "done")
        log_event(f"Processing complete for: {filepath}")
        return vectordb, chunks, hybrid_retriever
    except Exception as e:
        log_event(f"Processing failed for {filepath}: {str(e)}")
        set_processing_status(base_name, "error", str(e))
        return None

def process_document(filepath, embedding_model=None):
    # Option 3: Background processing
    base_name = Path(filepath).stem
    set_processing_status(base_name, "queued")
    _executor.submit(_process_document_sync, filepath, embedding_model)
    return {"status": "processing", "message": "File is being processed in the background.", "base_name": base_name}

def save_vector_db(vectordb, chunks, filepath):
    save_dir = Path("saved_dbs")
    save_dir.mkdir(exist_ok=True)
    base_name = Path(filepath).stem
    db_path = save_dir / f"{base_name}_vectordb"
    chunks_path = save_dir / f"{base_name}_chunks.pkl"
    vectordb.save_local(str(db_path))
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    return {"status": "success", "db_path": str(db_path), "chunks_path": str(chunks_path)}

def load_vector_db(base_name, embedding_model=None):
    save_dir = Path("saved_dbs")
    db_path = save_dir / f"{base_name}_vectordb"
    chunks_path = save_dir / f"{base_name}_chunks.pkl"
    if not db_path.exists() or not chunks_path.exists():
        raise ValueError(f"No saved database found for {base_name}")
    if embedding_model is None:
        embedding_model = OllamaEmbeddings(model="deepseek-r1:14b")
    vectordb = FAISS.load_local(str(db_path), embedding_model, allow_dangerous_deserialization=True)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    hybrid_retriever = EnsembleRetriever(retrievers=[vectordb.as_retriever(), bm25], weights=[0.7, 0.3])
    return vectordb, chunks, hybrid_retriever

def get_processing_status_api(base_name):
    return get_processing_status(base_name)
