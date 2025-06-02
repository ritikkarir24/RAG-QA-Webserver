from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
import os
import pickle
from pathlib import Path

VECTOR_DB = None
CHUNKS = []
SEARCH_MODE = "hybrid"
RETRIEVER = None

def process_pdf(filepath):
    global VECTOR_DB, CHUNKS, RETRIEVER
    try:
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        CHUNKS = splitter.split_documents(documents)
        
        # Updated to use deepseek model
        embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
        VECTOR_DB = FAISS.from_documents(CHUNKS, embeddings)

        bm25 = BM25Retriever.from_documents(CHUNKS)
        bm25.k = 4
        RETRIEVER = EnsembleRetriever(
            retrievers=[VECTOR_DB.as_retriever(), bm25], 
            weights=[0.7, 0.3]
        )
        
        # Save the database after creation
        save_result = save_vector_db(filepath)
        return {
            "status": "Uploaded and processed successfully",
            **save_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def query_rag(question: str):
    global VECTOR_DB, SEARCH_MODE, RETRIEVER, CHUNKS
    
    if not question:
        raise ValueError("Question parameter is required")
    
    if VECTOR_DB is None:
        raise ValueError("No document has been processed yet. Please upload a PDF first.")

    try:
        if SEARCH_MODE == "semantic":
            retriever = VECTOR_DB.as_retriever()
        elif SEARCH_MODE == "keyword":
            retriever = BM25Retriever.from_documents(CHUNKS)
        else:
            retriever = RETRIEVER  # hybrid

        model = ChatOllama(model="deepseek-r1:14b")
        template = """
        You are an expert in understanding and answering questions based only on the content provided in the uploaded PDF.

        Context:
        {context}

        Question:
        {question}

        Answer using only the information in the context. If the answer is not present, say you don't know.
        """
        prompt = ChatPromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        answer = qa_chain.run(prompt)
        if not answer:
            yield "I don't have enough information to answer that question."
            return

        # Split answer into words and yield them
        for word in answer.split():
            yield word + " "

    except Exception as e:
        raise ValueError(f"Error processing query: {str(e)}")

def set_mode(mode: str):
    global SEARCH_MODE
    if mode not in ["semantic", "keyword", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Choose 'semantic', 'keyword', or 'hybrid'."
        )
    SEARCH_MODE = mode
    return {"status": f"Search mode set to {SEARCH_MODE}"}

def save_vector_db(filepath: str):
    """Save the vector database and chunks to disk"""
    global VECTOR_DB, CHUNKS
    if VECTOR_DB is None:
        raise ValueError("No vector database exists to save")
    
    # Create a directory for saved databases if it doesn't exist
    save_dir = Path("saved_dbs")
    save_dir.mkdir(exist_ok=True)
    
    # Generate a unique name based on the PDF filename
    base_name = Path(filepath).stem
    db_path = save_dir / f"{base_name}_vectordb"
    chunks_path = save_dir / f"{base_name}_chunks.pkl"
    
    # Save FAISS index
    VECTOR_DB.save_local(str(db_path))
    
    # Save chunks
    with open(chunks_path, 'wb') as f:
        pickle.dump(CHUNKS, f)
        
    return {"status": "success", "db_path": str(db_path), "chunks_path": str(chunks_path)}

def load_vector_db(base_name: str):
    """Load a saved vector database and chunks"""
    global VECTOR_DB, CHUNKS, RETRIEVER
    
    save_dir = Path("saved_dbs")
    db_path = save_dir / f"{base_name}_vectordb"
    chunks_path = save_dir / f"{base_name}_chunks.pkl"
    
    if not db_path.exists() or not chunks_path.exists():
        raise ValueError(f"No saved database found for {base_name}")
    
    # Load FAISS index
    embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
    VECTOR_DB = FAISS.load_local(str(db_path), embeddings)
    
    # Load chunks
    with open(chunks_path, 'rb') as f:
        CHUNKS = pickle.load(f)
    
    # Recreate retriever
    bm25 = BM25Retriever.from_documents(CHUNKS)
    bm25.k = 4
    RETRIEVER = EnsembleRetriever(
        retrievers=[VECTOR_DB.as_retriever(), bm25],
        weights=[0.7, 0.3]
    )
    
    return {"status": "success", "message": f"Loaded vector database for {base_name}"}


"""from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from fastapi.responses import StreamingResponse
import os

VECTOR_DB = None
CHUNKS = []
SEARCH_MODE = "semantic"


def process_pdf(filepath):
    global VECTOR_DB, CHUNKS
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    CHUNKS = splitter.split_documents(documents)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    VECTOR_DB = FAISS.from_documents(CHUNKS, embeddings)
    return "Uploaded and processed successfully."


def set_mode(mode: str):
    global SEARCH_MODE
    if mode not in ["semantic", "keyword"]:
        return {"error": "Invalid mode. Choose 'semantic' or 'keyword'."}
    SEARCH_MODE = mode
    return {"status": f"Search mode set to {SEARCH_MODE}"}


def query_rag(question):
    global VECTOR_DB, SEARCH_MODE
    if not question or VECTOR_DB is None:
        return {"error": "Missing question or no document uploaded."}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

    if SEARCH_MODE == "semantic":
        retriever = VECTOR_DB.as_retriever()
    else:
        retriever = VECTOR_DB.as_retriever(search_type="mmr")

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key, streaming=True),
        retriever=retriever,
        return_source_documents=False
    )

    async def event_generator():
        answer = qa_chain.run(question)
        for word in answer.split():
            yield f"data: {word} \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def verify_access_key(request, expected_key):
    key = request.headers.get("X-Access-Key")
    return key == expected_key"""
