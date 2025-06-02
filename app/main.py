from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.services.document_service import process_document, save_vector_db, load_vector_db, get_processing_status_api
from app.services.query_service import stream_query
from app.services.log_service import log_event, get_logs
from dotenv import load_dotenv
import os
import shutil
import asyncio
import logging
import PyPDF2
from pathlib import Path

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix template directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def check_pdf_accessibility(filepath: str) -> bool:
    """Check if the PDF file is accessible and can be read."""
    try:
        with open(filepath, 'rb') as file:
            # Try to create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            # Try to read the first page
            if len(pdf_reader.pages) > 0:
                _ = pdf_reader.pages[0].extract_text()
            return True
    except Exception as e:
        logger.error(f"PDF accessibility check failed: {str(e)}")
        return False

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".html"]:
            raise HTTPException(status_code=400, detail="Only PDF and HTML files are supported.")
        filepath = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        file_size = 0
        with open(filepath, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)
                file_size += len(chunk)
        log_event(f"File uploaded: {file.filename} ({file_size} bytes)")
        # Start background processing and return status
        result = process_document(filepath)
        log_event(f"Processing started for: {file.filename}")
        return JSONResponse(content={
            "status": "processing",
            "message": "File uploaded. Processing in background.",
            "file_size": file_size,
            **result
        })
    except Exception as e:
        log_event(f"Upload failed: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing_status/{base_name}")
async def processing_status(base_name: str):
    return get_processing_status_api(base_name)

@app.post("/set_search_mode")
async def set_search_mode(request: Request):
    data = await request.json()
    mode = data.get("mode")
    if mode not in ["semantic", "keyword", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'semantic', 'keyword', or 'hybrid'.")
    # Store mode in a global or session (not shown here)
    log_event(f"Search mode set to: {mode}")
    return {"status": f"Search mode set to {mode}"}

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/query")
async def query(request: Request):
    question = request.query_params.get("question")
    try:
        save_dir = Path("saved_dbs")
        dbs = list(save_dir.glob("*_vectordb"))
        if not dbs:
            raise HTTPException(status_code=409, detail="No document has been uploaded and processed yet. Please upload a document first.")
        latest_db = sorted(dbs, key=os.path.getmtime)[-1].name.replace("_vectordb", "")
        status = get_processing_status_api(latest_db)
        if status.get("status") == "error":
            raise HTTPException(status_code=409, detail=f"Document processing failed: {status.get('error')}")
        if status.get("status") == "done":
            # If already processed, load and use the vector DB immediately
            try:
                vectordb, chunks, retriever = load_vector_db(latest_db)
            except Exception as e:
                log_event(f"Vector DB load failed for {latest_db}: {str(e)}")
                raise HTTPException(status_code=409, detail=f"Failed to load processed document. Please try re-uploading. Error: {str(e)}")
            try:
                log_event(f"Question received: {question}")
                return stream_query(question, retriever)
            except Exception as e:
                log_event(f"Query execution failed: {str(e)}")
                raise HTTPException(status_code=409, detail=f"Failed to generate answer. Please try again or re-upload your document. Error: {str(e)}")
        else:
            # If not done, return appropriate status
            raise HTTPException(status_code=409, detail="Document is still processing. Please wait until processing is complete.")
    except HTTPException as e:
        log_event(f"Query failed: {str(e.detail)}")
        raise e
    except Exception as e:
        log_event(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_log_events():
    return {"logs": get_logs()}

@app.get("/saved_dbs")
async def list_saved_dbs():
    """List all saved vector databases"""
    save_dir = Path("saved_dbs")
    if not save_dir.exists():
        return {"databases": []}
    
    # Get all unique base names (without _vectordb or _chunks suffix)
    dbs = set()
    for file in save_dir.glob("*_vectordb"):
        dbs.add(file.name.replace("_vectordb", ""))
    
    return {"databases": list(dbs)}

@app.get("/load_db/{base_name}")
async def load_db(base_name: str):
    """Load a saved vector database (metadata only, not the actual DB objects)"""
    try:
        vectordb, chunks, retriever = load_vector_db(base_name)
        # Only return serializable metadata
        return {
            "base_name": base_name,
            "num_chunks": len(chunks) if chunks else 0,
            "status": "loaded",
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.rag_pipeline import process_pdf, query_rag, set_mode
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates directory
templates = Jinja2Templates(directory="app/templates")

# Add this route to serve index.html
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    filepath = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    status = process_pdf(filepath)
    return {"status": status}

@app.post("/set_search_mode")
async def set_search_mode(mode: str, request: Request):
    return set_mode(mode)

@app.get("/query")
async def query(request: Request):
    question = request.query_params.get("question")
    return query_rag(question)"""