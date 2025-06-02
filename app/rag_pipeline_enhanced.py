from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
import os
import tempfile
import fitz  # PyMuPDF
import mimetypes

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'html'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

VECTOR_DB = None
CHUNKS = []
RETRIEVER = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_and_metadata(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        loader = PyMuPDFLoader(filepath)
    elif ext == '.html':
        loader = UnstructuredHTMLLoader(filepath)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    enriched_docs = []

    for doc in documents:
        metadata = {
            'source': filepath,
            'type': ext,
            'length': len(doc.page_content),
        }
        enriched_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    return enriched_docs

def build_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4

    hybrid_retriever = EnsembleRetriever(retrievers=[vectordb.as_retriever(), bm25], weights=[0.7, 0.3])

    return vectordb, chunks, hybrid_retriever

@app.route('/upload', methods=['POST'])
def upload_file():
    global VECTOR_DB, CHUNKS, RETRIEVER

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        documents = extract_text_and_metadata(filepath)
        VECTOR_DB, CHUNKS, RETRIEVER = build_vector_db(documents)

        return jsonify({'message': 'File uploaded and processed successfully.'})
    else:
        return jsonify({'error': 'Unsupported file type.'}), 400

@app.route('/query', methods=['POST'])
def query():
    global RETRIEVER
    question = request.json.get('question')
    if not RETRIEVER:
        return jsonify({'error': 'No documents loaded yet.'}), 400

    # Remove OpenAI LLM usage
    # chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=RETRIEVER, return_source_documents=True)
    # Instead, raise an error or leave as a placeholder for Ollama/other LLM integration
    return jsonify({'error': 'No LLM configured. This pipeline is legacy and not used in production.'}), 400

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
