from langchain.chains import RetrievalQA
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from langchain_ollama import OllamaLLM

def stream_query(question, retriever, llm=None):
    if not question:
        raise HTTPException(status_code=400, detail="Question parameter is required")
    if retriever is None:
        raise HTTPException(status_code=400, detail="No retriever available. Upload a document first.")
    # Always use OllamaLLM if not provided
    if llm is None:
        llm = OllamaLLM(model="deepseek-r1:14b", base_url="http://localhost:11434")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    def event_generator():
        answer = qa_chain.run(question)
        for word in answer.split():
            yield f"data: {word} \n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
