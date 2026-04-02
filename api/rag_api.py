import uvicorn
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Always resolve src/ relative to THIS file, not the current working directory
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from rag_engine import setup_rag_chain
except ImportError:
    print("FATAL: Failed to import RAG Engine. Are dependencies installed?")

app = FastAPI(title="AI Copilot: Cyber Threat RAG System", version="3.0.0")

class ChatRequest(BaseModel):
    query: str

qa_chain = None

@app.on_event("startup")
def bootstrap_rag():
    global qa_chain
    logging.info("Booting LangChain FAISS Retriever and HuggingFace Local LLM...")
    try:
        qa_chain = setup_rag_chain()
    except Exception as e:
        logging.error(f"RAG Boot Failure: {e}")

@app.post("/chat")
def chat_with_copilot(req: ChatRequest):
    """Answers queries strictly based on the FAISS CVE database context."""
    if not qa_chain:
        return {"error": "Local LLM is Offline (Requires 4GB+ System/GPU RAM). Check server logs.", "response": "Sorry, local LLM failed to load due to hardware constraints."}
        
    try:
        logging.info(f"RAG Query Received: {req.query}")
        result = qa_chain.invoke(req.query)
        return {
            "query": req.query,
            "response": result['result'].strip(),
            "sources": "Local FAISS Cached Cyber Knowledge Index"
        }
    except Exception as e:
        return {"error": str(e), "response": "Failed to generate text."}

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8003, reload=True)
