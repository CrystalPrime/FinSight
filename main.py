from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile, os
from pathlib import Path

load_dotenv()

from graph import finance_graph, THREAD_CONFIG
from ingest import ingest_pdf, get_indexed_files
from langchain_core.messages import HumanMessage

app = FastAPI(title="Finance Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    result = finance_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=THREAD_CONFIG,
    )
    answer = result["messages"][-1].content
    return {"answer": answer}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    chunks = ingest_pdf(tmp_path, file.filename)
    os.unlink(tmp_path)
    return {"filename": file.filename, "chunks": chunks}

@app.get("/documents")
def documents():
    return {"files": get_indexed_files()}