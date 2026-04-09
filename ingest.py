from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
import os

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

vectorstore = None

def ingest_pdf(pdf_path: str, filename: str) -> int:
    global vectorstore
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = filename
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    return len(chunks)

def get_vectorstore():
    return vectorstore

def get_indexed_files() -> list[str]:
    if vectorstore is None:
        return []
    return list({
        doc.metadata.get("source", "?")
        for doc in vectorstore.docstore._dict.values()
    })