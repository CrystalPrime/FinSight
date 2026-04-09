from pathlib import Path

INDEXES_DIR = Path("indexes")
EMBEDDING_MODEL = r"C:\Users\batuhan.aydin\Desktop\Langchain\Free Embedding Models\all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2

INDEXES_DIR.mkdir(exist_ok=True)