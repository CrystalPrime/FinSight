from langchain.tools import tool
from ingest import get_vectorstore
from config import TOP_K

@tool(response_format="content_and_artifact")
def search_financial_documents(query: str):
    """Search uploaded financial documents, reports, and PDFs for relevant information."""
    db = get_vectorstore()
    if db is None:
        return "No documents uploaded.", []
    docs = db.similarity_search(query, k=TOP_K)
    serialized = "\n\n".join(
        f"[{doc.metadata.get('source','?')} — p.{int(doc.metadata.get('page',0))+1}]\n{doc.page_content}"
        for doc in docs
    )
    return serialized, docs