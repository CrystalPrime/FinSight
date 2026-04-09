from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def search_financial_news(query: str) -> str:
    """Search for latest financial news, market updates, and economic events."""
    search = DuckDuckGoSearchRun()
    return search.run(f"{query} financial news")