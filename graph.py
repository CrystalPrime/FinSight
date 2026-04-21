from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from agents.rag import search_financial_documents
from agents.markets import get_stock_price, compare_stocks
from agents.math import calculate, calculate_roi
from agents.news import search_financial_news
from config import GROQ_MODEL
import os
import operator
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --- State ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    final_answer: str
    parallel_results: Annotated[list, operator.add]
    is_parallel: bool  # ← YENİ: paralel modda mı?


# --- LLM ---
def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# --- Agentlar ---
rag_agent = create_agent(
    model=get_llm(),
    tools=[search_financial_documents],
    system_prompt="You are a financial document specialist. Search uploaded documents and provide accurate information with source citations.",
    middleware=[ToolCallLimitMiddleware(run_limit=3)],
)
market_agent = create_agent(
    model=get_llm(),
    tools=[get_stock_price, compare_stocks],
    system_prompt="You are a market data specialist. Fetch and analyze real-time stock prices and market data.",
    middleware=[ToolCallLimitMiddleware(run_limit=3)],
)
math_agent = create_agent(
    model=get_llm(),
    tools=[calculate, calculate_roi],
    system_prompt="You are a financial calculation specialist. Perform precise financial calculations like ROI, compound interest, and comparisons.",
    middleware=[ToolCallLimitMiddleware(run_limit=3)],
)
news_agent = create_agent(
    model=get_llm(),
    tools=[search_financial_news],
    system_prompt="You are a financial news specialist. Find and summarize the latest financial news and market events.",
    middleware=[ToolCallLimitMiddleware(run_limit=3)],
)


# --- Supervisor ---
SUPERVISOR_PROMPT = """You are a financial assistant supervisor.
Based on the user's question, decide which specialist to call.

Available specialists:
- rag_agent: for questions about uploaded financial documents/reports
- market_agent: for stock prices, market data, ticker symbols
- math_agent: for financial calculations, ROI, compound interest
- news_agent: for latest financial news and market events
- PARALLEL_MARKET_NEWS: for questions asking BOTH price AND news together
- FINISH: for greetings, general conversation

Examples:
- "AAPL fiyatı ve haberleri" → PARALLEL_MARKET_NEWS
- "Tesla hem fiyat hem haber" → PARALLEL_MARKET_NEWS
- "AAPL fiyatı" → market_agent
- "Tesla haberleri" → news_agent

Respond with ONLY one of: rag_agent, market_agent, math_agent, news_agent, PARALLEL_MARKET_NEWS, FINISH"""


def supervisor_node(state: State) -> State:
    llm = get_llm()
    messages = state["messages"]

    last_human_idx = None
    for i, m in enumerate(messages):
        if hasattr(m, "type") and m.type == "human":
            last_human_idx = i

    if last_human_idx is None:
        return {"next_agent": "FINISH", "is_parallel": False}

    messages_after = messages[last_human_idx + 1:]
    ai_after = [m for m in messages_after if hasattr(m, "type") and m.type == "ai"]

    if len(ai_after) >= 1:
        print(f"[supervisor] bu tur cevaplandı, FINISH")
        return {"next_agent": "FINISH", "is_parallel": False}

    last_user_msg = messages[last_human_idx].content
    response = llm.invoke([
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": str(last_user_msg)}
    ])

    next_agent = response.content.strip()
    print(f"[supervisor] seçti: '{next_agent}'")

    VALID = ["rag_agent", "market_agent", "math_agent", "news_agent", "PARALLEL_MARKET_NEWS"]
    if next_agent not in VALID:  # ← artık PARALLEL_MARKET_NEWS geçiyor
        next_agent = "FINISH"

    is_parallel = next_agent == "PARALLEL_MARKET_NEWS"
    return {"next_agent": next_agent, "is_parallel": is_parallel}


def route_supervisor(state: State):
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "PARALLEL_MARKET_NEWS":
        return [
            Send("market_agent", state),
            Send("news_agent", state),
        ]
    if next_agent == "FINISH":
        return END
    return next_agent


# --- Agent node'ları ---
def run_agent(agent, state: State, config: RunnableConfig) -> State:
    print(f"[agent] çalışıyor: {state['messages'][-1].content[:60]}")
    result = agent.invoke({"messages": state["messages"]}, config=config)
    answer = result["messages"][-1].content
    print(f"[agent] cevap: {answer[:80]}")
    
    # Paralel moddaysa parallel_results'a ekle, messages'a değil
    if state.get("is_parallel"):
        return {"parallel_results": [answer]}
    
    return {
        "messages": [AIMessage(content=answer)],
        "next_agent": "FINISH",
    }


def rag_node(state: State, config: RunnableConfig) -> State:
    return run_agent(rag_agent, state, config)

def market_node(state: State, config: RunnableConfig) -> State:
    return run_agent(market_agent, state, config)

def math_node(state: State, config: RunnableConfig) -> State:
    return run_agent(math_agent, state, config)

def news_node(state: State, config: RunnableConfig) -> State:
    return run_agent(news_agent, state, config)


# --- Merge node: paralel sonuçları birleştirir ---
def merge_parallel_node(state: State) -> State:
    results = state.get("parallel_results", [])
    print(f"[merge] {len(results)} sonuç birleştiriliyor")

    if len(results) >= 2:
        merged = f"""📈 **Piyasa Verisi**
{results[0]}

---

📰 **Son Haberler**
{results[1]}"""
    elif len(results) == 1:
        merged = results[0]
    else:
        merged = "Sonuç alınamadı."

    return {
        "messages": [AIMessage(content=merged)],
        "next_agent": "FINISH",
        "parallel_results": [],
        "is_parallel": False,
    }


# --- Graph ---
def build_graph():
    graph = StateGraph(State)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("market_agent", market_node)
    graph.add_node("math_agent", math_node)
    graph.add_node("news_agent", news_node)
    graph.add_node("merge_parallel", merge_parallel_node)  # ← YENİ

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor)

    # Tekli agentlar → supervisor'a döner
    graph.add_edge("rag_agent", "supervisor")
    graph.add_edge("math_agent", "supervisor")

    # Paralel agentlar → merge node'a gider
    graph.add_edge("market_agent", "merge_parallel")  # ← DEĞİŞTİ
    graph.add_edge("news_agent", "merge_parallel")    # ← DEĞİŞTİ

    # Merge → supervisor (FINISH kontrolü için)
    graph.add_edge("merge_parallel", "supervisor")

    return graph.compile(checkpointer=MemorySaver())


finance_graph = build_graph()
