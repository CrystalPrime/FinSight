# Finance Assistant — Multi-Agent AI System

A production-grade financial assistant powered by a **LangGraph multi-agent architecture**. The system routes user queries to specialized AI agents, fetches real-time market data, and renders interactive stock charts — all in a conversational interface.

## Architecture

```
User Query
    │
    ▼
Supervisor (LLM)
decides which agent(s) to call
    │
    ├──────────────────────────────────────────────┐
    │                    │                         │
    ▼                    ▼                         ▼
RAG Agent          Market Agent            Math Agent          News Agent
Financial docs     Yahoo Finance           ROI, compound       DuckDuckGo
FAISS + PDF        real-time prices        interest, ratios    financial news
    │                    │                         │                │
    └────────────────────┴─────────────────────────┴────────────────┘
                                  │
                            Supervisor
                         synthesizes answer
                                  │
                                  ▼
                      Streamlit Chat Interface
                      + Interactive Stock Chart
```

Each agent is a specialized `create_agent` instance with its own tools and system prompt. The Supervisor node uses an LLM to route queries, then returns control after each agent responds — enabling multi-step reasoning across agents.

## Features

- **Multi-agent routing** — Supervisor LLM decides which specialist to invoke based on the query
- **Real-time market data** — Live stock prices, P/E ratios, market cap via Yahoo Finance
- **Interactive charts** — Candlestick + volume charts rendered inline in chat (Plotly)
- **Financial document RAG** — Upload PDFs and ask questions with page-level citations
- **Financial calculations** — ROI, compound interest, comparative analysis
- **Financial news search** — Latest market events via DuckDuckGo
- **Conversation memory** — Full multi-turn memory via LangGraph checkpointer

## Stack

- **LangGraph** — Multi-agent graph orchestration (StateGraph, conditional edges)
- **LangChain** — Agent creation, tool definitions, RAG pipeline
- **Groq** — LLM inference (llama-3.3-70b-versatile)
- **FAISS** — Local vector store for document retrieval
- **HuggingFace Embeddings** — Offline embedding model (all-MiniLM-L6-v2)
- **Yahoo Finance (yfinance)** — Real-time stock data
- **Plotly** — Interactive financial charts
- **Streamlit** — Chat interface

## Agents

| Agent | Tools | Triggers |
|---|---|---|
| **Market Agent** | `get_stock_price`, `compare_stocks` | Ticker symbols, price queries |
| **News Agent** | `search_financial_news` | Market news, recent events |
| **Math Agent** | `calculate`, `calculate_roi` | ROI, compound interest, ratios |
| **RAG Agent** | `search_financial_documents` | Questions about uploaded PDFs |

## Setup

**1. Clone and install**

```bash
git clone https://github.com/CrystalPrime/finance-agent
cd finance-agent
pip install -r requirements.txt
```

**2. Download embedding model**

```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

Update the path in `backend/config.py`:

```python
EMBEDDING_MODEL = "path/to/all-MiniLM-L6-v2"
```

**3. Set environment variables**

```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

**4. Run**

```bash
cd backend
streamlit run app.py
```

## Example queries

```
""What is the AAPL stock price?"        → Market Agent → live price + chart  "          → Market Agent  → live price + chart
"What are the latest Tesla news?"         → News Agent    → latest news
"If I buy at $100 and sell at $150, what’s the ROI?"   → Math Agent    → calculation
"What does the uploaded report say?"    → RAG Agent     → document search
"Compare NVDA and AMD"       → Market Agent  → comparison + chart
```

## Project structure

```
finance-agent/
├── backend/
│   ├── app.py           # Streamlit UI + chart rendering
│   ├── graph.py         # LangGraph StateGraph — supervisor + routing
│   ├── ingest.py        # PDF → chunks → FAISS index
│   ├── config.py        # model paths and settings
│   └── agents/
│       ├── market.py    # Yahoo Finance tools
│       ├── news.py      # DuckDuckGo news search
│       ├── math.py      # financial calculations
│       └── rag.py       # document retrieval tool
├── requirements.txt
└── .env
```

## Requirements

```
streamlit
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-text-splitters
langgraph
faiss-cpu
sentence-transformers
pypdf
yfinance
plotly
duckduckgo-search
python-dotenv
```

---

*Part of an ongoing series of LangChain and LangGraph projects — see also [research-agent](https://github.com/CrystalPrime/research-agent) and [book-agent](https://github.com/CrystalPrime/book-agent).*
