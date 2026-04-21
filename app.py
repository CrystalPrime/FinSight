import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import uuid
load_dotenv()

from graph import finance_graph

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

THREAD_CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="centered"
)

st.title("📈 FinSight")
st.caption("AI-Powered Market Intelligence")

if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_ticker(text: str) -> str | None:
    common_tickers = {
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "NFLX", "AMD", "INTC", "BABA", "SHOP",
        "COIN", "UBER", "LYFT", "SNOW", "PLTR", "SOFI"
    }
    words = text.upper().split()
    for word in words:
        clean = re.sub(r'[^A-Z]', '', word)
        if clean in common_tickers:
            return clean
    matches = re.findall(r'\b([A-Z]{2,5})\b', text.upper())
    for m in matches:
        if m in common_tickers:
            return m
    return None

def get_stock_chart(ticker: str, period: str = "3mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info

        if hist.empty:
            return None, None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05
        )

        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1, col=1
        )

        colors = [
            "#22c55e" if hist["Close"].iloc[i] >= hist["Open"].iloc[i]
            else "#ef4444"
            for i in range(len(hist))
        ]

        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1
        )

        hist = hist.dropna()  # NaN satırları temizle

        if len(hist) < 2:
            return None, None

        current_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2])
        change = ((current_price - prev_price) / prev_price) * 100
        
        
        fig.update_layout(
            title=f"{ticker} — {info.get('longName', ticker)} | ${current_price:.2f} ({change:+.2f}%)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=450,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig, {
            "price": current_price,
            "change": change,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", "N/A"),
        }
    except Exception:
        return None, None

def ask_agent(user_input: str) -> str:
    result = finance_graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=THREAD_CONFIG,
    )
    messages = result["messages"]

    last_human_idx = None
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human_idx = i

    answer = None
    if last_human_idx is not None:
        for m in messages[last_human_idx + 1:]:
            if isinstance(m, AIMessage) and m.content.strip():
                answer = m.content
                break

    # Agent cevap vermediyse (selam, genel soru vs.) LLM'e direkt sor
    if not answer:
        from langchain_groq import ChatGroq
        import os
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        response = llm.invoke([
            {
                "role": "system",
                "content": """You are a friendly financial assistant.
You help users with stock analysis, market data, financial calculations, and news.
For greetings and general questions, respond warmly and briefly.
Always answer in the user's language."""
            },
            {"role": "user", "content": user_input}
        ])
        answer = response.content

    return answer

# --- UI ---
st.title("📈 Finance Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("ticker"):
            fig, info = get_stock_chart(msg["ticker"])
            if fig and info:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Price",
                    f"${info['price']:.2f}" if info['price'] == info['price'] else "N/A",  # nan kontrolü
                    f"{info['change']:+.2f}%" if info['change'] == info['change'] else ""
                )

                m2.metric("P/E Ratio", info["pe_ratio"])
                m3.metric(
                    "Market Cap",
                    f"${info['market_cap']/1e9:.1f}B" if info["market_cap"] else "N/A"
                )
                st.plotly_chart(fig, use_container_width=True)

if user_input := st.chat_input("Hisse sor, hesaplat, haber ara..."):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    ticker = extract_ticker(user_input)

    with st.spinner("Düşünüyor..."):
        answer = ask_agent(user_input)

    if not ticker:
        ticker = extract_ticker(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "ticker": ticker,
    })

    st.rerun()
