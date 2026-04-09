from langchain.tools import tool
import yfinance as yf

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and basic info for a ticker symbol like AAPL, TSLA, MSFT."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="5d")

        if hist.empty:
            return f"No data found for {ticker}."

        current = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
        change = ((current - prev) / prev) * 100

        return (
            f"{ticker.upper()} — {info.get('longName', 'N/A')}\n"
            f"Price: ${current:.2f}\n"
            f"Change: {change:+.2f}%\n"
            f"Market Cap: {info.get('marketCap', 'N/A'):,}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"52w High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52w Low: {info.get('fiftyTwoWeekLow', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching {ticker}: {e}"

@tool
def compare_stocks(tickers: str) -> str:
    """Compare multiple stocks. Input: comma-separated tickers like 'AAPL,MSFT,GOOGL'."""
    results = []
    for ticker in tickers.split(","):
        ticker = ticker.strip().upper()
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if hist.empty:
                continue
            monthly_return = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
            results.append(f"{ticker}: {monthly_return:+.2f}% (1 month)")
        except:
            results.append(f"{ticker}: error")
    return "\n".join(results)