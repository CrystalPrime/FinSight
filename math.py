from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate financial math expressions.
    Examples: '(1500 - 1000) / 1000 * 100' for ROI, '1000 * (1 + 0.08) ** 10' for compound interest."""
    try:
        result = eval(expression, {"__builtins__": {}, "pow": pow, "abs": abs, "round": round})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def calculate_roi(buy_price: float, sell_price: float, quantity: int = 1) -> str:
    """Calculate Return on Investment given buy price, sell price and quantity."""
    profit = (sell_price - buy_price) * quantity
    roi = ((sell_price - buy_price) / buy_price) * 100
    return (
        f"Buy: ${buy_price} × {quantity} = ${buy_price * quantity:,.2f}\n"
        f"Sell: ${sell_price} × {quantity} = ${sell_price * quantity:,.2f}\n"
        f"Profit: ${profit:,.2f}\n"
        f"ROI: {roi:.2f}%"
    )