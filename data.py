import yfinance as yf

def get_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    return data
