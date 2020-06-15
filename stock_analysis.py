# https://pypi.org/project/yfinance/

import yfinance as yf
import pandas as pd


### Creating a list of stock price symbols
stock_list = ["AMZN", "MSFT", "AAPL", "BAC", "WFC", "KO", "AXP", "JPM", "USB"]

stock_data = pd.DataFrame()

for stocks in stock_list:
    stock_info = yf.Ticker(stocks)
    stock_info = stock_info.history(period="max", interval= "1mo").reset_index()
    stock_info['Stock'] = stocks
    
    # Perform further steps depending on the existense of stock_data
    if stock_data.shape[0] == 0:
        stock_data = stock_info
    else:
         stock_data = pd.concat([stock_data, stock_info])


### Creating additional columns

stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month_name().str.slice(stop=3)

stock_data.head()