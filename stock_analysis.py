# https://pypi.org/project/yfinance/
# https://blog.quantinsti.com/stock-market-data-analysis-python/#:~:text=One%20of%20the%20first%20sources,module%20to%20get%20the%20data.&text=To%20visualize%20the%20adjusted%20close,plot%20method%20as%20shown%20below.


import yfinance as yf
import pandas as pd
from datetime import datetime


# Reading the input file

import os
input_file = os.getcwd()
input_file = input_file + "\\us-stocks-analysis\\input\\stocks_data.csv"

# Getting information on the input file
input_mtime = os.path.getmtime(input_file)
m_month = datetime.fromtimestamp(input_mtime).date().month
m_year = datetime.fromtimestamp(input_mtime).date().year

curr_month = datetime.today().month
curr_year = datetime.today().year

### Reading (or updating) input data file
### Creating a list of stock price symbols
stock_list = ["AMZN", "MSFT", "AAPL", "BAC", "WFC", "KO", "AXP", "JPM", "USB"]

if ((m_month == curr_month) & (m_year == curr_year)):
    stock_data = pd.read_csv(input_file)
else:   
    ### Creating necessary variables for data creation
    stock_data = pd.DataFrame()
    start_date = str(datetime.today().year - 21) + "-01-01"
    end_date = str(datetime.today().year - 1) + "-12-31"


    for stocks in stock_list:
        stock_info = yf.Ticker(stocks)
        stock_info = stock_info.history(interval= "1mo", start=start_date, end=end_date).reset_index()
        stock_info['Stock'] = stocks
        
        # Perform further steps depending on the existense of stock_data
        if stock_data.shape[0] == 0:
            stock_data = stock_info
        else:
            stock_data = pd.concat([stock_data, stock_info])

    ### Creating additional columns

    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month_name().str.slice(stop=3)
    stock_data = stock_data[stock_data['Date'].dt.day == 1]

    ### Updating the local input_file
    stock_data.to_csv(input_file, index=False)

### Best month to sell the stocks

temp = stock_data[['High', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['High'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=False).copy()
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_sell = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()

### Best Months to buy the stocks

temp = stock_data[['Low', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['Low'].rank(method='dense', ascending=True).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=True).copy()
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_buy = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()

### Average Growth Per Year

temp = stock_data[['Stock', 'Year', 'Close', 'Month']].loc[stock_data['Month']=="Dec"].copy()
temp['Lagged Close'] = temp.sort_values(by=['Stock', 'Year']).groupby(by=['Stock'])['Close'].shift(1)
temp['Growth %'] = (temp['Close'] - temp['Lagged Close']) / temp['Lagged Close']