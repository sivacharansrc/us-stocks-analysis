# https://pypi.org/project/yfinance/
# https://blog.quantinsti.com/stock-market-data-analysis-python/#:~:text=One%20of%20the%20first%20sources,module%20to%20get%20the%20data.&text=To%20visualize%20the%20adjusted%20close,plot%20method%20as%20shown%20below.

### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import os


# GENERATING THE PATH OF THE INPUT STOCK DATA FILE

path = os.getcwd()
input_file = path + "\\us-stocks-analysis\\input\\stocks_data.csv"
dividends_file = path + "\\us-stocks-analysis\\input\\dividends_data.csv"
splits_file = path + "\\us-stocks-analysis\\input\\splits_data.csv"

### CHECK IF STOCK DATA INPUT FILE EXISTS AND GET THE FILE MODIFIED TIME
if os.path.exists(input_file):
    # Getting information on the input file
    input_mtime = os.path.getmtime(input_file)
    m_month = datetime.fromtimestamp(input_mtime).date().month
    m_year = datetime.fromtimestamp(input_mtime).date().year
else:
    m_month = 1
    m_year = 1900

### GET THE CURRENT DATE INFORMATION TO COMARE WITH THE STOCK INPUT DATA MODIFIED TIME
curr_month = datetime.today().month
curr_year = datetime.today().year

### READING (OR UPDATING INPUT FILE)
### CREATING LIST OF STOCK SYMBOLS OF INTEREST
stock_list = ["AMZN", "MSFT", "AAPL", "BAC", "WFC", "KO", "AXP", "JPM", "USB", "T", "COST", "EPD", "HON", "BAYRY"]

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
        
        # PERFORM FURTHER STEPS DEPENDING ON THE EXISTENSE OF STOCK_DATA
        if stock_data.shape[0] == 0:
            stock_data = stock_info
        else:
            stock_data = pd.concat([stock_data, stock_info])
    stock_data = stock_data[stock_data['Open'].notnull()]

    ### CREATING ADDITIONAL COLUMNS

    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month_name().str.slice(stop=3)
    stock_data = stock_data[stock_data['Date'].dt.day == 1]

    ### CREATE OR OVERWRITE THE STOCK DATA INPUT FILE
    stock_data.to_csv(input_file, index=False)


### TOP MONTHS TO SELL THE STOCK

temp = stock_data[['High', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['High'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=False).copy()
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_sell = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()

### TOP MONTHS TO BUY THE STOCKS

temp = stock_data[['Low', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['Low'].rank(method='dense', ascending=True).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=True).copy()
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_buy = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()

### GENERATE GROWTH METRICS FOR THE STOCKS

temp = stock_data[['Stock', 'Year', 'Close', 'Month']].loc[stock_data['Month']=="Dec"].sort_values(by=['Stock', 'Year']).reset_index().copy()
temp['Lagged Close'] = temp.groupby(by=['Stock'])['Close'].shift(1)
temp['Avg Yearly Growth'] = (temp['Close'] - temp['Lagged Close']) / temp['Lagged Close']
avg_yearly_growth = temp.groupby('Stock')['Avg Yearly Growth'].mean().reset_index()
temp['Growth Probability'] = np.where(np.isnan(temp['Avg Yearly Growth']), np.NaN, np.where(temp['Avg Yearly Growth'] > 0,1,-1))
growth_tendency = (temp.groupby('Stock')['Growth Probability'].sum() / (temp.groupby('Stock')['Growth Probability'].count())).reset_index()
growth_metrics = pd.merge(avg_yearly_growth, growth_tendency, how="inner")

### CALCULATING DIVIDENDS ACCUMULATED IN A YEAR

### GETTING THE DIVIDEND INFORMATION

if ((m_month == curr_month) & (m_year == curr_year)):
    dividends_data = pd.read_csv(dividends_file)
    splits_data = pd.read_csv(splits_file)
else:   
    ### Creating necessary variables for data creation
    actions_data = pd.DataFrame()
    
    for stocks in stock_list:
        actions_info = yf.Ticker(stocks)
        actions_info = actions_info.actions.reset_index()
        actions_info['Stock'] = stocks
        
        # PERFORM FURTHER STEPS DEPENDING ON THE EXISTENSE OF ACTIONS_DATA
        if actions_data.shape[0] == 0:
            actions_data = actions_info
        else:
            actions_data = pd.concat([actions_data, actions_info])
    
    ### CREATING ADDITIONAL COLUMNS

    actions_data['Year'] = actions_data['Date'].dt.year
    actions_data['Month'] = actions_data['Date'].dt.month_name().str.slice(stop=3)
    actions_data = actions_data[(actions_data['Year'] >= int(start_date[0:4])) & (actions_data['Year'] <= int(end_date[0:4]))]

    ### SEPERATING DIVIDENDS AND SPLITS DATA

    dividends_data = actions_data.groupby(['Stock', 'Year','Month'])['Dividends'].sum().reset_index()
    dividends_data = dividends_data[(dividends_data['Dividends'] != 0) & (dividends_data['Year'] >= (int(start_date[0:4]) +11))] # RETAINING ONLY 11 YEARS DIVIDEND DATA

    splits_data = actions_data.groupby(['Stock', 'Year','Month'])['Stock Splits'].sum().reset_index()
    splits_data = splits_data[splits_data['Stock Splits'] != 0]

    ### CREATE OR OVERWRITE THE STOCK DATA INPUT FILE
    dividends_data.to_csv(dividends_file, index=False)
    splits_data.to_csv(splits_file, index=False)

# dividends_data = pd.read_csv(dividends_file)

dividends_data = dividends_data.groupby(by=['Stock', 'Year'])['Dividends'].sum().reset_index()
dividends_data = pd.merge(dividends_data, stock_data[['Stock', 'Year','Close']][stock_data['Month'] == "Dec"], how="left", on=['Stock', 'Year'])
dividends_data['Dividend PCT'] = dividends_data['Dividends'] / dividends_data['Close']
dividends_data['Weight'] = dividends_data.groupby('Stock')['Year'].rank(method='dense', ascending=True).copy()

### CALCULATING 10 YR DIVIDEND YIELD
avg_annual_dividend_pct = dividends_data.groupby('Stock').apply(lambda x: np.average(x['Dividend PCT'], weights=x['Weight'])).reset_index()
avg_annual_dividend_pct.columns = ['Stock','10 Year Avg Dividend Yield']

### CALCULATING DIVIDEND GROWTH AND GROWTH TENDENCY
dividends_data['Lagged Dividend'] = dividends_data.groupby(by=['Stock'])['Dividends'].shift(1)
dividends_data['Dividend Avg Yearly Growth'] = (dividends_data['Dividends'] - dividends_data['Lagged Dividend']) / dividends_data['Lagged Dividend']
avg_yearly_growth = dividends_data.groupby('Stock')['Dividend Avg Yearly Growth'].mean().reset_index()
dividends_data['Dividend Growth Probability'] = np.where(np.isnan(dividends_data['Dividend Avg Yearly Growth']), np.NaN, np.where(dividends_data['Dividend Avg Yearly Growth'] > 0,1,-1))
growth_tendency = (dividends_data.groupby('Stock')['Dividend Growth Probability'].sum() / (dividends_data.groupby('Stock')['Dividend Growth Probability'].count())).reset_index()
dividend_metrics = pd.merge(avg_annual_dividend_pct, avg_yearly_growth, how='inner', on='Stock')
dividend_metrics = pd.merge(dividend_metrics, growth_tendency, how='inner', on='Stock')

### OTHER THINGS TO DO

# HOW VOLATILE IS THE STOCK WITHIN A MONTH AND WITHIN A YEAR - USE LAST 5 YEARS DAILY DATA
# HOW MUCH DIVIDEND ACCUMULATED ANNUALLY (%)
# AVERAGE NUMBER OF SPLITS PAST 20 YEARS? THE MORE THE SPLIT, THE MORE THE GROWTH OF COMPANY (LESS WEIGHTAGE)
# HOW DOES THE STOCK COMPARE TO THE SECTOR AVERAGE
# CALCULATE THE SMALL, MID, AND LARGE CAP
# FINALLY CREATE RANKING BY SECTOR, CAP, GENERAL RANKING


### GETTING OTHER STOCK INFORMATION

file_names = ['nasdaq_stock_list.csv','nyse_stock_list.csv','amex_stock_list.csv']
for market in file_names:
    ### READ DATA AND ASSOCIATE THE MARKET INFO WITHIN THE FILE
    stock_info = pd.read_csv(path + "\\us-stocks-analysis\\input\\" + market)
    stock_info = stock_info[['Symbol', 'Name', 'Sector', 'industry']][stock_info['Symbol'].isin(stock_list)]
stock_info
