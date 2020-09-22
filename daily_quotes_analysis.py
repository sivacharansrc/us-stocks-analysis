### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas_gbq


# Reference: https://www.datacamp.com/community/tutorials/moving-averages-in-pandas

### CAPTURE CURRENT HOLDINGS IN A LIST ###

daily_data_holdings = "VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC XITK VHT FHLC IHI XHE VYM SCHD FPBFX FIVFX SMH XSD ARKK ARKW ARKF ARKQ ARKG WCLD SKYY SLV GLDM IAU BND AGG FNBGX WFC TSLA FSCSX FSELX FSPHX FBIOX FFNOX AAPL"
# daily_data_holdings = "VOO ARKK VTI ARKG WCLD"

### CATEGORIZING STOCKS

investment_account = ['VTI', 'FZROX', 'FSKAX', 'VOO', 'IVV', 'FXAIX', 'FNILX', 'VGT', 'FTEC', 'XITK', 'VHT', 'FHLC', 'IHI', 'XHE', 'VYM', 'SCHD', 'FPBFX', 'FIVFX', 'SMH', 'XSD', 'ARKK', 'ARKW', 'ARKF', 'ARKQ', 'ARKG', 'WCLD', 'SKYY', 'SLV', 'GLDM', 'IAU', 'BND', 'AGG', 'FNBGX', 'WFC', 'TSLA', 'AAPL']
retirement_account = ['FZROX', 'FSKAX', 'FXAIX', 'FNILX', 'FSCSX', 'FSELX', 'FSPHX', 'FBIOX', 'FFNOX']
s_and_p_500 = ['FXAIX', 'VOO', 'IVV', 'FNILX']
total_market = ['VTI',  'FZROX', 'FSKAX', 'VOO']
technology = ['VGT', 'FTEC', 'XITK', 'FSCSX']
semiconductors = ['SMH', 'XSD', 'FSELX']
dividends = ['VYM', 'SCHD', 'WFC']
health = ['VHT', 'FHLC', 'IHI', 'XHE', 'FSPHX', 'FBIOX']
high_growth = ['ARKK', 'ARKW', 'ARKF', 'ARKQ', 'ARKG', 'WCLD', 'SKYY', 'TSLA', 'AAPL']
international = ['FPBFX', 'FIVFX', 'FFNOX']
metals = ['SLV', 'GLDM', 'IAU']
bond = ['BND', 'AGG', 'FNBGX']

### PREPARING DATA FOR DAILY AVERAGE ###

start_date = str(int(datetime.today().strftime('%Y')) - 10) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 2 Years of data
daily_df = yf.download(daily_data_holdings, start=start_date, interval="1d")
daily_df = daily_df.unstack().reset_index()
daily_df.columns = ['col_name', 'Ticker', 'Date', 'Value']
daily_df = daily_df.pivot_table(values='Value', index=['Ticker', 'Date'],columns='col_name').reset_index()
daily_df = daily_df.sort_values(['Ticker', 'Date'], ascending=False)
daily_df.columns.name = None
col_names = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
daily_df.columns = col_names
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True) #mergesort works better on pre-sorted items. For most other cases, quicksort works good
daily_df['account'] = np.where((daily_df.ticker.isin(investment_account)) & (daily_df.ticker.isin(retirement_account)), "Both Accounts", np.where(daily_df.ticker.isin(investment_account), "Investment Account", "Retirement Account"))
daily_df['sector'] = np.where(daily_df.ticker.isin(s_and_p_500),'S&P 500', np.where(daily_df.ticker.isin(total_market),'Total Market', np.where(daily_df.ticker.isin(technology),'Technology', np.where(daily_df.ticker.isin(semiconductors),'Semiconductors', np.where(daily_df.ticker.isin(health),'Health Service & Devices', np.where(daily_df.ticker.isin(dividends),'Dividends', np.where(daily_df.ticker.isin(high_growth),'High Growth', np.where(daily_df.ticker.isin(international),'International', np.where(daily_df.ticker.isin(bond),'Bonds', np.where(daily_df.ticker.isin(metals),'Precious Metals','Not Applicable'))))))))))
stock_data_analysis = daily_df.copy()

# CALCULATING MOVING AVERAGES
# daily_df['sma_50'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=50).mean().reset_index(drop=True)
# daily_df['sma_100'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=100).mean().reset_index(drop=True)
# daily_df['sma_200'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=200).mean().reset_index(drop=True)

# CALCULATING OTHER CALCULATED COLUMNS

daily_df['daily_change_pct'] = (daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1)) / daily_df.adj_close
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True)

# CALCULATING CHANGE FROM 52 WEEKS

daily_df['52_wk_comparison_price'] = np.where(daily_df.date.dt.year % 4 == 0, daily_df.groupby(['ticker']).adj_close.shift(366), daily_df.groupby(['ticker']).adj_close.shift(365))
daily_df['first_value'] = daily_df.groupby(['ticker'])['adj_close'].transform('first').reset_index(drop=True)
daily_df['52_wk_comparison_price'] = np.where((daily_df['52_wk_comparison_price'].isnull()) & (daily_df.date == daily_df.date.max()), daily_df['first_value'], daily_df['52_wk_comparison_price'])
daily_df['change_since_52_weeks'] = (daily_df['adj_close'] - daily_df['52_wk_comparison_price']) / daily_df.adj_close


# CALCULATING RSI
# LET US CALCULATE THE RSI USING THE PERIOD = 28
# RSI CALCULATION REFERENCE - https://www.macroption.com/rsi-calculation/

period = 28 # Initializing Period to calculate RSI
daily_df['ups'] = np.where((daily_df.adj_close.isnull()) |  (daily_df.groupby(['ticker']).adj_close.shift(1).isnull()) , np.nan, np.where(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1) > 0, daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1), 0))
daily_df['downs'] = np.where((daily_df.adj_close.isnull()) |  (daily_df.groupby(['ticker']).adj_close.shift(1).isnull()) , np.nan, np.where(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1) < 0, abs(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1)), 0))
daily_df['ups_avg'] = np.where((daily_df.ups.isnull()) | (daily_df.downs.isnull()), np.nan, daily_df.groupby(['ticker'])['ups'].rolling(window=period).mean().reset_index(drop=True).shift(1))
daily_df['downs_avg'] = np.where((daily_df.ups.isnull()) | (daily_df.downs.isnull()), np.nan, daily_df.groupby(['ticker'])['downs'].rolling(window=period).mean().reset_index(drop=True).shift(1))
daily_df['relative_strength_index'] = 100 - (100 / (1 + (daily_df.ups_avg / daily_df.downs_avg)))
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True)

# CHECK THE FIRST DATA POINT FOR A TICKER - daily_df[1480:1521]


# CALCULATE MACD ON TOP OF ADJUSTED CLOSING PRICE
# CALCULATION REFERENCE - https://www.youtube.com/watch?v=9wqvjl_smv4&t=14s&ab_channel=Troy%26Vaishali
# CALCULATION REFERENCE - https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp
# PYTHON REFERENCE - https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9

faster_period = 12
slower_period = 26
signal_period = 9

daily_df['ema_12'] = daily_df.groupby(['ticker'])['close'].transform(lambda x: x.ewm(span=faster_period, adjust=False).mean().reset_index(drop=True))
daily_df['ema_26'] = daily_df.groupby(['ticker'])['close'].transform(lambda x: x.ewm(span=slower_period, adjust=False).mean().reset_index(drop=True))
daily_df['macd_line'] = daily_df['ema_12'] - daily_df['ema_26']
daily_df['signal_line'] = daily_df.groupby(['ticker'])['macd_line'].transform(lambda x: x.ewm(span=signal_period, adjust=False).mean().reset_index(drop=True))
daily_df['macd_histogram'] = daily_df['macd_line'] - daily_df['signal_line']

# SUBSETTING 10 YEARS DATA AND CREATING MAX AND MIN DATE FILTER

# daily_df = daily_df[daily_df.date.dt.year >= daily_df.date.dt.year.max()-10].reset_index(drop=True)
daily_df['date_filter'] = np.where(daily_df.date == daily_df.groupby('ticker').date.transform('max').reset_index(drop=True),"MAX", "NO")


# SUBSETTING THE COLUMNS TO KEEP
cols_to_keep = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume', 'account', 'sector', 'daily_change_pct', 'date_filter', 'relative_strength_index', 'change_since_52_weeks', 'ema_12', 'ema_26', 'macd_line', 'signal_line', 'macd_histogram']
daily_df = daily_df[cols_to_keep]

# PREPARING THE STOCK ANALYSIS DATA

# Date	Open	High	Low	Close	Volume	Dividends	Stock Splits	Stock	Adj Close	Year	Month
# ticker open   high    low close   volume                              ticker  adj_close   year    month

# stock_data_analysis['year'] = stock_data_analysis.date.dt.year
# stock_data_analysis['month'] = stock_data_analysis['date'].dt.month_name().str.slice(stop=3)

# ## TOP MONTHS TO SELL THE STOCK

temp = stock_data_analysis[['high', 'ticker', 'year', 'month']]
temp['mean_highs'] = temp.groupby(['ticker', 'year', 'month'])['high'].transform('mean').reset_index(drop=True)
temp['rank'] = temp.groupby(['ticker', 'year'])['mean_highs'].rank(method='dense', ascending=False)
temp = temp.loc[temp['rank'] <= 3].sort_values(by=['ticker', 'year', 'rank'], ascending=False)
temp = temp.pivot_table(index=['ticker', 'month'], aggfunc={'year': 'count'}).reset_index()
temp['rank'] = temp.groupby(['ticker'])['year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['rank'] <= 3].sort_values(by=['ticker', 'rank'], ascending=True).copy()
months_to_sell = temp.groupby(['ticker'])['month'].apply('-'.join).reset_index()
months_to_sell.columns = ['ticker', 'months_to_sell']

### TOP MONTHS TO SELL THE STOCK

temp = stock_data[['High', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['High'].rank(method='dense', ascending=False)
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=False)
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_sell = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()
months_to_sell.columns = ['Stock', 'Months to Sell']

# ### TOP MONTHS TO BUY THE STOCKS

# temp = stock_data_analysis[['low', 'adj_close', 'ticker', 'year', 'month']]
# temp['mean_lows'] = temp.groupby(['ticker', 'year', 'month'])['low'].transform('mean').reset_index(drop=True)
# temp['rank'] = temp.groupby(['ticker', 'year'])['mean_lows'].rank(method='dense', ascending=True)
# temp = temp.loc[temp['rank'] <= 3].sort_values(by=['ticker', 'year', 'rank'], ascending=True)
# temp = temp.pivot_table(index=['ticker', 'month'], aggfunc={'year': 'count'}).reset_index()
# temp['rank'] = temp.groupby(['ticker'])['year'].rank(method='dense', ascending=True).copy()
# temp = temp.loc[temp['rank'] <= 3].sort_values(by=['ticker', 'rank'], ascending=True).copy()
# months_to_buy = temp.groupby(['ticker'])['month'].apply('-'.join).reset_index()
# months_to_buy.columns = ['ticker', 'months_to_sell']



### TOP MONTHS TO BUY THE STOCKS

temp = stock_data[['Low', 'Stock', 'Year', 'Month']]
temp['Rank'] = temp.groupby(['Stock', 'Year'])['Low'].rank(method='dense', ascending=True).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Year', 'Rank'], ascending=True).copy()
temp = temp.pivot_table(index=['Stock', 'Month'], aggfunc={'Year': 'count'}).reset_index()
temp['Rank'] = temp.groupby(['Stock'])['Year'].rank(method='dense', ascending=False).copy()
temp = temp.loc[temp['Rank'] <= 3].sort_values(by=['Stock', 'Rank'], ascending=True).copy()
months_to_buy = temp.groupby(['Stock'])['Month'].apply('-'.join).reset_index()
months_to_buy.columns = ['Stock', 'Months to Buy']


# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(daily_df, 'portfolio_data.daily_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over daily_quotes_analysis successfully completed")
