### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_gbq

### CAPTURE CURRENT HOLDINGS IN A LIST ###

minute_data_holdings = "VTI VOO IVV VGT FTEC XITK VHT FHLC IHI XHE SMH XSD ARKK ARKW ARKF ARKQ ARKG SLV GLDM IAU BND AGG MRNA MRVL WFC TSLA"
# minute_data_holdings = "VTI VOO IVV"

### PREPARING DATA FOR DAILY AVERAGE ###

minute_df = yf.download(minute_data_holdings, period="5d", interval="1m")
minute_df = minute_df.unstack().reset_index()
minute_df.columns = ['col_name', 'Ticker', 'Date', 'Value']
minute_df = minute_df.pivot_table(values='Value', index=['Ticker', 'Date'],columns='col_name').reset_index()
minute_df = minute_df.sort_values(['Ticker', 'Date'], ascending=False)
minute_df.columns.name = None
col_names = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
minute_df.columns = col_names
minute_df.date = minute_df.date.dt.tz_convert('US/Central')
minute_df = minute_df[minute_df.date.dt.date == minute_df.date.dt.date.max()]
minute_df = minute_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True) #mergesort works better on pre-sorted items. For most other cases, quicksort works good

# CALCULATING MOVING AVERAGES
# minute_df['sma_30'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=30).mean().reset_index(drop=True)
# minute_df['sma_120'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=120).mean().reset_index(drop=True)
# minute_df['sma_240'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=240).mean().reset_index(drop=True)

# CALCULATING OTHER CALCULATED COLUMNS
minute_df['prev_close'] = minute_df.groupby(['ticker']).adj_close.shift(1)
minute_df['minute_change_pct'] = (minute_df.adj_close - minute_df.prev_close) / minute_df.adj_close
minute_df['max_date_filter'] = np.where(minute_df.date == minute_df.groupby('ticker').date.transform('max').reset_index(drop=True),"YES", "NO")
minute_df = minute_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True)
minute_df['open_price'] = minute_df.groupby('ticker').open.transform('first').reset_index(drop=True)
minute_df['change_since_open'] = (minute_df.adj_close - minute_df.open_price) / minute_df.adj_close
minute_df = minute_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True)

# CALCULATING RSI
# LET US CALCULATE THE RSI USING THE PERIOD = 28
# RSI CALCULATION REFERENCE - https://www.macroption.com/rsi-calculation/

period = 14 # Initializing Period to calculate RSI
minute_df['ups'] = np.where(minute_df.adj_close - minute_df.prev_close > 0, minute_df.adj_close - minute_df.prev_close, 0)
minute_df['downs'] = np.where(minute_df.adj_close - minute_df.prev_close < 0, abs(minute_df.adj_close - minute_df.prev_close), 0)
minute_df['ups_avg'] = minute_df.groupby(['ticker'])['ups'].rolling(window=period).mean().reset_index(drop=True)
minute_df['downs_avg'] = minute_df.groupby(['ticker'])['downs'].rolling(window=period).mean().reset_index(drop=True)
minute_df['relative_strength_index'] = 100 - (100 / (1 + (minute_df.ups_avg / minute_df.downs_avg)))
minute_df = minute_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True)


# CALCULATE MACD ON TOP OF ADJUSTED CLOSING PRICE
# CALCULATION REFERENCE - https://www.youtube.com/watch?v=9wqvjl_smv4&t=14s&ab_channel=Troy%26Vaishali
# CALCULATION REFERENCE - https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp
# PYTHON REFERENCE - https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9

minute_df['ema_12'] = minute_df.close.ewm(span=12, adjust=False).mean().reset_index(drop=True)
minute_df['ema_26'] = minute_df.close.ewm(span=26, adjust=False).mean().reset_index(drop=True)
minute_df['macd_line'] = minute_df['ema_12'] - minute_df['ema_26']
minute_df['signal_line'] = minute_df['macd_line'].ewm(span=9, adjust=False).mean().reset_index(drop=True)
minute_df['macd_histogram'] = minute_df['macd_line'] - minute_df['signal_line']

# SUBSETTING THE COLUMNS TO KEEP
cols_to_keep = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume', 'minute_change_pct', 'change_since_open','max_date_filter', 'relative_strength_index', 'ema_12', 'ema_26', 'macd_line', 'signal_line', 'macd_histogram']
minute_df = minute_df[cols_to_keep]

# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(minute_df, 'portfolio_data.minute_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over minute_quotes_analysis successfully completed")

