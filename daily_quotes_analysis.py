### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_gbq


# Reference: https://www.datacamp.com/community/tutorials/moving-averages-in-pandas

### CAPTURE CURRENT HOLDINGS IN A LIST ###

daily_data_holdings = "VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC XITK VHT FHLC IHI XHE SMH XSD ARKK ARKW ARKF ARKQ ARKG SLV GLDM FPBFX FIVFX BND AGG FNBGX MRNA MRVL WFC IAU TSLA"
# daily_data_holdings = "VOO ARKK"

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
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort') #mergesort works better on pre-sorted items. For most other cases, quicksort works good

# CALCULATING MOVING AVERAGES
# daily_df['sma_50'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=50).mean().reset_index(drop=True)
# daily_df['sma_100'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=100).mean().reset_index(drop=True)
# daily_df['sma_200'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=200).mean().reset_index(drop=True)

# CALCULATING OTHER CALCULATED COLUMNS
 
daily_df['daily_change_pct'] = (daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1)) / daily_df.adj_close
daily_df['max_date_filter'] = np.where(daily_df.date == daily_df.groupby('ticker').date.transform('max').reset_index(drop=True),"YES", "NO")
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort')

# CALCULATING RSI
# LET US CALCULATE THE RSI USING THE PERIOD = 28
# RSI CALCULATION REFERENCE - https://www.macroption.com/rsi-calculation/

period = 28 # Initializing Period to calculate RSI
daily_df['ups'] = np.where(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1) > 0, daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1), 0)
daily_df['downs'] = np.where(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1) < 0, abs(daily_df.adj_close - daily_df.groupby(['ticker']).adj_close.shift(1)), 0)
daily_df['ups_avg'] = daily_df.groupby(['ticker'])['ups'].rolling(window=period).mean().reset_index(drop=True).shift(1)
daily_df['downs_avg'] = daily_df.groupby(['ticker'])['downs'].rolling(window=period).mean().reset_index(drop=True).shift(1)
daily_df['relative_strength_index'] = 100 - (100 / (1 + (daily_df.ups_avg / daily_df.downs_avg)))
daily_df = daily_df.sort_values(by=['ticker', 'date'], axis=0, ascending=True, kind='mergesort')

# CALCULATE MACD ON TOP OF ADJUSTED CLOSING PRICE
# CALCULATION REFERENCE - https://www.youtube.com/watch?v=9wqvjl_smv4&t=14s&ab_channel=Troy%26Vaishali
# CALCULATION REFERENCE - https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp
# PYTHON REFERENCE - https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9

daily_df['12_ema'] = daily_df.close.ewm(span=12, adjust=False).mean().reset_index(drop=True)
daily_df['26_ema'] = daily_df.close.ewm(span=26, adjust=False).mean().reset_index(drop=True)
daily_df['macd_line'] = daily_df['12_ema'] - daily_df['26_ema']
daily_df['signal_line'] = daily_df['macd_line'].ewm(span=9, adjust=False).mean().reset_index(drop=True)
daily_df['macd_histogram'] = daily_df['macd_line'] - daily_df['signal_line']

# SUBSETTING THE COLUMNS TO KEEP
cols_to_keep = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume', 'daily_change_pct', 'max_date_filter', 'relative_strength_index', '12_ema', '26_ema', 'macd_line', 'signal_line', 'macd_histogram']
daily_df = daily_df[cols_to_keep]

# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(daily_df, 'portfolio_data.daily_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over daily_quotes_analysis successfully completed")
