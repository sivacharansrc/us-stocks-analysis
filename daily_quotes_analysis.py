### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_gbq


# Reference: https://www.datacamp.com/community/tutorials/moving-averages-in-pandas

### CAPTURE CURRENT HOLDINGS IN A LIST ###

daily_data_holdings = "VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC XITK VHT FHLC IHI XHE SMH XSD ARKK ARKW ARKF ARKQ ARKG SLV GLDM FPBFX FIVFX BND AGG FNBGX MRNA MRVL WFC IAU TSLA"

### PREPARING DATA FOR DAILY AVERAGE ###

start_date = str(int(datetime.today().strftime('%Y')) - 2) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 2 Years of data
daily_df = yf.download(daily_data_holdings, start=start_date, interval="1d")
daily_df = daily_df.unstack().reset_index()
daily_df.columns = ['col_name', 'Ticker', 'Date', 'Value']
daily_df = daily_df.pivot_table(values='Value', index=['Ticker', 'Date'],columns='col_name').reset_index()
daily_df = daily_df.sort_values(['Ticker', 'Date'])
daily_df.columns.name = None
col_names = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
daily_df.columns = col_names

# CALCULATING MOVING AVERAGES
daily_df['sma_50'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=50).mean().reset_index(drop=True)
daily_df['sma_100'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=100).mean().reset_index(drop=True)
daily_df['sma_200'] = daily_df.groupby(['ticker'])['adj_close'].rolling(window=200).mean().reset_index(drop=True)

# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(daily_df, 'portfolio_data.daily_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over daily_quotes_analysis successfully completed")
