### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_gbq

### CAPTURE CURRENT HOLDINGS IN A LIST ###

minute_data_holdings = "VTI VOO IVV VGT FTEC XITK VHT FHLC IHI XHE SMH XSD ARKK ARKW ARKF ARKQ ARKG SLV GLDM IAU BND AGG MRNA MRVL WFC TSLA"

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

# CALCULATING MOVING AVERAGES
minute_df['sma_30'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=30).mean().reset_index(drop=True)
minute_df['sma_120'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=120).mean().reset_index(drop=True)
minute_df['sma_240'] = minute_df.groupby(['ticker'])['adj_close'].rolling(window=240).mean().reset_index(drop=True)


# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(minute_df, 'portfolio_data.minute_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over minute_quotes_analysis successfully completed")

