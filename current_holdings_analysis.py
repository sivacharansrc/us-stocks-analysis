### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_gbq
import os

### CAPTURE CURRENT HOLDINGS IN A LIST ###

my_current_holdings = ["VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC XITK VHT FHLC IHI XHE SMH XSD ARKK ARKW ARKF ARKQ ARKG SLV GLDM FPBFX FIVFX BND AGG FNBGX"]

### DOWNLOAD DATA FROM YFINANCE ###

start_date = str(int(datetime.today().strftime('%Y')) - 10)  + '-01-01' # Pulling 10 Years of data

df = yf.download("VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC VHT FHLC IHI SMH XSD ARKK ARKW ARKF ARKQ BND AGG FNBGX", start=start_date)
df = df.unstack().reset_index()
df.columns = ['col_name', 'Ticker', 'Date', 'Value']
df = df.pivot_table(values='Value', index=['Ticker', 'Date'],columns='col_name').reset_index()
df = df.sort_values(['Ticker', 'Date'])
df.columns.name = None
col_names = ['ticker', 'date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
df.columns = col_names


df.to_csv(r"/home/sivacharansrc/my-portfolio-analysis/output/my_portfolio_10_yr_historical_quotes.csv", index=False)
# Writing pandas dataframe to BigQuery DataSet

# GENERATING THE PATH OF THE INPUT STOCK DATA FILE

pandas_gbq.to_gbq(df, 'portfolio_data.holdings_historical_quotes', project_id= 'my-portfolio-analysis', if_exists='replace')