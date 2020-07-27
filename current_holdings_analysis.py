### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

### CAPTURE CURRENT HOLDINGS IN A LIST ###

my_current_holdings = ["VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC VHT FHLC IHI SMH XSD ARKK ARKW ARKF ARKQ BND AGG FNBGX"]

### DOWNLOAD DATA FROM YFINANCE ###

df = yf.download("VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC VHT FHLC IHI SMH XSD ARKK ARKW ARKF ARKQ BND AGG FNBGX", start="1990-01-01")
df = df.unstack().reset_index()
df.columns = ['col_name', 'Ticker', 'Date', 'Value']
df = df.pivot_table(values='Value', index=['Ticker', 'Date'],columns='col_name').reset_index()
df = df.sort_values(['Ticker', 'Date'])
df.columns.name = None

df[df['Ticker']=='FNBGX']

