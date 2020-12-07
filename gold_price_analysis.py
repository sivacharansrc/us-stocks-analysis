### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import pandas_gbq
import pyarrow
from fbprophet import Prophet

### OTHER REFERENCES ###
# https://towardsdatascience.com/neural-prophet-a-time-series-modeling-library-based-on-neural-networks-dd02dc8d868d
# https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b
# # https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events


### CAPTURE CURRENT HOLDINGS IN A LIST ###

current_local = os.getcwd()
ticker = "GC=F"

### PREPARING DATA FOR DAILY AVERAGE ###

start_date = str(int(datetime.today().strftime('%Y')) - 12) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 10 Years of data
gold_prices = yf.download(daily_data_holdings, start=start_date, interval="1d")
gold_prices = gold_prices.reset_index()
col_names = ['date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
gold_prices.columns = col_names
gold_prices = gold_prices.sort_values(by=['date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True) #mergesort works better on pre-sorted items. For most other cases, quicksort works good


date_object = pd.date_range(start=start_date,end=datetime.today())
date_object = pd.DataFrame(date_object, columns=['date']) 
gold_prices = gold_prices[['date', 'adj_close']]
gold_prices = pd.merge(date_object, gold_prices, how="left", on="date").reset_index(drop=True)
gold_prices['adj_close'] = np.where(gold_prices['adj_close'].isna(), np.where(np.isnan(gold_prices['adj_close'].shift(1)), gold_prices['adj_close'].shift(2), gold_prices['adj_close'].shift(1)), gold_prices['adj_close'])

### SPLITTING THE TRAIN AND THE TEST ####
filter_date = str(int(datetime.today().strftime('%Y'))) + '-'  + datetime.today().strftime('%m') + '-' + '01'
train = gold_prices[gold_prices['date'] < filter_date]
test = gold_prices[gold_prices['date'] >= filter_date]

train = train.reset_index().rename(columns={"date":"ds", "adj_close":"y"})
test = test.reset_index().rename(columns={"date":"ds", "adj_close":"y"})

model = Prophet(growth="linear", changepoints=None, 
                n_changepoints=25,
                seasonality_mode="multiplicative",
                yearly_seasonality="auto", 
                weekly_seasonality="auto", 
                daily_seasonality=False,
                holidays=None)

