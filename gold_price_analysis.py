### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from neuralprophet import NeuralProphet
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from forex_python.converter import CurrencyRates # https://forex-python.readthedocs.io/en/latest/usage.
import numpy as np

### OTHER REFERENCES ###
# https://towardsdatascience.com/neural-prophet-a-time-series-modeling-library-based-on-neural-networks-dd02dc8d868d
# https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b
# # https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events


### CAPTURE CURRENT HOLDINGS IN A LIST ###

current_local = os.getcwd()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # See if this can removed. Error occured when running the plt function. This is a workaround

ticker = "GC=F"

### PREPARING DATA FOR DAILY AVERAGE ###

start_date = str(int(datetime.today().strftime('%Y')) - 5) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 10 Years of data
gold_prices = yf.download(ticker, start=start_date, interval="1d")
gold_prices = gold_prices.reset_index()
col_names = ['date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
gold_prices.columns = col_names
gold_prices = gold_prices.sort_values(by=['date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True) #mergesort works better on pre-sorted items. For most other cases, quicksort works good

gold_prices = gold_prices[['date', 'adj_close']].reset_index(drop=True)


date_object = pd.date_range(start=start_date,end=datetime.today())
date_object = pd.DataFrame(date_object, columns=['date']) 
gold_prices = pd.merge(date_object, gold_prices, how="left", on="date").reset_index(drop=True)
gold_prices['adj_close'] = np.where(gold_prices['adj_close'].isna(), np.where(np.isnan(gold_prices['adj_close'].shift(1)), gold_prices['adj_close'].shift(-1), gold_prices['adj_close'].shift(1)), gold_prices['adj_close'])

gold_prices.rename(columns={"date":"ds", "adj_close":"y"}, inplace=True)

model = NeuralProphet(growth="linear",  # Determine trend types: 'linear', 'discontinuous', 'off'
                      changepoints=None, # list of dates that may include change points (None -> automatic )
                      n_changepoints=5,
                      changepoints_range=0.8,
                      trend_reg=0,
                      trend_reg_threshold=False,
                      yearly_seasonality="auto",
                      weekly_seasonality="auto",
                      daily_seasonality="auto",
                      seasonality_mode="additive",
                      seasonality_reg=0,
                      n_forecasts=1,
                      n_lags=0,
                      num_hidden_layers=0,
                      d_hidden=None,     # Dimension of hidden layers of AR-Net
                      ar_sparsity=None,  # Sparcity in the AR coefficients
                      learning_rate=None,
                      epochs=40,
                      loss_func="Huber",
                      normalize="auto",  # Type of normalization ('minmax', 'standardize', 'soft', 'off')
                      impute_missing=True,
                      log_level=None, # Determines the logging level of the logger object
)

metrics = model.fit(gold_prices, validate_each_epoch=True, freq="D") 
future = model.make_future_dataframe(gold_prices, periods=365, n_historic_predictions=len(gold_prices)) 
forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(14, 10))
model.plot(forecast, xlabel="Date", ylabel="Gold Price", ax=ax)
ax.set_title("Gold Price Predictions", fontsize=28, fontweight="bold")
plt.show()

forecast[(forecast.ds > '2021-01-05') & (forecast.ds < '2021-01-12')]

c = CurrencyRates()
c.get_rate('USD', 'INR')

# ### SPLITTING THE TRAIN AND THE TEST ####
# filter_date = str(int(datetime.today().strftime('%Y'))) + '-'  + datetime.today().strftime('%m') + '-' + '01'
# train = gold_prices[gold_prices['date'] < filter_date]
# test = gold_prices[gold_prices['date'] >= filter_date]

# train = train.reset_index().rename(columns={"date":"ds", "adj_close":"y"})
# test = test.reset_index().rename(columns={"date":"ds", "adj_close":"y"})

# model = Prophet(growth="linear", changepoints=None, 
#                 n_changepoints=25,
#                 seasonality_mode="multiplicative",
#                 yearly_seasonality="auto", 
#                 weekly_seasonality="auto", 
#                 daily_seasonality=False,
#                 holidays=None)

