### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from neuralprophet import NeuralProphet
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from forex_python.converter import CurrencyRates # https://forex-python.readthedocs.io/en/latest/usage.
import numpy as np

### OTHER REFERENCES ###
# https://towardsdatascience.com/neural-prophet-a-time-series-modeling-library-based-on-neural-networks-dd02dc8d868d
# https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b
# # https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events
# https://forex-python.readthedocs.io/en/latest/usage.html


### CAPTURE CURRENT HOLDINGS IN A LIST ###

current_local = os.getcwd()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # See if this can removed. Error occured when running the plt function. This is a workaround

ticker = "GC=F"

### PREPARING DATA FOR DAILY AVERAGE ###

if os.path.exists("./us-stocks-analysis/input/exchange_rate.csv"):
    # Read local file if already exists
    exchange_price_history = pd.read_csv("./us-stocks-analysis/input/exchange_rate.csv")
    exchange_price_history['date'] = pd.to_datetime(exchange_price_history.date, format="%m/%d/%Y")
    start_date = date.today() - relativedelta(years=10)
    exchange_price_history = exchange_price_history[exchange_price_history.date > start_date]
    
    # Updating the exchange rate until the current date
    max_date = exchange_price_history.date.max() 
    
    if max_date < datetime.today():
        start_date = max_date + timedelta(days=1)
        end_date = date.today()
        date_object = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame(date_object, columns=['date'])
        
        # Getting the exchange rates for new dates
        c = CurrencyRates()
        df['exchange_rate'] = df['date'].apply(lambda x: c.get_rate('USD', 'INR', x))
        exchange_price_history = pd.concat([exchange_price_history, df]).reset_index(drop=True)
        exchange_price_history.to_csv("./us-stocks-analysis/input/exchange_rate.csv", index=False)
        
        # Merging the existing dataset with the new dates and saving to local drive
else:
    start_date = str(int(datetime.today().strftime('%Y')) - 10) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 10 Years of data
    date_object = pd.date_range(start=start_date,end=datetime.today())
    exchange_price_history = pd.DataFrame(date_object, columns=['date'])
    c=CurrencyRates()
    exchange_price_history['exchange_rate'] = exchange_price_history['date'].apply(lambda x: c.get_rate('USD', 'INR', x))
    exchange_price_history.to_csv("./us-stocks-analysis/input/exchange_rate.csv", index=False)

exchange_price_history.tail()   

start_date = date.today() - relativedelta(years=10)
gold_prices = yf.download(ticker, start=start_date, interval="1d")
gold_prices = gold_prices.reset_index()
col_names = ['date', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
gold_prices.columns = col_names
gold_prices = gold_prices.sort_values(by=['date'], axis=0, ascending=True, kind='mergesort').reset_index(drop=True) #mergesort works better on pre-sorted items. For most other cases, quicksort works good
gold_prices = gold_prices[['date', 'adj_close']].reset_index(drop=True)

gold_prices.to_csv("./us-stocks-analysis/input/gold_price_history.csv", index=False)

gold_prices.adj_close.interpolate(method='nearest', inplace=True) # Fill missing values using the nearest value in either direction
gold_prices['adj_close'] = gold_prices['adj_close'] / 28.3495 # Converting cost of 1 oz to gms 

# Merging gold dataset with exchange rates for INR conversion

df = pd.merge(gold_prices, exchange_price_history, how='left', on='date').reset_index(drop=True)
df.isnull().sum()
df.head()
df.exchange_rate.interpolate(method='nearest', inplace=True) # Ideally no interpolate should be used. Check the occurence of NA
df['y'] = df.adj_close * df.exchange_rate
df.rename(columns={"date":"ds"}, inplace=True)

df = df[['ds', 'y']]
model = NeuralProphet()
 


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

metrics = model.fit(df, validate_each_epoch=True, freq="D") 
future = model.make_future_dataframe(df, periods=365, n_historic_predictions=len(df)) 
forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(14, 10))
model.plot(forecast, xlabel="Date", ylabel="Gold Price", ax=ax)
ax.set_title("Gold Price Predictions", fontsize=28, fontweight="bold")
plt.show()

forecast[(forecast.ds > '2021-01-07') & (forecast.ds < '2021-01-12')]


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

