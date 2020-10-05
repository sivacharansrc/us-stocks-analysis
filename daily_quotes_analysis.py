### INITIALIZING THE REQUIRED LIBRARIES
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import pandas_gbq


# Reference: https://www.datacamp.com/community/tutorials/moving-averages-in-pandas

### CAPTURE CURRENT HOLDINGS IN A LIST ###


current_local = os.getcwd()

if current_local[0:2] == 'C:':
    daily_data_holdings = "VOO VTI XITK IHI ARKG"
else:
    daily_data_holdings = "VTI FZROX FSKAX VOO IVV FXAIX FNILX VGT FTEC QQQ XITK VHT FHLC IHI XHE VYM SCHD FPBFX FIVFX SMH XSD ARKK ARKW ARKF ARKQ ARKG WCLD SKYY SLV GLDM IAU BND AGG FNBGX WFC TSLA FSCSX FSELX FSPHX FBIOX FFNOX AAPL"


### CATEGORIZING STOCKS

investment_account = ['VTI', 'FZROX', 'FSKAX', 'VOO', 'IVV', 'FXAIX', 'FNILX', 'VGT', 'FTEC', 'QQQ', 'XITK', 'VHT', 'FHLC', 'IHI', 'XHE', 'VYM', 'SCHD', 'FPBFX', 'FIVFX', 'SMH', 'XSD', 'ARKK', 'ARKW', 'ARKF', 'ARKQ', 'ARKG', 'WCLD', 'SKYY', 'SLV', 'GLDM', 'IAU', 'BND', 'AGG', 'FNBGX', 'WFC', 'TSLA', 'AAPL']
retirement_account = ['FZROX', 'FSKAX', 'FXAIX', 'FNILX', 'FSCSX', 'FSELX', 'FSPHX', 'FBIOX', 'FFNOX']
s_and_p_500 = ['FXAIX', 'VOO', 'IVV', 'FNILX']
total_market = ['VTI',  'FZROX', 'FSKAX', 'VOO']
technology = ['VGT', 'FTEC', 'XITK', 'FSCSX', 'QQQ']
semiconductors = ['SMH', 'XSD', 'FSELX']
dividends = ['VYM', 'SCHD', 'WFC']
health = ['VHT', 'FHLC', 'IHI', 'XHE', 'FSPHX', 'FBIOX']
high_growth = ['ARKK', 'ARKW', 'ARKF', 'ARKQ', 'ARKG', 'WCLD', 'SKYY', 'TSLA', 'AAPL']
international = ['FPBFX', 'FIVFX', 'FFNOX']
metals = ['SLV', 'GLDM', 'IAU']
bond = ['BND', 'AGG', 'FNBGX']

### PREPARING DATA FOR DAILY AVERAGE ###

start_date = str(int(datetime.today().strftime('%Y')) - 12) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')  # Pulling 10 Years of data
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

### APPLYING FILTERS TO THE DATASET

summary_date_start_filter = int(datetime.today().strftime('%Y')) - 10 # Filter 10 Years from CY-1
summary_date_end_filter = int(datetime.today().strftime('%Y'))-1 # CY-1
stock_data_analysis = stock_data_analysis[stock_data_analysis['date'].dt.year.isin(list(range(summary_date_start_filter, summary_date_end_filter+1)))]

stock_date_filter = str(int(datetime.today().strftime('%Y')) - 10) + '-'  + datetime.today().strftime('%m') + '-' + datetime.today().strftime('%d')
daily_df = daily_df[daily_df['date'] >= stock_date_filter]

price_prediction_data = daily_df[['ticker', 'date', 'adj_close', 'high', 'low']].copy().reset_index(drop=True)

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

stock_data_analysis['year'] = stock_data_analysis.date.dt.year
stock_data_analysis['month'] = stock_data_analysis['date'].dt.month_name().str.slice(stop=3)

# ## TOP MONTHS TO SELL THE STOCK

temp = stock_data_analysis[['high', 'ticker', 'year', 'month']].copy().reset_index(drop=True)
temp['mean_highs'] = temp.groupby(['ticker', 'year', 'month'])['high'].transform('mean').reset_index(drop=True)
temp['rank'] = temp.groupby(['ticker', 'year'])['mean_highs'].rank(method='dense', ascending=False)
temp = temp[temp['rank'] <= 3].sort_values(by=['ticker', 'year', 'rank'], ascending=False)
temp = temp[['ticker', 'year', 'month', 'rank']].drop_duplicates().reset_index(drop=True)
temp['month_count'] = temp.groupby(['ticker', 'month'])['year'].transform('count')
temp = temp.pivot_table(index=['ticker', 'month', 'rank', 'month_count'], aggfunc={'year': 'count'}).sort_values(by=['ticker', 'rank', 'year'], ascending=[True, True, False]).reset_index()
temp['rank1'] = temp.groupby(['ticker', 'rank'])['year'].rank(method='dense', ascending=False)
temp = temp[temp['rank1'] == 1].copy().reset_index(drop=True)
temp['months_to_sell'] = temp['month'] + " (" + temp['month_count'].astype(int).astype(str) + ", " + temp['year'].astype(int).astype(str) + ")"
months_to_sell = temp.groupby(['ticker'])['months_to_sell'].apply(', '.join).reset_index()


### TOP MONTHS TO BUY THE STOCKS
temp = stock_data_analysis[['low', 'ticker', 'year', 'month']].copy().reset_index(drop=True)
temp['mean_lows'] = temp.groupby(['ticker', 'year', 'month'])['low'].transform('mean').reset_index(drop=True)
temp['rank'] = temp.groupby(['ticker', 'year'])['mean_lows'].rank(method='dense', ascending=True)
temp = temp[temp['rank'] <= 3].sort_values(by=['ticker', 'year', 'rank'], ascending=True)
temp = temp[['ticker', 'year', 'month', 'rank']].drop_duplicates()
temp['month_count'] = temp.groupby(['ticker', 'month'])['year'].transform('count')
temp = temp.pivot_table(index=['ticker', 'month', 'rank', 'month_count'], aggfunc={'year': 'count'}).sort_values(by=['ticker', 'rank', 'year'], ascending=[True, True, False]).reset_index()
temp['rank1'] = temp.groupby(['ticker', 'rank'])['year'].rank(method='dense', ascending=False)
temp = temp[temp['rank1'] == 1].copy().reset_index(drop=True)
temp['months_to_buy'] = temp['month'] + " (" + temp['month_count'].astype(int).astype(str) + ", " + temp['year'].astype(int).astype(str) + ")"
months_to_buy = temp.groupby(['ticker'])['months_to_buy'].apply(', '.join).reset_index()


### GENERATE GROWTH METRICS FOR THE STOCKS
temp = stock_data_analysis[['adj_close', 'ticker', 'year', 'month', 'date']].loc[stock_data_analysis['month']=="Dec"].sort_values(by=['ticker', 'year']).reset_index().copy()
temp['max_dec'] = temp.groupby(['ticker', 'year']).date.transform('max').reset_index(drop=True)
temp = temp[temp['date']==temp['max_dec']][['ticker', 'year', 'month', 'adj_close']].copy()
temp['lagged_close'] = temp.groupby(by=['ticker'])['adj_close'].shift(1)
temp['yearly_growth'] = (temp['adj_close'] - temp['lagged_close']) / temp['lagged_close']
avg_yearly_growth = temp.groupby('ticker')['yearly_growth'].mean().reset_index()
temp['growth_probability'] = np.where(np.isnan(temp['yearly_growth']), np.NaN, np.where(temp['yearly_growth'] > 0,1,-1))
growth_tendency = (temp.groupby('ticker')['growth_probability'].sum() / (temp.groupby('ticker')['growth_probability'].count())).reset_index()
growth_metrics = pd.merge(avg_yearly_growth, growth_tendency, how="inner")

### CALCUALTE THE TOTAL HISTORICAL DATA POINTS
temp = stock_data_analysis[['ticker', 'date']].copy().reset_index(drop=True)
temp['max_date'] = temp.groupby('ticker')['date'].transform('max').reset_index(drop=True)
temp['min_date'] = temp.groupby('ticker')['date'].transform('min').reset_index(drop=True)
temp = temp[['ticker', 'max_date', 'min_date']].copy().drop_duplicates()
temp['no_of_months'] = ((temp.max_date - temp.min_date) / np.timedelta64(1, 'M')).round(decimals=0).astype('int')
total_data_points = temp[['ticker', 'no_of_months']].copy()

### SUBSETTING 1 YR DATA TO GENERATE 52 WEEK METRICS

temp = price_prediction_data.copy().reset_index(drop=True)
max_period = temp.date.max()
filter_period = np.where(max_period.year % 4 == 0, max_period - timedelta(days=366), max_period - timedelta(days=365))
temp = temp[temp.date >= filter_period]
temp['52_week_high'] = temp.groupby('ticker')['high'].transform('max')
temp['52_week_low'] = temp.groupby('ticker')['low'].transform('min')
temp['52_week_mean'] = temp.groupby('ticker')['adj_close'].transform('mean')
temp['52_week_median'] = temp.groupby('ticker')['adj_close'].transform('median')
fifty_two_week_metric = temp[['ticker', '52_week_high', '52_week_low', '52_week_mean', '52_week_median']].drop_duplicates().reset_index(drop=True)

### CODE FOR STOCK PRICE PREDICTION

price_prediction_data['year'] = price_prediction_data.date.dt.year
price_prediction_data['month'] = price_prediction_data['date'].dt.month
price_prediction_data['mon'] = price_prediction_data['date'].dt.month_name().str.slice(stop=3)

temp = price_prediction_data.copy().reset_index(drop=True)
temp['median_price'] = temp.groupby(['ticker', 'year', 'month'])['adj_close'].transform('median').reset_index(drop=True)
temp = temp[['ticker', 'year', 'month', 'mon', 'median_price']].drop_duplicates().sort_values(by=(['ticker', 'year', 'month'])).reset_index(drop=True)
temp2 = temp.copy() # To be used for calculating the actual price prediction
temp['cm_12'] = temp.groupby('ticker')['median_price'].shift(12)
temp['cm_9'] = temp.groupby('ticker')['median_price'].shift(9)
temp['cm_6'] = temp.groupby('ticker')['median_price'].shift(6)
temp['cm_3'] = temp.groupby('ticker')['median_price'].shift(3)

temp['cm_12_pct'] = (temp['median_price'] - temp['cm_12']) / temp['cm_12']
temp['cm_9_pct'] = (temp['median_price'] - temp['cm_9']) / temp['cm_9']
temp['cm_6_pct'] = (temp['median_price'] - temp['cm_6']) / temp['cm_6']
temp['cm_3_pct'] = (temp['median_price'] - temp['cm_3']) / temp['cm_3']

temp['change_12_months'] = temp.groupby(['ticker', 'month'])['cm_12_pct'].transform('mean')
temp['change_9_months'] = temp.groupby(['ticker', 'month'])['cm_9_pct'].transform('mean')
temp['change_6_months'] = temp.groupby(['ticker', 'month'])['cm_6_pct'].transform('mean')
temp['change_3_months'] = temp.groupby(['ticker', 'month'])['cm_3_pct'].transform('mean')

base_pct_change_data = temp[['ticker', 'month', 'mon', 'change_12_months',  'change_9_months',  'change_6_months',  'change_3_months']].copy().sort_values(by=(['ticker', 'month'])).drop_duplicates().reset_index(drop=True)

### CALCULATING THE PREDICTED VALUES FOR ACTUAL MONTH

## CALCULATING FOR THE CURRENT MONTH

current_year_month_filter = str(int(datetime.today().strftime('%Y'))) + str(int(datetime.today().strftime('%m')))
current_dt = date(int(datetime.today().strftime('%Y')), int(datetime.today().strftime('%m')), 1)

month1_filter = np.array([int((current_dt - relativedelta(months=3)).strftime('%m')), int((current_dt - relativedelta(months=6)).strftime('%m')), int((current_dt - relativedelta(months=9)).strftime('%m')), int((current_dt - relativedelta(months=12)).strftime('%m'))])
month2_filter = month1_filter + 1
month3_filter = month1_filter + 2

# FIXING THE WEIGHTS FOR THE MONTHS
wt_3 = 2
wt_6 = 1
wt_9 = 1
wt_12 = 0.75
wt_list = [wt_3, wt_6, wt_9, wt_12]

temp = temp2[(temp2.year >= summary_date_end_filter) & (temp2.year.astype(str) + temp2.month.astype(str) != current_year_month_filter) & (temp2.month.isin(month1_filter))].copy()
temp['date'] =  (temp['year'].astype(str) +  temp['month'].astype(str).str.pad(width=2, side='left', fillchar='0')).astype(int)
temp['rank'] = temp.groupby(['ticker'])['date'].rank(method='first', ascending=False).copy()
temp = temp[temp['rank'] <= 4]
temp['month_category'] = pd.cut(temp['rank'], bins=[0,1,2,3,4], include_lowest=True, labels=['month3', 'month6', 'month9', 'month12'])
temp = temp[['ticker', 'median_price', 'month_category']]
temp = temp.pivot_table(index='ticker', columns='month_category', aggfunc={'median_price':'mean'})
temp.columns = temp.columns.droplevel(0) # This is to remove the median from the column level
temp = pd.DataFrame(temp.to_records()) # This will remove any multilevel indexes, and also convert the index to column and reset index
temp['month'] = int(datetime.today().strftime('%m'))

current_month_price = pd.merge(temp, base_pct_change_data, how='inner', on=['ticker', 'month']).reset_index(drop=True)
current_month_price['change_3_months_value'] = (current_month_price['month3']+(current_month_price['month3']*current_month_price['change_3_months']))
current_month_price['change_6_months_value'] = (current_month_price['month6']+(current_month_price['month6']*current_month_price['change_6_months']))
current_month_price['change_9_months_value'] = (current_month_price['month9']+(current_month_price['month3']*current_month_price['change_9_months']))
current_month_price['change_12_months_value'] = (current_month_price['month12']+(current_month_price['month12']*current_month_price['change_12_months']))
data = np.array(current_month_price[['change_3_months_value', 'change_6_months_value', 'change_9_months_value', 'change_12_months_value']]) # USING NUMPY AS WEIGHTED AVERAGE IS NOT AVAILABLE WITH PANDAS
masked_data = np.ma.masked_array(data, np.isnan(data)) # USING A MASKED ARRAY TO OVERCOME ISSUES WITH NA
current_month_price['cm_target_price'] = np.ma.average(masked_data, axis=1, weights=wt_list).filled(np.nan)

current_month_price = current_month_price[['ticker', 'cm_target_price']].reset_index(drop=True)

### CALCULATIONS FOR CURRENT MONTH PLUS ONE

temp = temp2[(temp2.year >= summary_date_end_filter) & (temp2.year.astype(str) + temp2.month.astype(str) != current_year_month_filter) & (temp2.month.isin(month2_filter))].copy()
temp['date'] =  (temp['year'].astype(str) +  temp['month'].astype(str).str.pad(width=2, side='left', fillchar='0')).astype(int)
temp['rank'] = temp.groupby(['ticker'])['date'].rank(method='first', ascending=False).copy()
temp = temp[temp['rank'] <= 4]
temp['month_category'] = pd.cut(temp['rank'], bins=[0,1,2,3,4], include_lowest=True, labels=['month3', 'month6', 'month9', 'month12'])
temp = temp[['ticker', 'median_price', 'month_category']]
temp = temp.pivot_table(index='ticker', columns='month_category', aggfunc={'median_price':'mean'})
temp.columns = temp.columns.droplevel(0) # This is to remove the median from the column level
temp = pd.DataFrame(temp.to_records()) # This will remove any multilevel indexes, and also convert the index to column and reset index
temp['month'] = int((datetime.today()+relativedelta(months=+1)).strftime('%m'))

current_month_one_price = pd.merge(temp, base_pct_change_data, how='inner', on=['ticker', 'month']).reset_index(drop=True)
current_month_one_price['change_3_months_value'] = (current_month_one_price['month3']+(current_month_one_price['month3']*current_month_one_price['change_3_months']))
current_month_one_price['change_6_months_value'] = (current_month_one_price['month6']+(current_month_one_price['month6']*current_month_one_price['change_6_months']))
current_month_one_price['change_9_months_value'] = (current_month_one_price['month9']+(current_month_one_price['month3']*current_month_one_price['change_9_months']))
current_month_one_price['change_12_months_value'] = (current_month_one_price['month12']+(current_month_one_price['month12']*current_month_one_price['change_12_months']))
data = np.array(current_month_one_price[['change_3_months_value', 'change_6_months_value', 'change_9_months_value', 'change_12_months_value']]) # USING NUMPY AS WEIGHTED AVERAGE IS NOT AVAILABLE WITH PANDAS
masked_data = np.ma.masked_array(data, np.isnan(data)) # USING A MASKED ARRAY TO OVERCOME ISSUES WITH NA
current_month_one_price['cm_plus_one_target_price'] = np.ma.average(masked_data, axis=1, weights=wt_list).filled(np.nan)

current_month_one_price = current_month_one_price[['ticker', 'cm_plus_one_target_price']].reset_index(drop=True)


### CALCULATIONS FOR CURRENT MONTH PLUS TWO

temp = temp2[(temp2.year >= summary_date_end_filter) & (temp2.year.astype(str) + temp2.month.astype(str) != current_year_month_filter) & (temp2.month.isin(month3_filter))].copy()
temp['date'] =  (temp['year'].astype(str) +  temp['month'].astype(str).str.pad(width=2, side='left', fillchar='0')).astype(int)
temp['rank'] = temp.groupby(['ticker'])['date'].rank(method='first', ascending=False).copy()
temp = temp[temp['rank'] <= 4]
temp['month_category'] = pd.cut(temp['rank'], bins=[0,1,2,3,4], include_lowest=True, labels=['month3', 'month6', 'month9', 'month12'])
temp = temp[['ticker', 'median_price', 'month_category']]
temp = temp.pivot_table(index='ticker', columns='month_category', aggfunc={'median_price':'mean'})
temp.columns = temp.columns.droplevel(0) # This is to remove the median from the column level
temp = pd.DataFrame(temp.to_records()) # This will remove any multilevel indexes, and also convert the index to column and reset index
temp['month'] = int((datetime.today()+relativedelta(months=+2)).strftime('%m'))

current_month_two_price = pd.merge(temp, base_pct_change_data, how='inner', on=['ticker', 'month']).reset_index(drop=True)
current_month_two_price['change_3_months_value'] = (current_month_two_price['month3']+(current_month_two_price['month3']*current_month_two_price['change_3_months']))
current_month_two_price['change_6_months_value'] = (current_month_two_price['month6']+(current_month_two_price['month6']*current_month_two_price['change_6_months']))
current_month_two_price['change_9_months_value'] = (current_month_two_price['month9']+(current_month_two_price['month3']*current_month_two_price['change_9_months']))
current_month_two_price['change_12_months_value'] = (current_month_two_price['month12']+(current_month_two_price['month12']*current_month_two_price['change_12_months']))
data = np.array(current_month_two_price[['change_3_months_value', 'change_6_months_value', 'change_9_months_value', 'change_12_months_value']]) # USING NUMPY AS WEIGHTED AVERAGE IS NOT AVAILABLE WITH PANDAS
masked_data = np.ma.masked_array(data, np.isnan(data)) # USING A MASKED ARRAY TO OVERCOME ISSUES WITH NA
current_month_two_price['cm_plus_two_target_price'] = np.ma.average(masked_data, axis=1, weights=wt_list).filled(np.nan)

current_month_two_price = current_month_two_price[['ticker', 'cm_plus_two_target_price']].reset_index(drop=True)

### MERGING ALL THE PREDICTED PRICE DATASET

price_predictions = pd.merge(current_month_price, current_month_one_price, how='left', on='ticker')
price_predictions = pd.merge(price_predictions, current_month_two_price, how='left', on='ticker')

### MERGING ALL SUMMARY DATASET

summary_data = stock_data_analysis[['ticker', 'account', 'sector']].drop_duplicates().copy()
summary_data = pd.merge(summary_data, total_data_points, on='ticker', how='inner')
summary_data = pd.merge(summary_data, growth_metrics, on='ticker', how='inner')
summary_data = pd.merge(summary_data, months_to_sell, on='ticker', how='inner')
summary_data = pd.merge(summary_data, months_to_buy, on='ticker', how='inner')
summary_data = pd.merge(summary_data, fifty_two_week_metric, on='ticker', how='inner')
summary_data = pd.merge(summary_data, price_predictions, on='ticker', how='left')
# summary_data['target_price'] = ((summary_data['52_week_high'] * 1) + (summary_data['52_week_low'] * 1) + (summary_data['52_week_mean'] * 2) + summary_data['52_week_median'] * 2.5) / 6.5

### WRITING DATA TO LOCAL DRIVE

if current_local[0:2] == 'C:':
    summary_data.to_csv(current_local + "\\us-stocks-analysis\\input\\summary_data_new.csv", index=False)
else:
    summary_data.to_csv("~/my-portfolio-analysis/input-files/summary_data.csv", index=False)

print("Summary data file successfully exported to local drive")

# WRITING PANDAS DATAFRAME TO BIGQUERY DATASET

pandas_gbq.to_gbq(daily_df, 'portfolio_data.daily_quotes_analysis', project_id= 'my-portfolio-analysis', if_exists='replace')

# PRINTING SUCCESSFUL CODE EXECUTION MESSAGE
print("Writing to BigQuery over daily_quotes_analysis successfully completed")
