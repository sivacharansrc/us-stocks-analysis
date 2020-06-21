# https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
# https://joblib.readthedocs.io/en/latest/parallel.html
# https://queirozf.com/entries/parallel-for-loops-in-python-examples-with-joblib

import timeit

### READING (OR UPDATING INPUT FILE)
### CREATING LIST OF STOCK SYMBOLS OF INTEREST
stock_list = ["AMZN", "MSFT", "AAPL", "BAC", "WFC", "KO", "AXP", "JPM", "USB", "T", "COST", "EPD", "HON", "BAYRY"]


def retrive_stock_data(stock_list):
    if ((m_month == curr_month) & (m_year == curr_year)):
        stock_data = pd.read_csv(input_file)
    else:   
        ### Creating necessary variables for data creation
        stock_data = pd.DataFrame()
        start_date = str(datetime.today().year - 21) + "-01-01"
        end_date = str(datetime.today().year - 1) + "-12-31"

        for stocks in stock_list:
            stock_info = yf.Ticker(stocks)
            stock_info = stock_info.history(interval= "1mo", start=start_date, end=end_date).reset_index()
            stock_info['Stock'] = stocks
            
            # PERFORM FURTHER STEPS DEPENDING ON THE EXISTENSE OF STOCK_DATA
            if stock_data.shape[0] == 0:
                stock_data = stock_info
            else:
                stock_data = pd.concat([stock_data, stock_info])
        stock_data = stock_data[stock_data['Open'].notnull()]

        ### CREATING ADDITIONAL COLUMNS

        stock_data['Year'] = stock_data['Date'].dt.year
        stock_data['Month'] = stock_data['Date'].dt.month_name().str.slice(stop=3)
        stock_data = stock_data[stock_data['Date'].dt.day == 1]

        ### CREATE OR OVERWRITE THE STOCK DATA INPUT FILE
        stock_data.to_csv(input_file, index=False)
    return(stock_data)

start_time = datetime.now()
stock_data = retrive_stock_data(stock_list)
end_time = datetime.now()
end_time - start_time


import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()
inputs = stock_list

if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(retrive_stock_data)(i) for i in inputs)

processed_list.head()