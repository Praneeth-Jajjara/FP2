import pandas as pd
import numpy as np
import time
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import warnings
warnings.filterwarnings("ignore")
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
#!pip install autots
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("yfinance")
install("yahoofinancials")
import yfinance as yf
from yahoofinancials import YahooFinancials
install("autots")
from autots import AutoTS, load_daily
from os import path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import numpy as np
import time
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import warnings
warnings.filterwarnings("ignore")
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import seaborn as sns
import matplotlib.pyplot as plt
#!pip install autots
from autots import AutoTS, load_daily

"""Initiating chrome web driver """
tab_titles=['Fetch Data','Target Variable', 'Plot','Understanding Cummulative Return','Peformance','Prediction']
tabs=st.tabs(tab_titles)

    
class WebDriver(object):

    def __init__(self):
        self.options = Options()

        self.options.binary_location = '/opt/headless-chromium'
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--start-maximized')
        self.options.add_argument('--start-fullscreen')
        self.options.add_argument('--single-process')
        self.options.add_argument('--disable-dev-shm-usage')

    def get(self):
        driver = webdriver.Chrome(ChromeDriverManager().install())
        return driver

"""Ticker identification

It typically refers to the process of identifying the unique symbol or code used to represent a publicly traded company or security on a stock exchange
"""

def get_tickers(driver):
    """return the number of tickers available on the webpage"""
    TABLE_CLASS = "W(100%)"  
    tablerows = len(driver.find_elements(By.XPATH, value="//table[@class= '{}']/tbody/tr".format(TABLE_CLASS)))
    return tablerows

"""Once we have identified the ticker symbol for a specific company or security, there are various types of financial and market data that you are to be retrieved"""

def parse_ticker(rownum, table_driver):
    """Parsing each Ticker row by row and return the data in the form of Python dictionary"""
    Symbol = table_driver.find_element(By.XPATH, value="//tr[{}]/td[1]".format(rownum)).text
    Name = table_driver.find_element(By.XPATH, value="//tr[{}]/td[2]".format(rownum)).text
    LastPrice = table_driver.find_element(By.XPATH, value="//tr[{}]/td[3]".format(rownum)).text
    MarketTime = table_driver.find_element(By.XPATH, value="//tr[{}]/td[4]".format(rownum)).text
    Change = table_driver.find_element(By.XPATH, value="//tr[{}]/td[5]".format(rownum)).text
    PercentChange = table_driver.find_element(By.XPATH, value="//tr[{}]/td[6]".format(rownum)).text	
    Volume = table_driver.find_element(By.XPATH, value="//tr[{}]/td[7]".format(rownum)).text
    MarketCap = table_driver.find_element(By.XPATH, value="//tr[{}]/td[8]".format(rownum)).text	

    return {
    'Symbol': Symbol,
    'Name': Name,
    'LastPrice': LastPrice,
    'MarketTime': MarketTime,
    'Change': Change,
    'PercentChange': PercentChange,
    'Volume': Volume,
    'MarketCap': MarketCap
    }

"""By using a personalized screener we select stocks from the NIFTY 50 index

NIFTY 50 is an index of the top 50 companies listed on the National Stock Exchange (NSE) of India, and it includes some of the largest and most well-known companies in the country across a range of industries.
"""

YAHOO_FINANCE_URL = "https://finance.yahoo.com/screener/unsaved/d0ac6574-dc65-4bec-a860-428098c86c2c?offset=0&count=100" 

instance_ = WebDriver()
driver = instance_.get()
driver.get(YAHOO_FINANCE_URL)
print('Fetching the page')
table_rows = get_tickers(driver)
print('Found {} Tickers'.format(table_rows))
print('Parsing Trending tickers')
table_data = [parse_ticker(i, driver) for i in range (1, table_rows + 1)]
driver.close()
driver.quit()

#table_data

"""Removing duplicates from the data set"""

type(table_data)
table_data_df=pd.DataFrame(table_data)
table_data_df =table_data_df.drop_duplicates("Name")
#table_data_df
with tabs[0]:
    if st.button('Fetch Data From Yahoo Finance!'):
        st.write('Scraping trending 50 stocks from the website')
        st.table(table_data_df)

table_data_df.dtypes

"""Using change as a factor for shortlisting stocks is a common approach, as it can provide insight into how the stock has performed over a specific period of time. However, there are many other methods and factors that can be used to select stocks, depending on your investment objectives and risk tolerance."""

table_data_df['Change']=table_data_df['Change'].str.rstrip("%")
#table_data_df

table_data_df = table_data_df.astype({'Change':'float'})

#table_data_df.dtypes

"""Filter the best performer of current day stock"""

#table_data_df
names=table_data_df[table_data_df["Change"]==table_data_df["Change"].max()]["Name"] 
Symbol=table_data_df[table_data_df["Change"]==table_data_df["Change"].max()]["Symbol"]

#Symbol

""" Using selected ticker for long term and short term"""

yahoo_financials = YahooFinancials('DIVISLAB.BO')
data=yahoo_financials.get_historical_price_data("2022-06-10", "2023-03-25", "daily")
btc_df = pd.DataFrame(data['DIVISLAB.BO']['prices'])
btc_df = btc_df.drop('date', axis=1).set_index('formatted_date')
#btc_df
btc_df.dropna(how='any',inplace=True)

"""Analysing change in daily basis"""

change= btc_df['adjclose'].pct_change()
change.plot(title=" stock price")

#len(btc_df)

"""Formating series to time series"""

df_s = btc_df[[ 'adjclose']]

df_s.index = pd.to_datetime(df_s.index)
df_s = df_s.sort_values('formatted_date')
#df_s
with tabs[1]:
    st.table(df_s)

"""Train data"""

train_df_s = df_s.iloc[:172]
train_df_s

"""Test data"""

test_df_s = df_s.iloc[172:]
test_df_s
with tabs[2]:
    plt.title('Plotting train and test data', size=20)
    train_df_s.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
    test_df_s.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
    plt.legend()
    plt.grid()
    plt.show()
#""" Plotting train and test data"""

# plt.title('Divis Lab', size=20)
# train_df_s.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
# test_df_s.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
# plt.legend()
# plt.grid()
# plt.show()

"""Setting AutoTS for varius hyder parameters, basic are forecast time - 5 days """

model = AutoTS(
    forecast_length=5,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)

#selecting format horizontal or vertical data
long = False

"""Model fitting"""

model = model.fit(
    train_df_s,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

"""Model prediction"""

prediction = model.predict()
#plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2023-02-16")

"""Print the details of the best model"""

print(model)

"""Selecting stock for long Term Predection """

yahoo_financials = YahooFinancials('DIVISLAB.BO')
data=yahoo_financials.get_historical_price_data('2012-01-01','2022-01-01',"daily")
btc_df_l = pd.DataFrame(data['DIVISLAB.BO']['prices'])
btc_df_l = btc_df_l.drop('date', axis=1).set_index('formatted_date')
btc_df_l= btc_df_l[['adjclose']]

btc_df_l['Stock_Returns']=btc_df_l['adjclose'].pct_change()

btc_df_l['Stock_cumRETURNS']=btc_df_l['Stock_Returns'].cumsum().apply(np.exp)

btc_df_l.dropna(how='any',inplace=True)

"""Understanding cumulative return of stock for period considered """

# btc_df_l
# sns.set_style('whitegrid')
# btc_df_l['Stock_cumRETURNS'].plot(figsize=(8,8),label="Stock")
# plt.title('Equity Curves')
# plt.ylabel("Cumulative Returns")
# plt.xlabel("Index")
# plt.legend(loc='upper left')
# plt.show()
with tabs[3]:
    st.write('Understanding cumulative return of stock for period considered ')
    btc_df_l
    sns.set_style('whitegrid')
    btc_df_l['Stock_cumRETURNS'].plot(figsize=(8,8),label="Stock")
    plt.title('Equity Curves')
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Index")
    plt.legend(loc='upper left')
    plt.show()
"""Converting series to Time series """

df = btc_df_l[[ 'adjclose']]
df = df.sort_values('formatted_date')
df.index = pd.to_datetime(df.index)


"""Selecting train and test data"""

train_df = df.iloc[:2218]
test_df = df.iloc[2218:]


# plt.title('DIVIS', size=20)
# train_df.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
# test_df.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
# plt.legend()
# plt.grid()
# plt.show()
with tabs[4]:
    st.write('Selecting train and test data')
    #plt.title('DIVIS', size=20)
    train_df.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
    test_df.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
    plt.legend()
    plt.grid()
    plt.show()

"""Setting out various parameters for predection using AutoTS , setting out duration 2 years """

model = AutoTS(
    forecast_length=246,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)

long = False

model = model.fit(
    train_df,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
)

prediction = model.predict()

#plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2021-05-30")

print(model)

model.plot_per_series_error()
plt.show()

model.plot_generation_loss()
plt.show()

# point forecasts dataframe
forecasts_df = prediction.forecast
# upper and lower forecasts
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

"""Forecasting"""

forecasts_df

# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")

with tabs[5]:
    st.write('Forecast Results:')
    st.table(forecasts_df)
    model_results = model.results()
    # and aggregated from cross validation
    validation_results = model.results("validation")
    st.write('Model Results:')
    st.table(model_results)
    st.write('Validation Results:')
    st.table(validation_results)

import autots
import pickle

# Create an AutoTS model
#model = autots.AutoTS()

# Fit the model to data
#model.fit(model)

# # Save the model as a pickle file
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

"""DEPLOYEMENT"""

import streamlit as st
import pandas as pd
import joblib

# Load the model
#model = joblib.load(r"C:\Users\Praneeth\Downloads\model.pkl")

#st.title('Stock Prediction')

st.write('Enter the stock symbol and date range to make a prediction')
symbol = st.text_input( 'DIVISLAB.BO')
start_date = st.date_input('2012-01-01')
end_date = st.date_input('2022-01-01')

# Define the prediction function
def predict(symbol, start_date, end_date):
    # Load the stock data
    df = pd.read_csv(f'{symbol}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df.loc[mask]
    
    # Make the prediction
    X = df.drop(['Date', 'Close'], axis=1)
    pred = model.predict(X)
    
    # Return the prediction
    return pred[0]

# Call the prediction function and display the result
with tabs[6]:
    if st.button('Predict'):
        result = predict(symbol, start_date, end_date)
        st.write(f'The predicted closing price is {result:.2f}')