
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path

quandl.ApiConfig.api_key = 'TohCfidQJQeEsdD2o_N_'

############################################

#GET DATA

#############################################



def symbol_to_path(ticker, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(ticker)))

def get_data(ticker, begin_date):


   
    TimeSerie = quandl.get("WIKI/{}".format(ticker), start_date=begin_date)
    df = pd.DataFrame(data = TimeSerie)
    

    # ADD VERSION WITH YAHOO FIX

    return df

def data_run(ticker):
    # Define a date range
    # begin_date =input('format(2015-01-01):')
    begin_date='2015-01-01'
    # Choose stock ticker to read
    # ticker =input("enter the stock ticker?")
    # ticker=input('ticker:')
    # ticker='NVDA'
    # Get stock data
    df = get_data(ticker, begin_date)

    # print (df.ix['2017-04-18':'2017-04-25'])
    # print (df.head())

    return df, ticker


if __name__ == "__main__":
    # ticker = 'NVDA'
    data_run()

