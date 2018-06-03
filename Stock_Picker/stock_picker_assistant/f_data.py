
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path
quandl.ApiConfig.api_key = 'TohCfidQJQeEsdD2o_N_'





def symbol_to_path(tiker, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(tiker)))

def get_data(tiker, begin_date):


    """ !!!!add function to update csv files  """
    my_file = Path('data/{}.csv'.format(tiker))

    # if not my_file.is_file():

    TimeSerie = quandl.get("WIKI/{}".format(tiker), start_date=begin_date)
    df = pd.DataFrame(data = TimeSerie)
    df.to_csv('data/{}.csv'.format(tiker),index_label='Date')


    df =pd.read_csv(symbol_to_path(tiker), index_col='Date', na_values=['nan'], parse_dates=True)



    return df


def test_run():
    # Define a date range
    begin_date =input('format(2015-01-01):')

    # Choose stock tiker to read
    # tiker =input("enter the stock tiker?")
    tiker=input('tiker:')

    # Get stock data
    df = get_data(tiker,begin_date)

    # print (df.ix['2017-04-18':'2017-04-25'])
    print (df.head())

    return df , tiker


if __name__ == "__main__":
    test_run()







############################################

#SMA STRATEGY

#############################################
# try:
#     data=data['Adj. Close']
# except:
#     data=data['Adj Close']
#
#
# begin_date ='2015-01-01'
# finish_date='2017-01-01'
# dates= pd.date_range(begin_date,finish_date)
#
# df = pd.DataFrame(data)
#
#
# df['sma_short']=data.rolling(window=3).mean()
# df['sma_long']=data.rolling(window=20).mean()
#
#
#
# if df['Adj. Close'].iloc[-1] < df['sma_short'].iloc[-1]:
#     print(df.iloc[-1])
#     print('Short_Momentum:1')
#     df.plot()
#     plt.show()
#
# if df['Adj. Close'].iloc[-1] < df['sma_long'].iloc[-1]:
#     print(df.iloc[-1])
#     print('Long_Momentum:1')
#
#
#
# # for index, row in df.iterrows():
# #     if row['sma_short'] > row['Adj. Close']:
# #         print (index)
# #         print('{}: tendance croissance'.format(tiker))
# #     else:
# #         print(index)
# #         print('{}: tendance decroissance'.format(tiker))
#
#
#
# ############################################
#
# #VaR
#
# #############################################
#
#








