import numpy as np
import pandas as pd
import os
# import pandas_datareader.data as web
import matplotlib.pyplot as plt

# list of stocks in portfolio
stocks = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'GOOGL']

# replace end-date with system date !!!!!!!!!!!!!!

dates = pd.date_range('1995-01-01', '2017-05-18')


# download daily price data for each of the stocks in the portfolio
# data = web.DataReader(stocks, data_source='yahoo', start='01/01/2010')['Adj Close']


def symbol_to_path(symbol, base_dir="D:\Project\WebApp - Cours-Projet\Stock_Picker\stock_picker_assistant\data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj. Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj. Close': symbol})
        df = df.join(df_temp)

    return df


def portfolio_optimizer(stocks, dates, input_stdev=False, input_return=False):
    data = get_data(stocks, dates)
    # convert daily stock prices into daily returns
    returns = data.pct_change()

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # set number of runs of random portfolio weights
    num_portfolios = 25000

    # set up array to hold results
    # We have increased the size of the array to hold the weight values for each stock
    results = np.zeros((4 + len(stocks) - 1, num_portfolios))

    for i in range(num_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(5))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # store results in results array
        results[0, i] = round(portfolio_return, 4)
        results[1, i] = round(portfolio_std_dev, 4)
        # store Sharpe Ratio (return / volatility) - risk free rate of 2.45%
        results[2, i] = round(results[0, i] / results[1, i] - 0.0245, 2)
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j + 3, i] = round(weights[j], 4)

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T,
                                 columns=['ret', 'stdev', 'sharpe', stocks[0], stocks[1], stocks[2], stocks[3],
                                          stocks[4]])

    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

    # locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]



    # assignation of values for sharp optimized portfolio
    s_return = max_sharpe_port[0]
    s_stdev = max_sharpe_port[1]
    s_sharp = max_sharpe_port[2]
    s_stock1 = max_sharpe_port[3]
    s_stock2 = max_sharpe_port[4]
    s_stock3 = max_sharpe_port[5]
    s_stock4 = max_sharpe_port[6]
    s_stock5 = max_sharpe_port[7]

    # assignation of values for least risky  portfolio

    r_return = min_vol_port[0]
    r_stdev = min_vol_port[1]
    r_sharp = min_vol_port[2]
    r_stock1 = min_vol_port[3]
    r_stock2 = min_vol_port[4]
    r_stock3 = min_vol_port[5]
    r_stock4 = min_vol_port[6]
    r_stock5 = min_vol_port[7]



    return s_return, s_stdev, s_sharp, s_stock1, s_stock2, s_stock3, s_stock4, s_stock5, r_return, r_stdev, r_sharp, \
           r_stock1, r_stock2, r_stock3, r_stock4, r_stock5, results_frame, max_sharpe_port, min_vol_port




    # print(max_sharpe_port)


    # print(min_vol_port)