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


def custom_portfolio_optimizer(stocks, dates, input_stdev=0, input_return=0):
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

# locate position of portfolio with define volatility and highest return



    if input_stdev == '':
        input_stdev == input_stdev

    else:

        # results_frame_sort = results_frame.loc[results_frame['stdev'] == float(input_stdev)]

        if results_frame.loc[results_frame['stdev'] == float(input_stdev)].empty:

            results_frame_sort = results_frame.ix[(results_frame['stdev'] - float(input_stdev)).abs().argsort()[:3]]
            optimum_portfolio = results_frame_sort.loc[results_frame_sort['ret'].idxmax()]

        else:
            results_frame_sort = results_frame.loc[results_frame['stdev'] == float(input_stdev)]
            optimum_portfolio = results_frame_sort.loc[results_frame_sort['ret'].idxmax()]

        #print(optimum_portfolio)

# locate position of portfolio with define return and lowest volatility


    if input_return == '':
        input_return == input_return
    else:



        if results_frame.loc[results_frame['ret'] == float(input_return)].empty:

            results_frame_sort = results_frame.ix[(results_frame['ret'] - float(input_return)).abs().argsort()[:3]]
            optimum_portfolio = results_frame_sort.loc[results_frame_sort['stdev'].idxmin()]

        else:
            results_frame_sort = results_frame.loc[results_frame['ret'] == float(input_return)]
            optimum_portfolio = results_frame_sort.loc[results_frame_sort['stdev'].idxmin()]

            #print(optimum_portfolio)



    # assignation of values for custom portfolio
    # if optimum_portfolio.empty:
    c_return = optimum_portfolio[0]
    c_stdev = optimum_portfolio[1]
    c_sharp = optimum_portfolio[2]
    c_stock1 = optimum_portfolio[3]
    c_stock2 = optimum_portfolio[4]
    c_stock3 = optimum_portfolio[5]
    c_stock4 = optimum_portfolio[6]
    c_stock5 = optimum_portfolio[7]

    # else:
    #     c_return = False
    #     c_stdev = False
    #     c_sharp = False
    #     c_stock1 = False
    #     c_stock2 = False
    #     c_stock3 = False
    #     c_stock4 = False
    #     c_stock5 = False

    return  c_return, c_stdev, c_sharp, c_stock1, c_stock2,c_stock3, c_stock4, c_stock5


# custom_portfolio_optimizer(stocks,dates,input_return=False,input_stdev=0.23)

    # print(max_sharpe_port)


    # print(min_vol_port)