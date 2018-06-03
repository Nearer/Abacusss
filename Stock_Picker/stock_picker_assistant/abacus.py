from stock_picker_assistant import GetData, MonteCarlo
# import SMA_Datacenter

# import MachineLearningRegression

def abacus_function(stock_ticker):


    #store df in dictionary
    stocks_df = {}

    #get data for each ticker
    # for ticker in default_tickers:

        # df, ticker = GetData.data_run(ticker)

        # stocks_df[ticker] = df


    #execute modules:m_c,sma,m_l

    # st_monte_carlo='Monte-Carlo'
    # st_sma='Simple Moving Average(Momentum)'
    # st_m_l='Machine Learning'

    #create function that takes into account 3 strategy to class stocks

    # for ticker in stocks_df:
        # print('***************************************{}********************************{}***************************'.format(ticker,st_monte_carlo))
        #Monte-Carlo
    obs_above_last_price, prix_moyen, t_intervals, exp_return, price_list = MonteCarlo.monte_carlo(stock_ticker)

        # print('***************************************{}**************************{}***************************'.format(ticker,st_sma))
        # #SMA
        # SMA_Datacenter.sma(stocks_df[ticker])

        # # print('***************************************{}**************************{}***************************'.format(ticker, st_m_l))
        # # #M-L
        # MachineLearningRegression.machine_learning(df=stocks_df[ticker], df_predict=stocks_df[ticker])

    #plot stock according to stdev and returns

    # return obs_above_last_price, prix_moyen, t_intervals, exp_return, price_list

    return obs_above_last_price, prix_moyen, t_intervals, exp_return, price_list


