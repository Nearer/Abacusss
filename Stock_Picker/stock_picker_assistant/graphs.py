import matplotlib as plt
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors, model_selection
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import datetime

quandl.ApiConfig.api_key = 'TohCfidQJQeEsdD2o_N_'



def graph_sma():
    df = pd.read_csv(
        filepath_or_buffer='D:\Project\WebApp - Cours-Projet\Stock_Picker\stock_picker_assistant\data\AAPL.csv')
    # Initialize the short and long windows
    short_window = 40
    long_window = 100

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    # Create short simple moving average over the short window
    signals['short_mavg'] = df['Adj. Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    signals['long_mavg'] = df['Adj. Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    #################################################################################################################
    #                                           Second Graph                                                        #
    #################################################################################################################
    # Set the initial capital
    initial_capital = float(10000.0)

    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=signals.index).fillna(0.0)

    # Buy a 100 shares
    positions['df'] = 100 * signals['signal']

    # Initialize the portfolio with value owned
    portfolio = positions.multiply(df['Adj. Close'], axis=0)

    # Store the difference in shares owned
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    portfolio['holdings'] = (positions.multiply(df['Adj. Close'], axis=0)).sum(axis=1)

    # Add `cash` to portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(df['Adj. Close'], axis=0)).sum(axis=1).cumsum()

    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()


    return df, signals, portfolio



def graph_machine_learning_l_r(stock):
    df = quandl.get("WIKI/{}".format(str(stock)))
    df_predict = df
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))



    df['label'] = df[forecast_col].shift(-forecast_out)



    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print(confidence)
    forecast_set = clf.predict(X_lately)





    # df_predict = quandl.get("WIKI/GOOGL")
    df_predict = df_predict[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    df_predict['Forecast'] = np.nan
    # print(df_predict)
    i = df_predict['Adj. Close'][-1]
    # last_date = df.iloc[-1].name

    last_date = df_predict.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    rng = pd.date_range('6/6/2017', periods=365, freq='D')

    df_p = pd.DataFrame(index=rng)

    df_p['Forecast']=np.nan

    df_p.loc[last_date] = [np.nan for _ in range(len(df_p.columns) - 1)] + [i]

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        # print(next_date)
        next_unix += 86400
        df_p.loc[next_date] = [np.nan for _ in range(len(df_p.columns)-1)]+[i]

    # print(df_p.tail())

    return df_predict, df_p, forecast_out


def graph_machine_learning_knn(stock):

    df = quandl.get("WIKI/{}".format(str(stock)))
    df_predict = df
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))

    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    df = df.astype(int)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    Accuracy = clf.score(X_test, y_test)

    forecast_set = clf.predict(X_lately)

    # df_predict = quandl.get("WIKI/GOOGL")
    df_predict = df_predict[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df_predict['Forecast'] = np.nan
    # print(df_predict)
    i = df_predict['Adj. Close'][-1]
    # last_date = df.iloc[-1].name

    last_date = df_predict.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    rng = pd.date_range('6/6/2017', periods=365, freq='D')

    df_p = pd.DataFrame(index=rng)

    df_p['Forecast'] = np.nan

    df_p.loc[last_date] = [np.nan for _ in range(len(df_p.columns) - 1)] + [i]

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        # print(next_date)
        next_unix += 86400
        df_p.loc[next_date] = [np.nan for _ in range(len(df_p.columns) - 1)] + [i]
    df_predict_2=df_predict
    df_p_2 = df_p
    forecast_out_2 = forecast_out

    return df_predict_2, df_p_2, forecast_out_2

