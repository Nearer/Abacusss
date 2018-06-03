import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import neighbors, model_selection
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

quandl.ApiConfig.api_key = 'TohCfidQJQeEsdD2o_N_'

style.use('ggplot')

def machine_learning():

    df = quandl.get("WIKI/GOOGL")
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

    df = df.astype(int)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    Accuracy = clf.score(X_test, y_test)
    print(Accuracy)
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

    df_predict['Adj. Close'][-forecast_out:].plot()
    df_p['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

machine_learning()