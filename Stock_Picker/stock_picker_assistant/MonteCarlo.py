
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.stats import norm




############################################

#Monte-Carlo

#############################################



def monte_carlo(stock):


    #assign data and tiker

    # print(data['Adj. Close'].head())

    data = pd.read_csv(filepath_or_buffer='D:\Project\WebApp - Cours-Projet\Stock_Picker\stock_picker_assistant\data\{}.csv'.format(str(stock)))

    log_returns = np.log(1 + data['Adj. Close'].pct_change())

    # # data['Adj Close'].plot(figsize=(10,6))

    data=data['Adj. Close']
    # # # plt.show()

    # # log_returns.plot(figsize=(10,6))

    # # # plt.show()

    u = log_returns.mean()

    var = log_returns.var()

    # norm.ppf(0.95)

    x = np.random.rand(10,2)

    norm.ppf(x)

    drift = u - (0.5*var)


    stdev = log_returns.std()




    np.array(drift)

    z = norm.ppf(np.random.rand(10,2))

    t_intervals=70
    iterations = 10000

    daily_returns = np.exp(np.array(drift) + np.array(stdev) * norm.ppf(np.random.rand(t_intervals,iterations)))

    # print (daily_returns)

    s0= data.iloc[-1]

    # print(s0)

    price_list = np.zeros_like(daily_returns)

    price_list[0]=s0

    # print(price_list[0][1])

    for t in range(1,t_intervals):
        price_list[t] = price_list[t-1]*daily_returns[t]

        # if price_list[t] != 0 :

    seva=0

    for i in range (0, iterations):
        if price_list[t_intervals - 1 ][i] > s0:
           seva += 1

    prix_moyen = np.average(price_list[t_intervals - 1])
    obs_above_last_price = round ((seva/iterations) * 100 , 1)
    exp_return = round((prix_moyen/s0 -1)*100,1)

    # return obs_above_last_price, prix_moyen, t_intervals, exp_return

    # print('expected return:', u)
    # print('volatility:',stdev)

    # print ('% of observations above last price:',save)

    # print ('average price at the end of observations:', prix_moyen)

    # print ('% of expected returns in ',t_intervals,'days:',(prix_moyen/s0 -1)*100)

    # plt.figure(figsize=(10,6))
    # m_c_graph=plt.plot(price_list)

    # plt.show()

    return obs_above_last_price, prix_moyen, t_intervals, exp_return, price_list



if __name__ == "__main__":
        monte_carlo()
