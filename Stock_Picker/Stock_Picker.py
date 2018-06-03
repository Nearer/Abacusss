from flask import Flask, render_template, request, redirect, jsonify, url_for, make_response
from stock_picker_assistant import abacus, graphs, p_optimizer,custom_optimizer
from pygal.style import DarkSolarizedStyle
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import numpy as np
from pprint import pprint
import math
import pygal

app = Flask(__name__)





@app.route('/')
@app.route('/abacus/')
def ChooseStocks():
    return render_template('abacus.html')


stock = ''
@app.route('/abacus/stock/', methods = ['POST', 'GET'])
@app.route('/abacus/stock/<string:stock_ticker>', methods=['POST', 'GET'])
def stockAnalysis(stock_ticker=None):
    if request.method == 'POST':
        stock_ticker = request.form['stock']
    global stock
    stock = stock_ticker

    obs_above_last_price, prix_moyen, t_intervals, exp_return, price_list = abacus.abacus_function(stock_ticker)
    prix_moyen = math.floor(prix_moyen)

    # Monte-Carlo Graph
    try:
        line_chart = pygal.Line(style=DarkSolarizedStyle)
        line_chart.title = 'Monte-Carlo Graph'
        line_chart.x_labels = map(str, range(0, t_intervals))

        for y in range(0,10):
            my_list = []
            for i in range(0, t_intervals):

                my_list.append(price_list[i][y])
            line_chart.add('{}'.format(y), my_list)

        graph_data = line_chart.render_data_uri()
        return render_template('test.html', graph_data=graph_data, stock_ticker=stock_ticker, obs_above_last_price=obs_above_last_price, prix_moyen=prix_moyen, exp_return=exp_return
                               )
    except Exception as e:
        return (str(e))



import random
from io import StringIO, BytesIO

from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure




@app.route('/abacus/optimizer/')
def optimizer():

    return render_template('optimizer.html')



@app.route('/abacus/optimizer/portfolio', methods = ['POST'])
def portfolio_optimizer():

    if request.method == 'POST':
        stocks = request.form['stock']

    stocks=stocks.split()
    dates = pd.date_range('1995-01-01', '2017-05-18')



    s_return, s_stdev, s_sharp, s_stock1, s_stock2, s_stock3, s_stock4, s_stock5, r_return, r_stdev, r_sharp, r_stock1, r_stock2, r_stock3, r_stock4, r_stock5, results_frame, max_sharpe_port, min_vol_port = p_optimizer.portfolio_optimizer(stocks, dates)

    global results_frame_g
    global max_sharpe_port_g
    global min_vol_port_g
    results_frame_g = results_frame
    max_sharpe_port_g = max_sharpe_port
    min_vol_port_g = min_vol_port


    return render_template('portfolio_optimizer.html',s_return=s_return, s_stdev=s_stdev, s_sharp=s_sharp, s_stock1=s_stock1, s_stock2=s_stock2, s_stock3=s_stock3, s_stock4=s_stock4, s_stock5=s_stock5, r_return=r_return, r_stdev=r_stdev, r_sharp=r_sharp, r_stock1=r_stock1, r_stock2=r_stock2, r_stock3=r_stock3, r_stock4=r_stock4, r_stock5=r_stock5,stock1=stocks[0],stock2=stocks[1],stock3=stocks[2],stock4=stocks[3],stock5=stocks[4])


@app.route('/abacus/optimizer/custom')
def custom_opt():

    return render_template('custom_optimizer.html')

@app.route('/abacus/optimizer/custom/portfolio', methods = ['POST'])
def custom_portfolio_opt():

    if request.method == 'POST':
        stock_ticker = request.form['stock']
        input_stdev = request.form['volatility']
        input_return = request.form['return']

    pprint(input_return)

    stock_ticker = stock_ticker.split()
    dates = pd.date_range('1995-01-01', '2017-05-18')

    c_return, c_stdev, c_sharp, c_stock1, c_stock2, c_stock3, c_stock4, c_stock5 = custom_optimizer.custom_portfolio_optimizer(stock_ticker, dates, input_stdev=input_stdev, input_return=input_return)




    return render_template('custom_portfolio_optimizer.html',c_return=c_return, c_stdev=c_stdev, c_sharp=c_sharp, c_stock1=c_stock1, c_stock2=c_stock2, c_stock3=c_stock3, c_stock4=c_stock4, c_stock5=c_stock5,stock1=stock_ticker[0],stock2=stock_ticker[1],stock3=stock_ticker[2],stock4=stock_ticker[3],stock5=stock_ticker[4])





@app.route('/plot_sma_portfolio.png')
def plot():
    df, signals, portfolio = graphs.graph_sma()

    # Create a figure
    fig = plt.figure()

    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

    # Plot the equity curve in dollars
    portfolio['total'].plot(ax=ax1, lw=2.)

    ax1.plot(portfolio.loc[signals.positions == 1.0].index,
             portfolio.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index,
             portfolio.total[signals.positions == -1.0],
             'v', markersize=10, color='k')

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response




@app.route('/plot_sma_signals.png')
def plot2():
    # get dataframes for the graphs
    df, signals, portfolio = graphs.graph_sma()

    # Initialize the plot figure
    fig = plt.figure()

    # Add a subplot and label for y-axis
    ax1 = fig.add_subplot(111, ylabel='Price in $')

    # Plot the closing price
    df['Adj. Close'].plot(ax=ax1, color='r', lw=2.)

    # Plot the short and long moving averages
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the buy signals
    ax1.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='m')

    # Plot the sell signals
    ax1.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response




@app.route('/plot_machine_learning_l_r.png')
def plot3():


    # Initialize the plot figure
    fig = plt.figure()
    
    df_predict, df_p, forecast_out = graphs.graph_machine_learning_l_r(stock)

    df_predict['Adj. Close'][-forecast_out:].plot()
    df_p['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response


@app.route('/plot_machine_learning_knn.png')
def plot4():
    # Initialize the plot figure
    fig = plt.figure()

    df_predict_2, df_p_2, forecast_out_2 = graphs.graph_machine_learning_knn(stock)

    df_predict_2['Adj. Close'][-forecast_out_2:].plot()
    df_p_2['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

@app.route('/portfolio_optimization.png')
def plot5():
    # Initialize the plot figure
    fig = plt.figure()

    # create scatter plot coloured by Sharpe Ratio
    plt.scatter(results_frame_g.stdev, results_frame_g.ret, c=results_frame_g.sharpe, cmap='RdYlBu')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.colorbar()
    # plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port_g[1], max_sharpe_port_g[0], marker=(5, 1, 0), color='r', s=1000)
    # plot green star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port_g[1], min_vol_port_g[0], marker=(5, 1, 0), color='g', s=1000)

    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

if __name__ == '__main__':
    app.debug = True
    app.run()
