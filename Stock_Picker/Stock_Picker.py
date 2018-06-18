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


import random
from io import StringIO, BytesIO

from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



@app.route('/')
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


















if __name__ == '__main__':
    app.debug = True
    app.run()
