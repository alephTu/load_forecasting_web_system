from flask import Flask, jsonify, render_template,redirect,request
import math 
import numpy as np
import pandas as pd
import datetime
from pandas import Series, DataFrame
import tensorflow as tf
from flask import Markup
from method.FC_DNN import fc_dnn
from method.Proposed_1 import proposed_1
from Analysis import analysis
import random

app = Flask(__name__)


@app.route("/",methods = ["GET","POST"])
def index():
    return render_template("index.html")


@app.route('/prediction/',methods=['GET','POST'])
def defaultPrediction():
    return render_template('dynamicForecast.html')


@app.route('/prediction/<load>',methods=['GET','POST'])
def prediction(load):
    if request.method == 'POST':
        # form = request.form
        load = request.form['city_code']
        req = request
        print(req.form)
        city_code = request.form['city_code']
        date = request.form['date']
        method = request.form['method']
        crossover = ''

        if method == 'FC_DNN':
            predictor = fc_dnn
        elif method == 'Proposed_1':
            predictor = proposed_1
        else:
            predictor = fc_dnn

        mape, max_load, max_time, min_load, min_time, cal_time, fe_info = predictor(forecasting_date=date,
                                                                                    city_code=city_code)

        return render_template("dynamicForecast.html", city_code=city_code, date=date, method=method,cal_time=cal_time,
                               mape=mape, max_load=max_load, max_time=max_time, min_load=min_load, min_time=min_time,
                               value=Markup(fe_info))
    else:
        return render_template('dynamicForecast.html')


@app.route('/multi',methods=["POST","GET"])
def multi():
    if request.method == 'POST':
        city_code = request.form['city_code']
        from_date = request.form['from_date']
        to_date = request.form['to_date']
        analysis(from_date=from_date, to_date=to_date)
        return render_template('multi.html',city_code=city_code,from_date=from_date, to_date=to_date)
    else:
        return render_template('multi.html')


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.errorhandler(404)
def pageNotFound(error):
    return render_template('404.html')


@app.errorhandler(500)
def notFound(error):
    return render_template('error.html')


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=80, debug=True)
    app.run()