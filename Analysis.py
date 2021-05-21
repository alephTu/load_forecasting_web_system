#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time    : 2021/5/17 17:24
# @Author  : Sijun Du
# @FileName: analysis.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import time
import seaborn as sns


def analysis(from_date, to_date):
    data_dir = 'dataset'
    load = pd.read_csv(data_dir + '/load_24.csv', parse_dates=['date'])
    tep = pd.read_csv(data_dir + '/tep_24.csv', parse_dates=['date'])
    hum = pd.read_csv(data_dir + '/hum_24.csv', parse_dates=['date'])
    wind = pd.read_csv(data_dir + '/wind_24.csv', parse_dates=['date'])
    rain = pd.read_csv(data_dir + '/rain_24.csv', parse_dates=['date'])
    holiday = pd.read_csv(data_dir + '/holiday.csv', parse_dates=['date'])

    # 预测日

    # print(test_day)
    load = load[(load['date'] >= from_date) & (load['date'] <= to_date)].drop(['date'], axis=1)  # 数据范围
    tep = tep[(tep['date'] >= from_date) & (tep['date'] <= to_date)].drop(['date'], axis=1)  # 数据范围
    hum = hum[(hum['date'] >= from_date) & (hum['date'] <= to_date)].drop(['date'], axis=1)  # 数据范围
    wind = wind[(wind['date'] >= from_date) & (wind['date'] <= to_date)].drop(['date'], axis=1)  # 数据范围
    rain = rain[(rain['date'] >= from_date) & (rain['date'] <= to_date)].drop(['date'], axis=1)  # 数据范围

    def new_shape(ts, name):
        n = len(ts)
        new_ts = np.array(ts).reshape((1, 96 * n))[0]
        pd_ts = pd.DataFrame(new_ts, columns=[name])
        return pd_ts

    new_load = new_shape(load, 'Load')
    new_tep = new_shape(tep, 'Tep')
    new_hum = new_shape(hum, 'Hum')
    new_wind = new_shape(wind, 'Wind')
    new_rain = new_shape(rain, 'Rain')

    df = pd.concat([new_load, new_tep, new_hum, new_wind, new_rain], axis=1)
    corr = df.corr()
    plt.figure(1)
    plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, vmax=1, cmap="Reds")
    plt.title('Features Correlation: ' + from_date + ' to ' + to_date)
    plt.savefig('static/corr.png')
    plt.close()
    #plt.show()

    def time_lag_corr(weather):
        lag = [i for i in range(0, 97)]
        lag_corr = [df['Load'].corr(df[weather].shift(lag[i])) for i in lag]
        max_index = lag_corr.index(max(lag_corr))

        plt.figure(2, figsize=(10, 4), facecolor='#F4F4F4', edgecolor='#F4F4F4')
        plt.plot(lag, lag_corr)
        plt.axvline(0, color='k', linestyle='--', label=('Origin Correlation: ' + str(round(lag_corr[0], 5))))
        plt.axvline(max_index, color='red', linestyle='--',
                    label='Best Time Lag Correlation: ' + str(round(lag_corr[max_index], 5)))
        plt.title('Time Lag Correlation: ' + weather + ' leads Load')
        plt.xlabel('Time/15min')
        plt.ylabel('Correlation Factor')
        plt.grid(axis='y')
        plt.legend()
        plt.savefig('static/corr_' + weather + '.png')
        #plt.show()
        plt.close()

    for i in ['Tep', 'Hum', 'Wind', 'Rain']:
        time_lag_corr(weather=i)

    return 'finish'


if __name__ == '__main__':
    analysis(from_date='2018-01-01', to_date='2018-01-31')