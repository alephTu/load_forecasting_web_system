#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time    : 2021/5/11 19:03
# @Author  : Sijun Du
# @FileName: simple_dnn.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, optimizers,regularizers
import matplotlib.pyplot as plt
import random
import datetime
import time


def fc_dnn(forecasting_date, city_code='C1'):
    start = time.clock()

    my_seed = 582
    np.random.seed(my_seed)
    random.seed(my_seed)
    tf.set_random_seed(my_seed)

    data_dir = 'dataset'
    load = pd.read_csv(data_dir + '/load_24.csv', parse_dates=['date'])
    tep = pd.read_csv(data_dir + '/tep_24.csv', parse_dates=['date'])
    hum = pd.read_csv(data_dir + '/hum_24.csv', parse_dates=['date'])
    wind = pd.read_csv(data_dir + '/wind_24.csv', parse_dates=['date'])
    rain = pd.read_csv(data_dir + '/rain_24.csv', parse_dates=['date'])
    holiday = pd.read_csv(data_dir + '/holiday.csv', parse_dates=['date'])

    # 预测日
    test_day = forecasting_date
    # print(test_day)
    load = load[(load['date'] >= '2015-01-01') & (load['date'] <= test_day)]  # 数据范围
    tep = tep[(tep['date'] >= '2015-01-01') & (tep['date'] <= test_day)]  # 数据范围
    hum = hum[(hum['date'] >= '2015-01-01') & (hum['date'] <= test_day)]  # 数据范围
    wind = wind[(wind['date'] >= '2015-01-01') & (wind['date'] <= test_day)]  # 数据范围
    rain = rain[(rain['date'] >= '2015-01-01') & (rain['date'] <= test_day)]  # 数据范围

    # 输出特征
    output_load = load.drop(load.head(1).index).reset_index(drop=True)

    # 输入特征
    # 负荷
    input_load = load.drop(load.tail(1).index).reset_index(drop=True).drop(['date'], axis=1)

    # 预测日日期特征
    input_date = pd.DataFrame(output_load['date'])
    input_date['week'] = input_date.date.apply(lambda x: x.dayofweek)  # 周几
    input_date = input_date.merge(holiday, on='date', how='left').drop(['date'], axis=1)
    enc = OneHotEncoder().fit(input_date)
    input_date_enc = enc.transform(input_date).toarray()
    input_date_enc = pd.DataFrame(input_date_enc)

    # 预测日气象特征
    input_tep = tep.drop(tep.head(1).index).reset_index(drop=True).drop(['date'], axis=1)
    input_hum = hum.drop(hum.head(1).index).reset_index(drop=True).drop(['date'], axis=1)
    input_wind = wind.drop(wind.head(1).index).reset_index(drop=True).drop(['date'], axis=1)
    input_rain = rain.drop(rain.head(1).index).reset_index(drop=True).drop(['date'], axis=1)
    # 计算统计值
    avr_tep = input_tep.mean(axis=1)
    max_tep = input_tep.max(axis=1)
    min_tep = input_tep.min(axis=1)
    avr_hum = input_hum.mean(axis=1)
    avr_wind = input_wind.mean(axis=1)
    avr_rain = input_rain.mean(axis=1)
    input_weather = pd.concat([avr_tep, max_tep, min_tep, avr_hum, avr_wind, avr_rain], axis=1)

    # 输入特征
    input_fe = pd.concat([input_load, input_weather, input_date_enc], axis=1)

    # 输出特征
    output_fe = output_load.drop(['date'], axis=1)

    # 训练集
    x_train = input_fe.drop(input_fe.tail(1).index).reset_index(drop=True)
    y_train = output_fe.drop(output_fe.tail(1).index).reset_index(drop=True)

    # 测试集
    x_test = input_fe.tail(1).reset_index(drop=True)
    y_test = output_fe.tail(1).reset_index(drop=True)

    # 归一化
    Z_score_x = StandardScaler().fit(x_train)
    # print(Z_score_x)
    x_for_train = Z_score_x.transform(x_train)  # 训练属性
    x_for_test = Z_score_x.transform(x_test)  # 测试属性
    # #
    Z_score_y = StandardScaler().fit(y_train)
    y_for_train = Z_score_y.transform(y_train)  # 训练属性

    # DNN
    dnn_model = tf.keras.Sequential([
        layers.Dense(72, activation='relu', input_dim=111, kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(96, activation='linear')])

    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
    dnn_model.compile(optimizer=rms, loss='mse')
    dnn_model.fit(x_for_train, y_for_train, epochs=180, batch_size=16)

    dnn_pred = dnn_model.predict(x_for_test)
    dnn_result = Z_score_y.inverse_transform(dnn_pred)

    predict_load = dnn_result[0]
    real_load = np.array(y_test)[0]

    plt.figure(1, facecolor='#F4F4F4', edgecolor='#F4F4F4')
    plt.plot(predict_load, '-.', label='predict_by_FC_DNN')
    plt.plot(real_load, 'red', label='real_load')
    plt.title('Forecasting Load: ' + city_code + ',' + forecasting_date)
    plt.xlabel('Time/15min')
    plt.ylabel('Load/MW')
    plt.legend()
    plt.savefig('static/predict_test.png')
    plt.close()

    plt.figure(2, figsize=(9.5, 4),facecolor='#F4F4F4', edgecolor='#F4F4F4')
    plt.plot(np.array(x_test)[0][0:96], 'orange', label='input_load')
    plt.plot(predict_load, label='output_load')
    plt.title('Forecasting Performance: ' + 'Input Load Vs. Output Load')
    plt.xlabel('Time/15min')
    plt.ylabel('Load/MW')
    plt.legend()
    plt.savefig('static/input_output.png')
    plt.close()

    mape = 100 * mean_absolute_percentage_error(predict_load, real_load) # 计算预测误差
    max_load = max(predict_load)
    max_index = int(np.argwhere(predict_load == max_load)[0][0])
    max_time = str(datetime.timedelta(minutes=max_index*15))
    min_load = min(predict_load)
    min_index = int(np.argwhere(predict_load == min_load)[0][0])
    min_time = str(datetime.timedelta(minutes=min_index*15))

    # info
    date_info = holiday[(holiday['date'] == test_day)]  # 数据范围
    week_day = np.array(date_info.date.apply(lambda x: x.dayofweek))[0]
    holi_day = np.array(date_info['holiday'])[0]
    weather_info = np.array(x_test)[0][96:102]
    total_info = {
        '节假日': str(holi_day),
        '周几': str(week_day),
        '最高温度': round(weather_info[1], 2),
        '最低温度': round(weather_info[2], 2),
        '平均温度': round(weather_info[0], 2),
        '平均相对湿度': round(weather_info[3], 2),
        '平均风速': round(weather_info[4], 2),
        '平均降雨量': round(weather_info[5], 2),
    }
    df = pd.DataFrame(pd.Series(total_info), columns=['值'])
    fe_info = df.reset_index().rename(columns={'index': '特征'})

    end = time.clock()

    return round(mape, 4), max_load, max_time, min_load, min_time, round((end-start), 2), \
           fe_info.to_html(index=False, header=False)


if __name__ == '__main__':
    mape, max_load, max_time, min_load, min_time,cal_time, fe_info = fc_dnn(forecasting_date='2018-01-10')
    print(mape)
    print(max_load)
    print(max_time)
    print(min_load)
    print(min_time)
    print(cal_time)






