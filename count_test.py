import time
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, mean_squared_error
from keras.models import Model, load_model, save_model
import lightgbm as lgb
import datetime
import math
import tensorflow as tf
import h5py
import statistics
import glob
import pickle
import random
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, Softmax, Flatten
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from keras import optimizers, losses, metrics
from keras import callbacks
import keras
from keras import backend as K
import os
import itertools
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 400

params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 4,
    'scale_pos_weight':400
}

init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }


def get_autoencoder(df):
    model = Sequential()
    model.add(Dense(128, activation='elu', input_shape=(df.shape[1],)))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(8, activation='elu'))
    model.add(Dense(4, activation='elu'))
    model.add(Dense(8, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(df.shape[1], activation='elu'))
    model.compile(loss='mse',
                 optimizer=optimizers.RMSprop(),
                 metrics=['mse'])
    return model


def add_outlier(df):
    nn_df = df.copy()
    nn_columns = nn_df.columns

    print('autoencoder loaded', time.time() - start_time)

    scaler = MinMaxScaler(feature_range=(0.0, 10.0))
    nn2_df = scaler.fit_transform(nn_df)
    try:
        model = load_model(path + 'auto4.h5')
    except:
        model = get_autoencoder(nn_df)
        model.fit(nn2_df, nn2_df, epochs=1)
        model.save(path + 'auto4.h5')

    # model = get_autoencoder(nn_df)
    # model.fit(nn2_df, nn2_df, epochs=1)
    # model.save(path + 'auto1.h5')

    gc.collect()
    df_pre = nn2_df.copy()
    df_pre = pd.DataFrame(data=df_pre,
                       columns =[i + '_pre' for i in nn_columns],
                       index=nn_df.index)
    gc.collect()
    print('starting predictions', time.time() - start_time)
    preds = model.predict(df_pre)
    print('predicted', time.time() - start_time)
    df2 = pd.DataFrame(data=preds,
                       columns =[i+'_pred' for i in nn_columns],
                       index=nn_df.index)

    error = mean_squared_error(df_pre, df2,multioutput='raw_values')
    print(error.shape, df_pre.shape, df2.shape)

    #TODO: vectorize
    output_array = []
    for count, ((_, i), (_, j)) in enumerate(zip(df_pre.iterrows(), df2.iterrows())):
        output_array.append(mean_squared_error(i, j))
        if count % 100000 == 0:
            print('predicting outliers', count, time.time() - start_time)
    df['nn_score'] = np.array(output_array)


    return df


def rank_df(df, columns, name):
    print(columns)
    df2 = df.groupby(columns).count().add_suffix('_count').reset_index()
    df2 = df2[columns + ['counting_column_count']]
    df2[name] = df2['counting_column_count'].rank(method='dense')
    max_rank = max(df2[name])
    df2[name] /= max_rank
    df2 = df2[columns + [name]]
    df2[name] = df2[name].astype('float32')
    df = df.merge(df2, on=columns)
    return df

def count_df(df, columns, name):
    print(columns)
    df[name] = df.groupby(columns).cumcount()
    df[name] = df[name].astype('uint16')
    return df

def time_since_last(df, columns, name):
    df[name] = df.groupby(columns)['click_time'].diff()
    df[name] = df[name].fillna(-1)
    # df[name+'_sum'] = df.groupby(columns)[name].sum()
    # df[name + '_count'] = df.groupby(columns)[name].count()
    # df[name + '_avg'] = df.apply(lambda x: x[name+'_sum'] / x[name + '_count'], axis = 1)
    # df[name + '_avg'] = df[name+'_avg'].fillna(0)
    # df[name + '_click_time_port'] = df.apply(lambda x: x[name] / x[name+'_avg'] if x[name+'_avg'] > 0 else  0, axis = 1)
    # df = df.sort_values('ip')
    # df = df.drop([name+'_sum', name + '_count'], axis = 1)
    return df

def time_till_next(df, columns, name):
    df[name] = df.groupby(columns)['click_time'].transform(
        lambda x: x.diff().shift(-1))
    df[name] = df[name].fillna(-1)
    # df[name+'_sum'] = df.groupby(columns)[name].sum(skipna=True)
    # df[name + '_count'] = df.groupby(columns)[name].count()
    # df[name + '_avg'] = df.apply(lambda x: x[name+'_sum'] / x[name + '_count'], axis = 1)
    # df[name + '_avg'] = df[name+'_avg'].fillna(0)
    # df[name + '_click_time_port'] = df.apply(lambda x: x[name] / x[name+'_avg'] if x[name+'_avg'] > 0 else 0, axis = 1)
    # df = df.drop([name + '_sum', name + '_count'], axis = 1)
    return df

def preproccess_df(df):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['datetime'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['click_hour'] = df['datetime'].apply(lambda x: x.hour).astype('uint8')
    df['click_day'] = df['datetime'].apply(lambda x: x.day).astype('uint8')
    df['click_second'] = df['datetime'].apply(lambda x: x.second).astype('uint8')
    df['click_minute'] = df['datetime'].apply(lambda x: x.minute).astype('uint8')
    #df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['datetime'].apply(lambda x: x.timestamp())
    df = df.drop(['datetime'], axis = 1)

    df = df.sort_values(by=['click_time'])

    df['click_hour'] = df['click_hour'].apply(lambda x: 5 if x == 6 else 10 if x == 11 else 14)


    print('time added', df.shape)
    #


    possible_names = ['ip', 'device', 'os', 'channel', 'click_day']

    # for l in range(len(possible_names)):
    #     combinations = itertools.combinations(possible_names, l+1)
    #     for i in combinations:
    #         if 'click_day' not in i or len(i) == 1 or len(i) == 5:
    #             continue
    #         print(i, time.time() - start_time)
    #         df = time_since_last(df, list(i), '_'.join(i)+'_next')
    #         gc.collect()
    #         # df = time_till_next(df, list(i), '_'.join(i) + '_last')
    #         # gc.collect()


    rank_list = [['ip', 'os', 'channel', 'click_day'],
                 ['ip', 'device', 'channel', 'click_day'],
                 ['ip', 'device', 'os', 'click_day']]
    for i in rank_list:
        df = time_since_last(df, list(i), '_'.join(i) + '_last')
        gc.collect()
        df = time_till_next(df, list(i), '_'.join(i) + '_next')
        gc.collect()


    rank_list = [['ip'],
                 ['ip', 'device'],
                 ['ip', 'os'],
                 ['device', 'os'],
                 ['ip', 'device', 'channel'],
                 ['channel'],
                 ['ip', 'device', 'os'],
                 ['ip', 'channel'],
                 ['device', 'os', 'channel'],
                 ['os', 'channel'],
                 ['device'],
                 ['device', 'channel'],
                 ['os'],
                 ['app', 'channel']]
    for i in rank_list:
        df = rank_df(df, list(i), '_'.join(i) + '_rank')
        df = df.sort_values(by=['click_time'])
        gc.collect()

    # count_lists = [
    #             ['ip', 'device', 'os', 'channel', 'app'],
    #             ['click_hour', 'click_day', 'ip', 'device', 'os'],
    #                ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel'],
    #                ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel', 'app']]
    #
    # print(time.time() - start_time)
    # for i in count_lists:
    #     df = rank_df(df, list(i), '_'.join(i) + '_rank')

    df['device'] = df['device'].astype('category')
    df['os'] = df['os'].astype('category')
    df['channel'] = df['channel'].astype('category')
    df['click_hour'] = df['click_hour'].astype('category')
    df = df.drop(['ip','click_time', 'counting_column', 'click_day', 'click_day'], axis=1)


    print(df.shape)


    #df.drop(['counting_column'], axis=1, inplace=True)
    return df



def main_with_val():
    df = pd.read_csv(path + "train2.csv", dtype=init_dtype)

    train, val = model_selection.train_test_split(df, test_size=.4)
    del df

    train = preproccess_df(train)
    gc.collect()
    val = preproccess_df(val)
    gc.collect()
    y_train = train['is_attributed']
    train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    y_val = val['is_attributed']
    val.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    gc.collect()
    train = add_outlier(train)
    gc.collect()
    val = add_outlier(val)
    gc.collect()

    dtrain = lgb.Dataset(train, label=y_train)
    dval = lgb.Dataset(val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    columns = train.columns
    f_i = model.feature_importance(importance_type='split')
    f_i2 = model.feature_importance(importance_type='gain')

    fi_df = pd.DataFrame(columns=columns)
    fi_df.loc[len(fi_df)] = f_i
    fi_df.loc[len(fi_df)] = f_i2
    fi_df.to_csv('f2.csv', index = False)

    del y_train, train, y_val, val



    sub = pd.DataFrame()

    test = pd.read_csv(path + "test.csv", dtype=init_dtype)
    test = preproccess_df(test)
    sub['click_id'] = test['click_id']
    test = add_outlier(test)
    test.drop('click_id', axis=1, inplace=True)

    sub['is_attributed'] = model.predict(test, num_iteration=model.best_iteration or MAX_ROUNDS)
    sub.to_csv('lgb_sub.csv', index=False, compression='gzip')

def main_wo_val():
    df = pd.read_csv(path + "train2.csv", dtype=init_dtype)
    train = preproccess_df(df)
    gc.collect()

    y_train = train['is_attributed']
    train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

    train = add_outlier(train)
    gc.collect()

    train.to_csv('training_data.csv', index=False, sep = '|')
    y_train.to_csv('training_data_y.csv', index=False, sep = '|')

    train, val, y_train, y_val = model_selection.train_test_split(train, y_train, test_size=.1)

    dtrain = lgb.Dataset(train, label=y_train)
    dval = lgb.Dataset(val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    columns = train.columns
    f_i = model.feature_importance(importance_type='split')
    f_i2 = model.feature_importance(importance_type='gain')

    fi_df = pd.DataFrame(columns=columns)
    fi_df.loc[len(fi_df)] = f_i
    fi_df.loc[len(fi_df)] = f_i2
    fi_df.to_csv('f2.csv', index=False)

    del y_train, train

    sub = pd.DataFrame()

    test = pd.read_csv(path + "test.csv", dtype=init_dtype)
    test = preproccess_df(test)
    sub['click_id'] = test['click_id']
    test.drop('click_id', axis=1, inplace=True)

    test = add_outlier(test)
    sub['is_attributed'] = model.predict(test, num_iteration=model.best_iteration or MAX_ROUNDS)
    sub.to_csv('lgb_sub.csv', index=False)

if __name__ == '__main__':
    main_wo_val()






