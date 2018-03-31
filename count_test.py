import time
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import datetime
import math
import tensorflow as tf
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



from sklearn.preprocessing import OneHotEncoder


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 2000

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
    'scale_pos_weight':389
}

init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }


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

def time_between_df(df, columns, name):
    print(columns)
    df[name] = df.groupby(columns)['click_time'].diff()
    df[name] = df[name].fillna(value=0)
    df[name] = df[name].astype('uint16')
    return df

def preproccess_df(df):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour).astype('uint8')
    df['click_day'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day).astype('uint8')
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute).astype('uint8')
    #df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
    print('time added', df.shape)

    possible_names = ['ip', 'device', 'os', 'channel', 'click_hour', 'click_day']


    count_lists = [['ip'],
                ['ip', 'device', 'os'],
                ['click_hour', 'click_day', 'ip', 'device', 'os'],
                   ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel'],
                   ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel', 'app']]

    print(time.time() - start_time)
    for i in count_lists:
        df = count_df(df, list(i), '_'.join(i) + '_counter')
        print(time.time() - start_time)
        df = rank_df(df, list(i), '_'.join(i) + '_rank')
        print(time.time() - start_time)
        #df = time_between_df(df, list(i), '_'.join(i) + '_diff')
        print(time.time() - start_time)
    df = df.drop(['ip','click_time', 'counting_column', 'click_day'], axis=1)
    print(df.shape)
    #df.drop(['counting_column'], axis=1, inplace=True)
    return df

df = pd.read_csv(path + "train2.csv", dtype=init_dtype)

train, val = model_selection.train_test_split(df, test_size=.4)
del df

train = preproccess_df(train)
val = preproccess_df(val)

y_train = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
y_val = val['is_attributed']
val.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

dtrain = lgb.Dataset(train, label=y_train)
dval = lgb.Dataset(val, label=y_val, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50, verbose_eval=10, categorical_feature=['device', 'os', 'channel', 'click_hour'])

del y_train, train, y_val, val

sub = pd.DataFrame()

test = pd.read_csv(path + "test.csv", dtype=init_dtype)
test = preproccess_df(test)
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

sub['is_attributed'] = model.predict(test, num_iteration=model.best_iteration or MAX_ROUNDS)
sub.to_csv('lgb_sub.csv', index=False)






