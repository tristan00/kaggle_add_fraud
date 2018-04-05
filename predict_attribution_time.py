import pandas as pd
import datetime
import time
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, mean_squared_error
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'mae',
    'num_threads': 4,
    'scale_pos_weight':400
}

def get_model(df):
    model = Sequential()
    model.add(Dense(128, activation='elu', input_shape=(df.shape[1],)))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(1, activation='elu'))
    model.compile(loss='mse',
                 optimizer=optimizers.RMSprop(),
                 metrics=['mse'])
    return model


init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }

path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'
df = pd.read_csv(path + 'train.csv', dtype=init_dtype, nrows = 1000000)

df_attributed = df.loc[df['is_attributed'] == 1]
print(df_attributed.shape)
df_attributed['attributed_time_span'] = df_attributed.apply(lambda x: datetime.datetime.strptime(x['attributed_time'], '%Y-%m-%d %H:%M:%S').timestamp() -
                                                datetime.datetime.strptime(x['click_time'],'%Y-%m-%d %H:%M:%S').timestamp(), axis = 1)

print(df_attributed.describe())

y = df_attributed['attributed_time_span']

df_attributed['datetime'] = df_attributed['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df_attributed['click_hour'] = df_attributed['datetime'].apply(lambda x: x.hour).astype('uint8')
df_attributed['click_day'] = df_attributed['datetime'].apply(lambda x: x.day).astype('uint8')
df_attributed['click_second'] = df_attributed['datetime'].apply(lambda x: x.second).astype('uint8')
df_attributed['click_minute'] = df_attributed['datetime'].apply(lambda x: x.minute).astype('uint8')
# df['click_minute'] /= 60 #TODO: change to 60 on new models

df_attributed = df_attributed.drop(['attributed_time', 'click_time', 'attributed_time_span', 'ip', 'datetime'], axis = 1)

train_x, test_x, train_y, test_y = model_selection.train_test_split(df_attributed, y, test_size=.1)


clf = ensemble.AdaBoostRegressor(n_estimators=200, loss='exponential')
clf.fit(train_x, train_y)
print('ada', clf.score(test_x, test_y))

clf = ensemble.GradientBoostingRegressor(n_estimators=200)
clf.fit(train_x, train_y)
print('gbr', clf.score(test_x, test_y))

clf = ensemble.RandomForestRegressor(n_estimators=200)
clf.fit(train_x, train_y)
print('rfr', clf.score(test_x, test_y))

clf = ensemble.ExtraTreesRegressor(n_estimators=200)
clf.fit(train_x, train_y)
print('etr', clf.score(test_x, test_y))


