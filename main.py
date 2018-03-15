
import pandas as pd
import time
import numpy as np
from sklearn import model_selection
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
from sklearn.neighbors import KNeighborsClassifier
from keras import optimizers, losses, metrics
from keras import callbacks
import keras

from sklearn.preprocessing import OneHotEncoder


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 1000

class_weight = {0 : 1,
    1: 390}

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


def print_sample():
    df = pd.read_csv(path + "train.csv", nrows=500000)
    df = preproccess_df(df)
    positives = df.loc[df['is_attributed'] == 1]
    negatives = df.loc[df['is_attributed'] == 0]
    print(negatives.shape[0]/positives.shape[0])
    negatives = negatives.sample(n=max(positives.shape))
    output = pd.concat([positives, negatives])
    output.to_csv('sample.csv')


def preproccess_df(df):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    df['click_hour'] /= 24
    df['click_day'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
    df['click_day'] /= 31
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)
    df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
    print('time added', df.shape)

    df['os'] /= 1.0
    df['device'] /= 1.0

    df_ip = df.groupby(['ip', 'device', 'os', 'click_hour', 'click_day']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'device', 'os', 'click_hour', 'click_day', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['ip_app_os_hour_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_app_os_hour_rank'])
    df_ip['ip_app_os_hour_rank'] /= max_ip_rank
    df_ip = df_ip[['ip', 'device', 'os', 'click_hour', 'click_day', 'ip_app_os_hour_rank']]
    df = df.merge(df_ip, on=['ip', 'device', 'os','click_day', 'click_hour'])
    del df_ip
    print('ip_app added', df.shape)

    df_ip = df.groupby(['ip', 'app']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'app', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['ip_app_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_app_rank'])
    df_ip['ip_app_rank'] /= max_ip_rank
    df_ip = df_ip[['ip', 'app', 'ip_app_rank']]
    df = df.merge(df_ip, on=['ip', 'app'])
    del df_ip
    print('ip_app added', df.shape)

    df_ip = df.groupby(['ip', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'channel', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['ip_channel_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_channel_rank'])
    df_ip['ip_channel_rank'] /= max_ip_rank
    df_ip = df_ip[['ip', 'channel', 'ip_channel_rank']]
    df = df.merge(df_ip, on=['ip', 'channel'])
    del df_ip
    print('ip_channel added', df.shape)

    df = df.sort_values(by=['app', 'channel'])

    df_ip = df.groupby(['app', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['app', 'channel', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['app_channel_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['app_channel_rank'])
    df_ip['app_channel_rank'] /= max_ip_rank
    df_ip = df_ip[['app', 'channel', 'app_channel_rank']]
    df = df.merge(df_ip, on=['app', 'channel'])
    del df_ip
    print('app_channel added', df.shape)

    df = df.sort_values(by=['ip', 'app', 'channel'])
    df_ip = df.groupby(['ip', 'app', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip','app', 'channel', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['ip_app_channel_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_app_channel_rank'])
    df_ip['ip_app_channel_rank'] /= max_ip_rank
    df_ip = df_ip[['ip', 'app', 'channel', 'ip_app_channel_rank']]
    df = df.merge(df_ip, on=['ip', 'app', 'channel'])
    del df_ip
    print('app_channel added', df.shape)

    df_ip = df.groupby(['ip']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'counting_column_count']]
    df_ip = df_ip.sort_values(by='counting_column_count')
    df_ip['ip_rank'] = df_ip['counting_column_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_rank'])
    df_ip['ip_rank'] /= max_ip_rank
    df_ip = df_ip[['ip', 'ip_rank']]
    df = df.merge(df_ip, on='ip')
    del df_ip
    print('ip added', df.shape)

    df_app = df.groupby(['app']).count().add_suffix('_count').reset_index()
    df_app = df_app[['app', 'counting_column_count']]
    df_app = df_app.sort_values(by='counting_column_count')
    df_app['app_rank'] = df_app['counting_column_count'].rank(method='dense')
    max_app_rank = max(df_app['app_rank'])
    df_app['app_rank'] /= max_app_rank
    df_app = df_app[['app', 'app_rank']]
    df = df.merge(df_app, on='app')
    del df_app
    print('app added', df.shape)

    df_channel = df.groupby(['channel']).count().add_suffix('_count').reset_index()
    df_channel = df_channel[['channel', 'counting_column_count']]
    df_channel = df_channel.sort_values(by='counting_column_count')
    df_channel['channel_rank'] = df_channel['counting_column_count'].rank(method='dense')
    max_channel_rank = max(df_channel['counting_column_count'])
    df_channel['channel_rank'] /= max_channel_rank
    df_channel = df_channel[['channel', 'channel_rank']]
    df = df.merge(df_channel, on='channel')
    del df_channel
    print('channel added', df.shape)

    df_os = df.groupby(['os']).count().add_suffix('_count').reset_index()
    df_os = df_os[['os', 'counting_column_count']]
    df_os = df_os.sort_values(by='counting_column_count')
    df_os['os_rank'] = df_os['counting_column_count'].rank(method='dense')
    max_os = max(df_os['os_rank'])
    df_os['os_rank'] /= max_os
    df_os = df_os[['os', 'os_rank']]
    df = df.merge(df_os, on='os')
    del df_os
    print('channel added', df.shape)

    print(df.columns)

    #df = df.drop(['ip', 'app', 'os', 'channel','click_time', 'counting_column'], axis=1)
    df = df.drop(['click_time', 'counting_column'], axis=1)
    #df.drop(['counting_column'], axis=1, inplace=True)
    return df

def get_nn(df):
    model = Sequential()
    model.add(Dense(64, activation='elu', input_shape=(df.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.RMSprop(),
                 metrics=['accuracy'])
    return model


def get_auc_score(clf, x1, y2):
    y1 = clf.predict(x1)
    return roc_auc_score(y1,y2)

def train_l1_models():
    models = []
    train_raw = pd.read_csv(path + "train.csv", nrows=80000000)
    starting_columns = train_raw.columns
    val = pd.read_csv(path + "train.csv", skiprows=80000000, nrows=20000000)
    val.columns = starting_columns
    print('[{0}] Finished to load data'.format(time.time() - start_time))
    train = preproccess_df(train_raw)
    val = preproccess_df(val)
    y_train = train['is_attributed']
    y_val = val['is_attributed']

    train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    val.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

    print('[{}] Start LGBM Training'.format(time.time() - start_time))
    dtrain = lgb.Dataset(train, label=y_train)
    dval = lgb.Dataset(val, label=y_val, reference=dtrain)
    lgbm_model1 = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                      early_stopping_rounds=50, verbose_eval=10)

    print('[{0}] Finish LGBM Training, {1}'.format(time.time() - start_time, 1))
    with open(path + 'l1/light_gbm1.plk', 'wb') as infile:
        pickle.dump(lgbm_model1, infile)

    x3 = train.as_matrix()
    y3 = np.expand_dims(y_train.as_matrix(), 1)
    x4 = val.as_matrix()
    y4 = np.expand_dims(y_val.as_matrix(), 1)
    del train, val, lgbm_model1
    y3 = keras.utils.to_categorical(y3, 2)
    y4 = keras.utils.to_categorical(y4, 2)
    print(x3.shape, y3.shape)

    nn_model = get_nn(x3)
    nn_model.fit(x3, y3, epochs=2, class_weight=class_weight, verbose=0)
    print('nn trained:', nn_model.evaluate(x4,y4, verbose=0))
    nn_model.save(path + 'l1/model_nn.h5')
    del nn_model

    gb = ensemble.GradientBoostingClassifier()
    gb.fit(train,y_train)
    print('gb', gb.score(val, y_val))
    with open(path + 'l1/gb.plk', 'wb') as infile:
        pickle.dump(gb, infile)
    del gb

    ada = ensemble.AdaBoostClassifier()
    ada.fit(train, y_train)
    print('ada', ada.score(val, y_val))
    with open(path + 'l1/ada.plk', 'wb') as infile:
        pickle.dump(ada, infile)

    del ada

    rf = ensemble.RandomForestClassifier()
    rf.fit(train, y_train)
    print('rf', rf.score(val, y_val))
    with open(path + 'l1/rf.plk', 'wb') as infile:
        pickle.dump(rf, infile)

    del rf

    et = ensemble.ExtraTreesClassifier()
    et.fit(train, y_train)
    print('et', et.score(val, y_val))
    with open(path + 'l1/et.plk', 'wb') as infile:
        pickle.dump(et, infile)

    del et

    k = KNeighborsClassifier()
    k.fit(train, y_train)
    print('k', k.score(val, y_val))
    with open(path + 'l1/et.plk', 'wb') as infile:
        pickle.dump(k, infile)



def load_models():
    model_locs = glob.glob(path + '*.plk')
    models = []
    for m in model_locs:
        with open(m, 'rb') as infile:
            models.append(pickle.load(infile))
    train_raw = pd.read_csv(path + "train.csv", skiprows=150000000)
    train_raw.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
       'is_attributed']
    return models, train_raw


def get_l1_predictions(start_index):
    model_locs = glob.glob(path + 'l1/*.plk')
    model_locs_nn = glob.glob(path + 'l1/*.h5')

    df = pd.read_csv(path + "train.csv", skiprows=start_index)
    df.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
                         'is_attributed']
    df = preproccess_df(df)
    y = df['is_attributed']
    df.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    chunk_result = pd.DataFrame()

    for count, m in enumerate(model_locs):
        with open(m, 'rb') as infile:
            model = pickle.load(infile)
        chunk_result['is_attributed_{0}'.format(count)] = model.predict(df)
    for count, m in enumerate(model_locs_nn):
        model = keras.models.load_model(m)
        chunk_result['is_attributed_nn_{0}'.format(count)] = model.predict(df)[:,0]
    start_index += max(df.shape)

    return chunk_result, y


def chunk_test_predictions(df):
    model_locs = glob.glob(path + 'l1/*.plk')
    model_locs_nn = glob.glob(path + 'l1/*.h5')
    predictions = []


    chunk_result = pd.DataFrame()
    for count, m in enumerate(model_locs):
        with open(m, 'rb') as infile:
            model = pickle.load(infile)
        chunk_result['is_attributed_{0}'.format(count)] = model.predict(df,num_iteration=model.best_iteration or MAX_ROUNDS)
    for count, m in enumerate(model_locs_nn):
        model = keras.models.load_model(m)
        chunk_result['is_attributed_nn_{0}'.format(count)] = model.predict(df)[:,0]

    # while df is None or max(df.shape) == chunk_size:
    #     df = pd.read_csv(path + "test.csv", nrows=chunk_size)
    #     df.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
    #                          'is_attributed']
    #     df = preproccess_df(df)
    #
    #     df.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    #
    #     chunk_result = pd.DataFrame()
    #
    #     for count, m in enumerate(model_locs):
    #         with open(m, 'rb') as infile:
    #             model = pickle.load(infile)
    #         chunk_result['is_attributed_{0}'.format(count)] = model.predict(df, num_iteration=model.best_iteration or MAX_ROUNDS)
    #     predictions.append(chunk_result)
    # predictions = pd.concat(predictions)
    return chunk_result


def get_l2_model():
    if len(glob.glob(path + 'l2/*.plk')) == 0:

        train_l1_models()
        x, y = get_l1_predictions(100000000)
        x1, x2, y1, y2 = model_selection.train_test_split(x, y, test_size=0.2, shuffle=True)
        dtrain = lgb.Dataset(x1, label=y1)
        dval = lgb.Dataset(x2, label=y2, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                          early_stopping_rounds=50, verbose_eval=10)
        with open(path + 'l2/model.plk', 'wb') as infile:
            pickle.dump(model, infile)
    else:
        model_loc = glob.glob(path + 'l2/*.plk')[0]
        with open(model_loc, 'rb') as infile:
            model = pickle.load(infile)
    return model


def main():
    l1_test = pd.DataFrame()
    l1_train = pd.DataFrame()
    sub = pd.DataFrame()

    model = get_l2_model()

    test = pd.read_csv(path + "test.csv")
    test = preproccess_df(test)
    sub['click_id'] = test['click_id']
    test.drop('click_id', axis=1, inplace=True)

    x = chunk_test_predictions(test)

    sub['is_attributed'] = model.predict(x, num_iteration=model.best_iteration or MAX_ROUNDS)
    sub.to_csv('lgb_sub.csv', index=False)


if __name__ == '__main__':
    main()
    #print_sample()