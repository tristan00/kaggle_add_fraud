#started from from https://www.kaggle.com/tunguz/lgbm-starter-2

import pandas as pd
import time
import numpy as np
from sklearn import model_selection
import lightgbm as lgb
import datetime
import math
import statistics
import glob
import pickle

path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 250

params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 25,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'metric': 'auc',
    'num_threads': 4
}


def preproccess_df(df):
    print(df.columns)
    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    print('time added')

    df_ip = df.groupby(['ip', 'app']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'app', 'device_count']]
    df_ip = df_ip.sort_values(by='device_count')
    df_ip['ip_app_rank'] = df_ip['device_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_app_rank'])
    df_ip['ip_app_rank'] /= max_ip_rank
    df = df.merge(df_ip, how='left', on=['ip', 'app'])
    del df_ip
    print('ip_app added')

    df_ip = df.groupby(['ip', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'channel', 'device_count']]
    df_ip = df_ip.sort_values(by='device_count')
    df_ip['ip_channel_rank'] = df_ip['device_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_channel_rank'])
    df_ip['ip_channel_rank'] /= max_ip_rank
    df = df.merge(df_ip, how='left', on=['ip', 'channel'])
    del df_ip
    print('ip_channel added')

    df = df.sort_values(by=['app', 'channel'])

    df_ip = df.groupby(['app', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['app', 'channel', 'device_count']]
    df_ip = df_ip.sort_values(by='device_count')
    df_ip['app_channel_rank'] = df_ip['device_count'].rank(method='dense')
    max_ip_rank = max(df_ip['app_channel_rank'])
    df_ip['app_channel_rank'] /= max_ip_rank
    df = df.merge(df_ip, how='left', on=['app', 'channel'])
    del df_ip
    print('app_channel added')

    df = df.sort_values(by=['ip', 'app', 'channel'])
    df_ip = df.groupby(['ip', 'app', 'channel']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip','app', 'channel', 'device_count']]
    df_ip = df_ip.sort_values(by='device_count')
    df_ip['ip_app_channel_rank'] = df_ip['device_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_app_channel_rank'])
    df_ip['ip_app_channel_rank'] /= max_ip_rank
    df = df.merge(df_ip, how='left', on=['ip', 'app', 'channel'])
    del df_ip
    print('app_channel added')

    df_ip = df.groupby(['ip']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'app_count']]
    df_ip = df_ip.sort_values(by='app_count')
    df_ip['ip_rank'] = df_ip['app_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_rank'])
    df_ip['ip_rank'] /= max_ip_rank

    df = df.merge(df_ip, how='left', on='ip')
    del df_ip
    print('ip added')

    df_app = df.groupby(['app']).count().add_suffix('_count').reset_index()
    df_app = df_app[['app', 'channel_count']]
    df_app = df_app.sort_values(by='channel_count')
    df_app['app_rank'] = df_app['channel_count'].rank(method='dense')
    max_app_rank = max(df_app['app_rank'])
    df_app['app_rank'] /= max_app_rank

    df = df.merge(df_app, how='left', on='app')
    del df_app
    print('app added')

    df_channel = df.groupby(['channel']).count().add_suffix('_count').reset_index()
    df_channel = df_channel[['channel', 'app_count']]
    df_channel = df_channel.sort_values(by='app_count')
    df_channel['channel_rank'] = df_channel['app_count'].rank(method='dense')
    max_channel_rank = max(df_channel['channel_rank'])
    df_channel['channel_rank'] /= max_channel_rank

    df = df.merge(df_channel, how='left', on='channel')
    del df_channel
    print('channel added')

    return df


def train_l1_models(sample_size = 10000000, num_of_chunks = 15):
    models = []
    starting_columns = []
    train_raw = pd.read_csv(path + "train.csv")
    for i in range(num_of_chunks):
        train = train_raw.loc[(i)*sample_size:(i  + 1)*sample_size].copy()
        if 'click_time' not in starting_columns:
            starting_columns =  train.columns
        else:
            train.columns = starting_columns
        print('[{0}] Finished to load data, {1}'.format(time.time() - start_time, i))
        train = preproccess_df(train)
        y = train['is_attributed']
        train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


        print('[{}] Start LGBM Training'.format(time.time() - start_time))
        MAX_ROUNDS = 250

        x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.25, shuffle=True)

        dtrain = lgb.Dataset(x1, label=y1)
        dval = lgb.Dataset(x2, label=y2, reference=dtrain)

        model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                          early_stopping_rounds=50,
                          verbose_eval=10)

        print('[{0}] Finish LGBM Training, {1}'.format(time.time() - start_time, 1))

        with open(path + 'model_{0}.plk'.format(i), 'wb') as infile:
            pickle.dump(model, infile)

        del train
        models.append(model)
    l2_input = train_raw.loc[150000000:]
    l2_input.columns = starting_columns
    return models, l2_input


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

def chunk_predictions():
    pass

l1_test = pd.DataFrame()
l1_train = pd.DataFrame()
sub = pd.DataFrame()

models, train_2 = train_l1_models()
#models,train_2 = load_models()

train_2 = preproccess_df(train_2)
y = train_2['is_attributed']
train_2.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

for count, i in enumerate(models):
    l1_train['is_attributed_{0}'.format(count)] = i.predict(train_2, num_iteration=i.best_iteration or MAX_ROUNDS)

x1, x2, y1, y2 = model_selection.train_test_split(l1_train, y, test_size=0.2, shuffle=True)
dtrain = lgb.Dataset(x1, label=y1)
dval = lgb.Dataset(x2, label=y2, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                          early_stopping_rounds=50, verbose_eval=10)


test = pd.read_csv(path + "test.csv")
test = preproccess_df(test)
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

for count, i in enumerate(models):
    l1_test['is_attributed_{0}'.format(count)] = i.predict(test, num_iteration=i.best_iteration or MAX_ROUNDS)

sub['is_attributed'] = model.predict(l1_test, num_iteration=model.best_iteration or MAX_ROUNDS)
sub.to_csv('lgb_sub.csv', index=False)