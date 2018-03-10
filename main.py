import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import ensemble
from sklearn import svm
import numpy as np
import pickle
import lightgbm
from sklearn.model_selection import train_test_split

file_loc = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'


def preproccess_df(df):
    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    df_ip = df.groupby(['ip']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'app_count']]
    df_ip = df_ip.sort_values(by='app_count')
    df_ip['ip_rank'] = df_ip['app_count'].rank(method='dense')
    max_ip_rank = max(df_ip['ip_rank'])
    df_ip['ip_rank'] /= max_ip_rank

    df_app = df.groupby(['app']).count().add_suffix('_count').reset_index()
    df_app = df_app[['app', 'channel_count']]
    df_app = df_app.sort_values(by='channel_count')
    df_app['app_rank'] = df_app['channel_count'].rank(method='dense')
    max_app_rank = max(df_app['app_rank'])
    df_app['app_rank'] /= max_app_rank

    df_channel = df.groupby(['channel']).count().add_suffix('_count').reset_index()
    df_channel = df_channel[['channel', 'app_count']]
    df_channel = df_channel.sort_values(by='app_count')
    df_channel['channel_rank'] = df_channel['app_count'].rank(method='dense')
    max_channel_rank = max(df_channel['channel_rank'])
    df_channel['channel_rank'] /= max_channel_rank

    df = df.merge(df_ip, how='left', on='ip')
    df = df.merge(df_app, how='left', on='app')
    df = df.merge(df_channel, how='left', on='channel')
    y = df['is_attributed'].as_matrix()
    x = df[['click_hour', 'click_minute', 'channel_rank', 'app_rank', 'ip_rank', 'device']].as_matrix()
    return x, y

def get_data():
    train_file_loc = file_loc + 'train.csv'

    df = pd.read_csv(train_file_loc, nrows=100000)
    x, y = preproccess_df(df)
    return x, y


def test_predict(df):
    train_file_loc = file_loc + 'train.csv'

    df = pd.read_csv(train_file_loc, nrows=100000)
    x, y = preproccess_df(df)
    return x, y


def predict(x, y):
    test_file_loc = file_loc + 'test.csv'
    rf = lightgbm.LGBMClassifier()
    rf.fit(x, y)
    del x
    del y

    df = pd.read_csv(test_file_loc)
    click_ids = list(df['click_id'].tolist())

    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)

    df_ip = df.groupby(['ip']).count().add_suffix('_count').reset_index()
    df_ip = df_ip[['ip', 'app_count']]
    df_ip = df_ip.sort_values(by='app_count')
    df_ip['ip_rank'] = df_ip['app_count'].rank(method = 'dense')
    max_ip_rank = max(df_ip['ip_rank'])
    df_ip['ip_rank'] /= max_ip_rank

    df_app = df.groupby(['app']).count().add_suffix('_count').reset_index()
    df_app = df_app[['app', 'channel_count']]
    df_app = df_app.sort_values(by='channel_count')
    df_app['app_rank'] = df_app['channel_count'].rank(method = 'dense')
    max_app_rank = max(df_app['app_rank'])
    df_app['app_rank'] /= max_app_rank

    df_channel = df.groupby(['channel']).count().add_suffix('_count').reset_index()
    df_channel = df_channel[['channel', 'app_count']]
    df_channel = df_channel.sort_values(by='app_count')
    df_channel['channel_rank'] = df_channel['app_count'].rank(method = 'dense')
    max_channel_rank = max(df_channel['channel_rank'])
    df_channel['channel_rank'] /= max_channel_rank

    df = df.merge(df_ip, how = 'left', on='ip')
    df = df.merge(df_app, how='left', on='app')
    df = df.merge(df_channel, how='left', on='channel')

    x = df[['click_hour', 'click_minute', 'channel_rank', 'app_rank', 'ip_rank', 'device']].as_matrix()
    del df
    y = rf.predict(x)

    prediction_list = list(y)

    output_list = []

    for i, j in zip(click_ids, prediction_list):
        output_list.append({'click_id':i, 'is_attributed':j})

    o_df = pd.DataFrame.from_dict(output_list)
    o_df = o_df[['click_id', 'is_attributed']]
    o_df.to_csv('output.csv', index = False)


def main():
    x, y = get_data()
    predict(x, y)

if __name__ == '__main__':
    main()