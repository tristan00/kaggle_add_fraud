#import dask.dataframe as dd
import pandas as pd
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


from sklearn.preprocessing import OneHotEncoder


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 2000

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


def print_sample():
    df = pd.read_csv(path + "train.csv", nrows=50000)
    df = preproccess_df(df)
    positives = df.loc[df['is_attributed'] == 1]
    negatives = df.loc[df['is_attributed'] == 0]
    print(negatives.shape[0]/positives.shape[0])
    negatives = negatives.sample(n=max(positives.shape))
    output = pd.concat([positives, negatives])
    output.to_csv('sample.csv')


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

def preproccess_df(df):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
    df['click_hour'] /= 24
    df['click_day'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
    df['click_day'] /= 31
    # df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)
    # df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
    print('time added', df.shape)

    possible_names = ['ip', 'device', 'os', 'channel', 'click_hour', 'click_day']

    for l in range(len(possible_names)):
        combinations = itertools.combinations(possible_names, l+1)
        for i in combinations:
            if len(set(['click_hour', 'click_day']) & set(i)) != 1 and len(set(['click_hour', 'click_day']) | set(i)) > 2:
                df = rank_df(df, list(i), '_'.join(i)+'_rank')


    df = df.drop(['ip', 'device', 'os', 'channel','click_time', 'counting_column', 'click_day'], axis=1)
    #df.drop(['counting_column'], axis=1, inplace=True)
    return df

def get_nn(df):
    model = Sequential()
    model.add(Dense(64, activation='elu', input_shape=(df.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.RMSprop(),
                 metrics=[auc])
    return model


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def train_l1_models():
    chunk_size = 30000000
    num_of_chunks = 3
    models = []

    train_raw = pd.read_csv(path + "train.csv", nrows=2, dtype=init_dtype)
    starting_columns = train_raw.columns

    val = pd.read_csv(path + "train.csv", skiprows=chunk_size*num_of_chunks, nrows=chunk_size, dtype=init_dtype)
    val.columns = starting_columns
    val = preproccess_df(val)
    y_val = val['is_attributed']
    val.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

    for i in range(num_of_chunks):
        train_raw = pd.read_csv(path + "train.csv", nrows=chunk_size, skiprows=i*chunk_size, dtype=init_dtype)
        train_raw.columns = starting_columns
        print('[{0}] Finished to load data'.format(time.time() - start_time))
        train = preproccess_df(train_raw)
        y_train = train['is_attributed']
        train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
        # print('[{}] Start LGBM Training'.format(time.time() - start_time))
        # dtrain = lgb.Dataset(train, label=y_train)
        # dval = lgb.Dataset(val, label=y_val, reference=dtrain)
        # lgbm_model1 = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
        #                   early_stopping_rounds=50, verbose_eval=10)
        #
        # print('[{0}] Finish LGBM Training, {1}'.format(time.time() - start_time, 1))
        # with open(path + 'l1/light_gbm1_{0}.plk'.format(i), 'wb') as infile:
        #     pickle.dump(lgbm_model1, infile)
        # del lgbm_model1

        x3 = train.as_matrix()
        y3 = np.expand_dims(y_train.as_matrix(), 1)
        x4 = val.as_matrix()
        y4 = np.expand_dims(y_val.as_matrix(), 1)

        y3 = keras.utils.to_categorical(y3, 2)
        y4 = keras.utils.to_categorical(y4, 2)
        print(x3.shape, y3.shape)

        nn_model = get_nn(x3)
        nn_model.fit(x3, y3, epochs=5, class_weight=class_weight, verbose=0, batch_size=20000)
        print('nn trained:', nn_model.evaluate(x4,y4, verbose=0))
        nn_model.save(path + 'l1/model_nn_{0}.h5'.format(i))
        del nn_model

        gb = ensemble.GradientBoostingClassifier()
        gb.fit(train,y_train)
        print('gb', gb.score(val, y_val))
        with open(path + 'l1/gb_{0}.plk'.format(i), 'wb') as infile:
            pickle.dump(gb, infile)
        del gb

        ada = ensemble.AdaBoostClassifier()
        ada.fit(train, y_train)
        print('ada', ada.score(val, y_val))
        with open(path + 'l1/ada_{0}.plk'.format(i), 'wb') as infile:
            pickle.dump(ada, infile)

        del ada

        rf = ensemble.RandomForestClassifier(class_weight=class_weight,n_jobs=-1)
        rf.fit(train, y_train)
        print('rf', rf.score(val, y_val))
        with open(path + 'l1/rf_{0}.plk'.format(i), 'wb') as infile:
            pickle.dump(rf, infile)

        del rf

        et = ensemble.ExtraTreesClassifier(class_weight=class_weight,n_jobs=-1)
        et.fit(train, y_train)
        print('et', et.score(val, y_val, ))
        with open(path + 'l1/et_{0}.plk'.format(i), 'wb') as infile:
            pickle.dump(et, infile)

        del et

        # k = KNeighborsClassifier(n_jobs=-1)
        # k.fit(train, y_train)
        # print('k', k.score(val, y_val))
        # with open(path + 'l1/k_{0}.plk'.format(i), 'wb') as infile:
        #     pickle.dump(k, infile)
        #
        # del k
        #
        # r = RadiusNeighborsClassifier(n_jobs=-1)
        # r.fit(train, y_train)
        # print('r', r.score(val, y_val))
        # with open(path + 'l1/r_{0}.plk'.format(i), 'wb') as infile:
        #     pickle.dump(r, infile)
        #
        # del k

    # svc = SVC(class_weight=class_weight)
    # svc.fit(train, y_train)
    # print('k', svc.score(val, y_val))
    # with open(path + 'l1/svc.plk', 'wb') as infile:
    #     pickle.dump(svc, infile)
    return chunk_size*(num_of_chunks + 1)



def load_models():
    model_locs = glob.glob(path + '*.plk')
    models = []
    for m in model_locs:
        with open(m, 'rb') as infile:
            models.append(pickle.load(infile))
    train_raw = pd.read_csv(path + "train.csv", skiprows=150000000, dtype=init_dtype)
    train_raw.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
       'is_attributed']
    return models, train_raw


def get_l1_predictions(start_index):
    model_locs = glob.glob(path + 'l1/*.plk')
    model_locs_nn = glob.glob(path + 'l1/*.h5')

    df = pd.read_csv(path + "train.csv", skiprows=start_index, dtype=init_dtype)
    #df = pd.read_csv(path + "train.csv", nrows=start_index, dtype=init_dtype)
    df.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
                         'is_attributed']
    df = preproccess_df(df)
    y = df['is_attributed']
    df.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    chunk_result = pd.DataFrame()

    for count, m in enumerate(model_locs):
        with open(m, 'rb') as infile:
            model = pickle.load(infile)
        try:
            chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df,num_iteration=model.best_iteration or MAX_ROUNDS)
        except:
            chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df)
    for count, m in enumerate(model_locs_nn):
        model = keras.models.load_model(m, custom_objects={'auc':auc})
        chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df)[:,0]

    print(chunk_result.shape, chunk_result.columns)
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
        try:
            chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df,num_iteration=model.best_iteration or MAX_ROUNDS)
        except:
            chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df)
    for count, m in enumerate(model_locs_nn):
        model = keras.models.load_model(m, custom_objects={'auc':auc})
        chunk_result[os.path.basename(m).split('.')[0]] = model.predict(df)[:,0]

    print(chunk_result.shape, chunk_result.columns)
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

        start = train_l1_models()
        #start = 120000000
        x, y = get_l1_predictions(start)
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

    test = pd.read_csv(path + "test.csv", dtype=init_dtype)
    test = preproccess_df(test)
    sub['click_id'] = test['click_id']
    test.drop('click_id', axis=1, inplace=True)

    x = chunk_test_predictions(test)

    sub['is_attributed'] = model.predict(x, num_iteration=model.best_iteration or MAX_ROUNDS)
    sub.to_csv('lgb_sub.csv', index=False)


if __name__ == '__main__':
    main()
    #print_sample()