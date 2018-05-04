import time
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, mean_squared_error
from keras.models import Model, load_model, save_model
import lightgbm as lgb
import statistics
import datetime
import math
import tensorflow as tf
import xgboost as xgb
import h5py
import statistics
import glob
import pickle
import catboost
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
import pickle
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt.space import Integer, Real
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

start_time = time.time()
MAX_ROUNDS = 5000

params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 12,
    'scale_pos_weight':400
}


x_params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic',
          'scale_pos_weight':400,
          'eval_metric': 'auc',
          'nthread':12}

init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }



def tune_lgbm():
    clf = lgb.LGBMClassifier()
    df = pd.read_csv(path + "train.csv", dtype=init_dtype)
    train = preproccess_df(df, train = False)
    gc.collect()

    train = train.sample(n = 30000000)

    train_y = train['is_attributed']
    train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

    input_parameter_dict = {'num_leaves': [15, 31, 63, 128],
                            'min_data_in_leaf': [50, 100, 150],
                            'learning_rate':[.05, .1, .2],
                            'max_bin':[255, 512, 1024],
                            'boosting_type':['gbdt', 'rf'],
                            'num_iterations': [100, 150],
                            'feature_fraction':[.8, .95],
                            'bagging_fraction':[.8, .95],
                            'bagging_freq':[2, 4],
                            'metric':['auc'],
                            'scale_pos_weight':[400]
    }

    random_search = RandomizedSearchCV(clf, param_distributions=input_parameter_dict,
                                       n_iter=100, verbose=3)

    start = time.time()
    random_search.fit(train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), 50))
    report(random_search.cv_results_)




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


def get_nn(df):
    model = Sequential()
    model.add(Dense(128, activation='elu', input_shape=(df.shape[1],)))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(1, activation='elu'))
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
        model = load_model(path + 'auto3.h5')
    except:
        model = get_autoencoder(nn_df)
        model.fit(nn2_df, nn2_df, epochs=1)
        model.save(path + 'auto3.h5')

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
    gc.collect()

    #TODO: vectorize
    output_array = []
    for count, ((_, i), (_, j)) in enumerate(zip(df_pre.iterrows(), df2.iterrows())):
        output_array.append(mean_squared_error(i, j))
        if count % 100000 == 0:
            print('predicting outliers', count, time.time() - start_time)
    df['nn_score'] = np.array(output_array)


    return df


def rank_df(df, columns, name):
    print('rank', columns)
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
    print('count', columns)
    df[name] = df.groupby(columns).cumcount()
    df[name] = df[name].astype('uint16')
    return df

def time_since_last(df, columns, name):
    print('time_since_last', columns)
    df[name] = df.groupby(columns)['click_time'].diff()
    df[name] = df[name].fillna(-1)
    df[name] = df[name].astype(int)
    mean_std = df.groupby(columns)[name].agg([np.median, np.std]).reset_index()
    df = df.merge(mean_std, on = columns)
    df[name] = df[name].fillna(-1)
    df[name + '_std'] = df['std'].fillna(-1)
    df[name + '_median'] = df['median'].fillna(-1)
    df = df.drop(['median', 'std'], axis = 1)
    return df

def time_till_next(df, columns, name):
    print('time_till_next', columns)
    df[name] = df.groupby(columns)['click_time'].transform(
        lambda x: x.diff().shift(-1))
    df[name] = df[name].fillna(-1)
    df[name] = df[name].astype(int)
    mean_std = df.groupby(columns)[name].agg([np.median, np.std]).reset_index()
    df = df.merge(mean_std, on = columns)
    df[name] = df[name].fillna(-1)
    df[name + '_std'] = df['std'].fillna(-1)
    df[name + '_median'] = df['median'].fillna(-1)
    df = df.drop(['median', 'std'], axis = 1)
    return df


def add_features(df):
    rank_list = [['ip', 'os', 'device', 'channel', 'click_day', 'click_hour'],
                 ['ip', 'device', 'os', 'click_day', 'click_hour'],
                 ['ip', 'os', 'device', 'channel', 'click_day'],
                 ['ip', 'device', 'os', 'click_day']]
    for i in rank_list:
        df = count_df(df, list(i), '_'.join(i) + '_count')
        gc.collect()

    rank_list = [['ip', 'click_day'],
                 ['ip', 'device', 'channel', 'click_day'],
                 ['ip', 'device', 'channel', 'click_day', 'click_hour'],
                 ['ip', 'device', 'os', 'click_day'],
                 ['ip', 'device', 'os', 'click_day', 'click_hour'],
                 ['device', 'os', 'channel', 'app', 'click_day'],
                 ['device', 'channel', 'click_day'],
                 ['app', 'channel', 'click_day', 'click_hour'],
                 ['ip', 'device', 'os', 'app', 'click_day', 'click_hour'],
                 ['ip', 'device', 'os', 'channel', 'click_day', 'click_hour']]
    for i in rank_list:
        df = rank_df(df, list(i), '_'.join(i) + '_rank')
        gc.collect()

    df['device'] = df['device'].astype('int')
    df['os'] = df['os'].astype('int')
    df['channel'] = df['channel'].astype('int')
    df['click_hour'] = df['click_hour'].astype('int')
    df = df.drop(['ip','click_time', 'counting_column', 'click_day', 'click_day'], axis=1)

    return df

def preproccess_df(df, train = True, val_day = 7):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['datetime'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['click_day'] = df['datetime'].apply(lambda x: x.day).astype('uint8')
    df['click_hour'] = df['datetime'].apply(lambda x: x.hour).astype('uint8')
    df['click_second'] = df['datetime'].apply(lambda x: x.second).astype('uint8')
    df['click_minute'] = df['datetime'].apply(lambda x: x.minute).astype('uint8')
    #df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['datetime'].apply(lambda x: x.timestamp())
    df = df.drop(['datetime'], axis = 1)

    df = df.sort_values(by=['click_time'])

    if train:
        train = df[df['click_day'] != val_day]
        val = df[df['click_day'] == val_day]
        #
        # if min(train.shape) == 0:
        #     train = val.sample(frac = .5)
        print(train.shape, val.shape)
        train = add_features(train)
        val = add_features(val)

        return train, val
    if not train:
        df['click_hour'] = df['click_hour'].apply(lambda x: 5 if x == 6 else 10 if x == 11 else 14)
        return add_features(df)


    print(df.shape)


    #df.drop(['counting_column'], axis=1, inplace=True)
    return df


def train_nn(train_x, train_y, test_x, test_y):

    if 'device' in train_x.columns:
        train_x_copy = train_x.drop(['device', 'os', 'channel', 'click_hour'], axis = 1)
        test_x_copy = test_x.drop(['device', 'os', 'channel', 'click_hour'], axis = 1)
    else:
        train_x_copy = train_x.copy()
        test_x_copy = test_x.copy()

    nn = get_nn(train_x_copy)
    nn.fit(train_x_copy, train_y)
    return nn


def train_lgbm(train_x, train_y, test_x, test_y):
    train_x = train_x.copy()
    train_y = train_y.copy()
    if 'device' in train_x.columns:
        train_x['device'] = train_x['device'].astype("category")
        train_x['os'] = train_x['os'].astype("category")
        train_x['channel'] = train_x['channel'].astype("category")
        train_x['click_hour'] = train_x['click_hour'].astype("category")

        test_x['device'] = test_x['device'].astype("category")
        test_x['os'] = test_x['os'].astype("category")
        test_x['channel'] = test_x['channel'].astype("category")
        test_x['click_hour'] = test_x['click_hour'].astype("category")

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(test_x, label=test_y, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')
    return model


def train_catboost(train_x, train_y, test_x, test_y):
    train_x = train_x.copy()
    train_y = train_y.copy()

    train_x['device'] = train_x['device'].astype("category")
    train_x['os'] = train_x['os'].astype("category")
    train_x['channel'] = train_x['channel'].astype("category")
    train_x['click_hour'] = train_x['click_hour'].astype("category")

    test_x['device'] = test_x['device'].astype("category")
    test_x['os'] = test_x['os'].astype("category")
    test_x['channel'] = test_x['channel'].astype("category")
    test_x['click_hour'] = test_x['click_hour'].astype("category")


    model = catboost.CatBoostRegressor()
    model.fit(train_x, train_y, eval_set=(test_x, test_y),
              cat_features=np.array([test_x.columns.get_loc('device'), test_x.columns.get_loc('os'), test_x.columns.get_loc('channel'), test_x.columns.get_loc('click_hour')]))
    gc.collect()
    return model


def train_xgb(train_x, train_y, test_x, test_y):
    print(train_x.columns)
    train_x = train_x.copy()
    train_y = train_y.copy()
    if 'device' in train_x.columns:
        train_x['device'] = train_x['device'].astype("float")
        train_x['os'] = train_x['os'].astype("float")
        train_x['channel'] = train_x['channel'].astype("float")
        train_x['click_hour'] = train_x['click_hour'].astype("float")

        test_x['device'] = test_x['device'].astype("float")
        test_x['os'] = test_x['os'].astype("float")
        test_x['channel'] = test_x['channel'].astype("float")
        test_x['click_hour'] = test_x['click_hour'].astype("float")

        dtrain = xgb.DMatrix(train_x, train_y)
        dvalid = xgb.DMatrix(test_x, test_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        model = xgb.train(x_params, dtrain, 30, watchlist)
        gc.collect()
    else:
        dtrain = xgb.DMatrix(train_x, train_y)
        dvalid = xgb.DMatrix(test_x, test_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        model = xgb.train(x_params, dtrain, 30, watchlist)
        gc.collect()
    return model


def predict_nn(model, x, sub):
    x = x.copy()
    if 'device' in x.columns:
        x_copy = x.drop(['device', 'os', 'channel', 'click_hour'], axis = 1)
    else:
        x_copy = x.copy()

    sub['is_attributed_n'] = model.predict(x_copy)
    return sub


def predict_lgbm(model, x, sub):
    x = x.copy()
    if 'device' in x.columns:
        x['device'] = x['device'].astype("category")
        x['os'] = x['os'].astype("category")
        x['channel'] = x['channel'].astype("category")
        x['click_hour'] = x['click_hour'].astype("category")

        sub['is_attributed'] = model.predict(x, num_iteration=model.best_iteration or MAX_ROUNDS)
    else:
        sub['is_attributed'] = model.predict(x, num_iteration=model.best_iteration or MAX_ROUNDS)
    return sub


def predict_catboost(model, x, sub):
    x = x.copy()
    x['device'] = x['device'].astype("category")
    x['os'] = x['os'].astype("category")
    x['channel'] = x['channel'].astype("category")
    x['click_hour'] = x['click_hour'].astype("category")

    sub['is_attributed_c'] = model.predict(x)
    return sub


def predict_xgb(model, x, sub):
    x = x.copy()
    print(x.columns)
    if 'device' in x.columns:
        x['device'] = x['device'].astype("float")
        x['os'] = x['os'].astype("float")
        x['channel'] = x['channel'].astype("float")
        x['click_hour'] = x['click_hour'].astype("float")
        dtest = xgb.DMatrix(x)
        sub['is_attributed_x'] = model.predict(dtest)
    else:
        dtest = xgb.DMatrix(x)
        sub['is_attributed'] = model.predict(dtest)
    return sub


def optimize_lgbm():

    df = pd.read_csv(path + "train.csv", dtype=init_dtype)
    train = preproccess_df(df, train = False)
    gc.collect()
    train = train.sample(n = 50000000)
    gc.collect()
    y = train['is_attributed']
    x = train.drop(['is_attributed', 'attributed_time'], axis=1)
    del train
    gc.collect()
    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))

        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name + "_cv_results.csv")

    gc.collect()
    if 'device' in x.columns:
        x['device'] = x['device'].astype("category")
        x['os'] = x['os'].astype("category")
        x['channel'] = x['channel'].astype("category")
        x['click_hour'] = x['click_hour'].astype("category")
    gc.collect()

    iterations = 100
    bayes_cv_tuner = BayesSearchCV(
        estimator=lgb.LGBMRegressor(
            objective='binary',
            metric='auc',
            n_jobs=6,
            verbose=1,
            categorical_feature='auto'
        ),
        search_spaces={
            'learning_rate': (0.001, 1.0, 'log-uniform'),
            'num_leaves': (2, 300),
            'max_depth': (2, 50),
            'min_data_in_leaf': (1, 50),
            'max_bin': (100, 2000),
            'subsample': (0.01, 1.0, 'uniform'),
            'subsample_freq': (0, 10),
            'colsample_bytree': (0.01, 1.0, 'uniform'),
            'min_child_weight': (1, 10),
            'subsample_for_bin': (100000, 500000),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'scale_pos_weight': (300, 500, 'log-uniform'),
            'n_estimators': (50, 300),
        },
        scoring='roc_auc',
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
        ),

        n_iter=iterations,
        verbose=3,
        refit=True,
    )



    res = bayes_cv_tuner.fit(x, y, callback=status_print)

    print(res.best_params_)



def main_wo_val(val_day = 7, reproccess = True):
    df = pd.read_csv(path + "train.csv", dtype=init_dtype)
    train, val = preproccess_df(df, val_day = val_day)
    gc.collect()

    train = train.sample(frac = 1.0)
    val = val.sample(frac = 1.0)

    train_y = train['is_attributed']
    val_y = val['is_attributed']
    train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
    val.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

    model = train_lgbm(train, train_y, val, val_y)
    gc.collect()

    columns = train.columns
    f_i = model.feature_importance(importance_type='split')
    f_i2 = model.feature_importance(importance_type='gain')

    fi_df = pd.DataFrame(columns=columns)
    fi_df.loc[len(fi_df)] = f_i
    fi_df.loc[len(fi_df)] = f_i2
    fi_df.to_csv('f2.csv', index=False)

    del train, val, train_y, val_y
    gc.collect()

    test1 = pd.read_csv(path + "test.csv", dtype=init_dtype)
    test1 = preproccess_df(test1, train=False)

    sub = pd.DataFrame()
    sub['click_id'] = test1['click_id']
    test1.drop('click_id', axis=1, inplace=True)
    sub = predict_lgbm(model, test1, sub)
    sub.to_csv('lgb_sub{0}.csv'.format(val_day), index=False)


def main_ensemble():
    # for i in range(7, 10):
    #     main_wo_val(val_day=i)
    #     gc.collect()

    #mean
    df = pd.DataFrame()
    for i in range(7, 10):

        next_df = pd.read_csv('lgb_sub{0}.csv'.format(i))
        next_df['is_attributed_{0}'.format(i)] = next_df['is_attributed']
        next_df.drop('is_attributed', axis = 1, inplace=True)
        if min(df.shape) == 0:
            df = next_df.copy()
        else:
            df = df.merge(next_df, how = 'outer', on = 'click_id', suffixes=('', '_' + str(i)))

    df = df.fillna(0)
    df['is_attributed'] = df.apply(lambda x: (x['is_attributed_7'] + x['is_attributed_8'] + x['is_attributed_9'])/3, axis = 1)
    df.drop(['is_attributed_7','is_attributed_8','is_attributed_9'], axis=1, inplace=True)
    df.to_csv('lgb_sub_mean.csv', index=False)


    #median
    df = pd.DataFrame()
    for i in range(7, 10):

        next_df = pd.read_csv('lgb_sub{0}.csv'.format(i))
        next_df['is_attributed_{0}'.format(i)] = next_df['is_attributed']
        next_df.drop('is_attributed', axis = 1, inplace=True)
        if min(df.shape) == 0:
            df = next_df.copy()
        else:
            df = df.merge(next_df, how = 'outer', on = 'click_id', suffixes=('', '_' + str(i)))

    df = df.fillna(0)
    df['is_attributed'] = df.apply(lambda x: statistics.median([x['is_attributed_7'], x['is_attributed_8'], x['is_attributed_9']]), axis = 1)
    df.drop(['is_attributed_7','is_attributed_8','is_attributed_9'], axis=1, inplace=True)
    df.to_csv('lgb_sub_median.csv', index=False)


if __name__ == '__main__':
    # main_ensemble()
    # main_wo_val(reproccess=True)
    optimize_lgbm()
    # optimize_lgbm()





