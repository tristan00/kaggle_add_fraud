import toolz
import dask.dataframe as dd
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
import datetime
import itertools

file_locs = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

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
    df = df.merge(df2, on=columns)
    return df

#add features
def preproccess_df(df):
    df['counting_column'] = 1

    df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour, meta={})
    df['click_hour'] /= 24
    df['click_day'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
    df['click_day'] /= 31
    df['click_minute'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').minute)
    df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
    print('time added', df.shape)

    df['os'] /= 1.0
    df['device'] /= 1.0

    possible_names = ['ip', 'device', 'os', 'channel', 'click_hour', 'click_day']

    for l in range(len(possible_names)):
        combinations = itertools.combinations(possible_names, l+1)
        for i in combinations:
            df = rank_df(df, list(i), '_'.join(i)+'_rank')
            print(df.shape, df.columns)

    df = df.drop(['ip', 'device', 'os', 'channel','click_time', 'counting_column'], axis=1)
    #df.drop(['counting_column'], axis=1, inplace=True)
    return df

# you can test on sample 20 mil records, easier to manage memory for testing, final solution should use everything
def get_training_set():
    df = dd.read_csv(file_locs + 'train.csv').head(n=1000)
    return df


def main():
    df = get_training_set()
    y = df['is_attributed']
    df.drop(['is_attributed', 'attributed_time', 'click_time'], axis=1, inplace=True)
    x1, x2, y1, y2 = model_selection.train_test_split(df, y, test_size=0.1, shuffle=True)
    clf = linear_model.LogisticRegression()

    # other possible models
    # clf = ensemble.RandomForestClassifier()
    # clf = ensemble.AdaBoostClassifier()
    # clf = neighbors.KNeighborsClassifier()
    clf = neural_network.MLPClassifier()

    clf.fit(x1,y1)
    print(clf.score(x2, y2))

    #predict output
    test = dd.read_csv(file_locs + "test.csv")
    test = preproccess_df(test)
    sub = test['click_id']
    #sub['click_id'] = test['click_id']
    test.drop(['click_id', 'click_time'], axis=1, inplace=True)
    sub['is_attributed'] = clf.predict(test)
    sub.to_csv('lgb_sub.csv', index=False)


if __name__ == '__main__':
    main()
