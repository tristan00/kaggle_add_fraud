import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network

file_locs = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

#add features
def preproccess_df(df):
    return df

# you can test on sample 20 mil records, easier to manage memory for testing, final solution should use everything
def get_training_set():
    df = pd.read_csv(file_locs + 'train.csv', nrows=10000000)
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
    sub = pd.DataFrame()
    test = pd.read_csv(file_locs + "test.csv")
    test = preproccess_df(test)
    sub['click_id'] = test['click_id']
    test.drop(['click_id', 'click_time'], axis=1, inplace=True)
    sub['is_attributed'] = clf.predict(test)
    sub.to_csv('lgb_sub.csv', index=False)


if __name__ == '__main__':
    main()
