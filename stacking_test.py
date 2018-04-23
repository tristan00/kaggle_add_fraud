import pandas as pd
import gc
import xgboost
import lightgbm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import os
import pickle
import copy
from  Stack_main import Stack, LGBMRegressorModel, Node, AdaBoostClassifierModel, RandomForestClassifierModel, ExtraTreesClassifierModel, GradientBoostingClassifierModel, DecisionTreeClassifierModel, XGBoosterModel


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def main():
    num_of_stacks = 50
    output_dir = create_directory(path + 'trained_stacks')

    models = []

    with open('100.plk', 'rb') as infile:
        model_stack = pickle.load(infile)

    df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\fraud\proccessed_train2.csv', sep='|')
    for i in range(num_of_stacks):
        print('training', i)
        temp_stack = copy.deepcopy(model_stack)

        df_pos = df.loc[df['is_attributed'] == 1]
        df_neg = df.loc[df['is_attributed'] == 0]
        df_neg = df_neg.sample(n=df_pos.shape[0])
        gc.collect()
        df_concat = pd.concat([df_pos, df_neg])
        df_concat = df_concat.sample(frac=1)

        y = df_concat['is_attributed']
        x = df_concat.drop(['is_attributed', 'attributed_time'], axis=1)
        x = x.as_matrix()
        y = y.as_matrix()

        temp_stack.load_models()
        temp_stack.train(x, y)
        with open(output_dir + '/{0}.plk', 'wb') as outfile:
            pickle.dump(temp_stack, outfile)
        del temp_stack
        gc.collect()

    sub = pd.DataFrame()
    df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\fraud\proccessed_test2.csv', sep='|')
    sub['click_id'] = df['click_id']
    df.drop('click_id', axis = 1, inplace=True)

    x = df.as_matrix()
    for i in range(num_of_stacks):
        print('predicting', i)
        with open(output_dir + '/{0}.plk', 'rb') as infile:
            model_stack = pickle.load(infile)

        sub[i] = model_stack.predict(x)
        del model_stack
        gc.collect()

    sub['is_attributed'] = sub.apply(lambda x: sum([x[i] for i in range(num_of_stacks)])/num_of_stacks, axis = 1)
    sub.drop([i for i in range(num_of_stacks)], axis = 1, inplace = True)
    sub.to_csv('sub.csv', index = False)


if __name__ == '__main__':
    main()
