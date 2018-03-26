import pandas as pd

init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }

path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'


df = pd.read_csv(path + 'train.csv', dtype=init_dtype, nrows=1000000)
df.to_csv('sample.csv')


