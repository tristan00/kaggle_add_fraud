import pandas as pd
import datetime

init_dtype = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        }


path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'
#
# df = pd.read_csv(path + 'test.csv', dtype=init_dtype)
# df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
# allowed_hours = set(df['click_hour'])
# del df
# allowed_hours = allowed_hours - set([6 , 11, 15])
# print(allowed_hours)
#
# df = pd.read_csv(path + 'train.csv', dtype=init_dtype)
# print(df.shape)
# df['click_hour'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
# df = df.loc[df['click_hour'].isin(allowed_hours)]
# print(df.shape)
# print(sum(df['is_attributed']))
# #df.to_csv(path + 'train2.csv', index = False)


df = pd.read_csv(path + 'train2.csv', dtype=init_dtype)
print(sum(df['is_attributed']))
print(df.shape)