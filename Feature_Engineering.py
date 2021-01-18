# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from itertools import product, starmap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



df_train.index = pd.to_datetime(df_train['date'])
df_train.drop('date', axis=1, inplace=True)
df_test.index = pd.to_datetime(df_test['date'])
df_test.drop('date', axis=1, inplace=True)



### adding some useful functions

def storeitems():
    
    return product(range(1,51), range(1,11))




def storeitems_column_names():
    
    return list(starmap(lambda i,s: f'item_{i}_store_{s}_sales', storeitems()))




def sales_by_storeitem(df):
    
    ret = pd.DataFrame(index=df.index.unique())
    
    for i, s in storeitems():
        ret[f'item_{i}_store_{s}_sales'] = df[(df['item'] == i) & (df['store'] == s)]['sales'].values
        
    return ret




df_train = sales_by_storeitem(df_train)


# for test set, we just fill y values with zeros (they won't be used anyway)
df_test['sales'] = np.zeros(df_test.shape[0])
df_test = sales_by_storeitem(df_test)


print(df_train.info())
print(df_test.info())



###  combine data to prepare it for the model, and later split back into train and test set

###  make sure all column names are the same and in the same order


col_names = list(zip(df_test.columns, df_train.columns))
for cn in col_names:
    
    assert cn[0] == cn[1]
    

df_test['is_test'] = np.repeat(True, df_test.shape[0])
df_train['is_test'] = np.repeat(False, df_train.shape[0])
df_total = pd.concat([df_train, df_test])
df_total.info()


weekday_df = pd.get_dummies(df_total.index.weekday, prefix='weekday')
weekday_df.index = df_total.index


month_df = pd.get_dummies(df_total.index.month, prefix='month')
month_df.index =  df_total.index


df_total = pd.concat([weekday_df, month_df, df_total], axis=1)

assert df_total.isnull().any().any() == False




### additionally append sales from previous day to each row, which we will then use as input data."""


def shift_series(series, days):
    
    return series.transform(lambda x: x.shift(days))



def shift_series_in_df(df, series_names=[], days_delta=90):
    
    ret = pd.DataFrame(index=df.index.copy())
    str_sgn = 'future' if np.sign(days_delta) < 0 else 'past'
    
    for sn in series_names:
        ret[f'{sn}_{str_sgn}_{np.abs(days_delta)}'] = shift_series(df[sn], days_delta)
        
    return ret


    
def stack_shifted_sales(df, days_deltas=[1, 90, 360]):
    
    names = storeitems_column_names()
    dfs = [df.copy()]
    
    for delta in days_deltas:
        shifted = shift_series_in_df(df, series_names=names, days_delta=delta)
        dfs.append(shifted)
        
    return pd.concat(dfs, axis=1, copy=False)



df_total = stack_shifted_sales(df_total, days_deltas=[1])
df_total.dropna(inplace=True)

print(df_total.info())





### We need to make sure that stacked and not-stacked sales columns appar in the same order. We do this 
### by sorting the names as strings which works fine because we only need 1 past day for the network


sales_cols = [col for col in df_total.columns if '_sales' in col and '_sales_' not in col]
stacked_sales_cols = [col for col in df_total.columns if '_sales_' in col]
other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

sales_cols = sorted(sales_cols)
stacked_sales_cols = sorted(stacked_sales_cols)

new_cols = other_cols + stacked_sales_cols + sales_cols

df_total = df_total.reindex(columns=new_cols)

assert df_total.isnull().any().any() == False


### save for future use
df_total.to_csv('df_total.csv',encoding = 'utf-8', index=False)

### scale our data

scaler = MinMaxScaler()

cols_to_scale = [col for col in df_total.columns if 'weekday' not in col and 'month' not in col]
scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
df_total[cols_to_scale] = scaled_cols

df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)
df_test = df_total[df_total['is_test'] == True].drop('is_test', axis=1)


X_cols_stacked = [col for col in df_train.columns if '_past_' in col]
X_cols_caldata = [col for col in df_train.columns if 'weekday_' in col or 'month_' in col or 'year' in col]
X_cols = X_cols_stacked + X_cols_caldata


X = df_train[X_cols]

X_colset = set(X_cols)
y_cols = [col for col in df_train.columns if col not in X_colset]

y = df_train[y_cols]


X.to_csv('final_predictors.csv', encoding = 'utf-8', index=False)
y.to_csv('final_dependent.csv', encoding = 'utf-8', index=False)



### save the scaled columns as well in order to use them for model development
pd.DataFrame(cols_to_scale[1:]).to_csv('cols_to_scale.csv',encoding = 'utf-8', index=False)
pd.DataFrame(y_cols).to_csv('y_cols.csv',encoding = 'utf-8', index=False)




