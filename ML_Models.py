# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings
from scipy.stats import describe
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Conv1D, Input, Dropout, AvgPool1D, Reshape, Concatenate



pd.options.display.max_columns = 12
pd.options.display.max_rows = 24

# disable warnings in Anaconda
warnings.simplefilter('ignore')

# plots inisde jupyter notebook
# %matplotlib inline

sns.set(style='darkgrid', palette='muted')
color_scheme = {
    'red': '#F1637A',
    'green': '#6ABB3E',
    'blue': '#3D8DEA',
    'black': '#000000'
}

# increase default plot size
rcParams['figure.figsize'] = 8, 6



X = pd.read_csv('final_predictors.csv')
y = pd.read_csv('final_dependent.csv')

df_total = pd.read_csv('df_total.csv')

cols_1 = pd.read_csv('cols_to_scale.csv')
cols_2 = pd.read_csv('y_cols.csv')

cols_to_scale = list(cols_1.iloc[:,0])
y_cols = list(cols_2.iloc[:,0])



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
X_valid, y_valid = X_valid.head(90), y_valid.head(90)


###  some transformations in order to input the data into Keras

X_train_vals = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid_vals = X_valid.values.reshape((X_valid.shape[0], 1, X_valid.shape[1]))



### a simple LSTM model for baseline

basic_model = Sequential()
basic_model.add(LSTM(500, input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])))
basic_model.add(Dense(500))
basic_model.compile(loss='mean_absolute_error', optimizer='adam')




### a combination of LSTM and CNN for boosted performance

inputs = Input(shape=(X_train_vals.shape[1], X_train_vals.shape[2]))

### top pipeline

top_lstm = LSTM(500)(inputs)
top_dense = Dense(500, activation='relu')(top_lstm)
top_dropout = Dropout(500)(top_dense)


### bottom pipeline

bottom_dense = Dense(500)(inputs)
bottom_conv1 = Conv1D(
    500, 
    kernel_size=1,
    input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])
)(bottom_dense)
bottom_conv2 = Conv1D(
    1000,
    kernel_size=50,
    padding='same',
    activation='relu'
)(bottom_conv1)
bottom_conv3 = Conv1D(
    500,
    kernel_size=10,
    padding='same',
    activation='relu'
)(bottom_conv2)
bottom_pooling = AvgPool1D(
    pool_size=60, 
    padding='same'
)(bottom_conv3)
bottom_reshape = Reshape(
    target_shape=[500]
)(bottom_conv3)


### concatenate output from both pipelines

final_concat = Concatenate()([top_dropout, bottom_reshape])
final_dense = Dense(500)(final_concat)

# compile and return

complex_model = Model(inputs=inputs, outputs=final_dense)
complex_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])


### now we can fit our models


### the basic one
basic_history = basic_model.fit(
    X_train_vals, 
    y_train.values, 
    epochs=60, 
    batch_size=30,
    validation_data=(X_valid_vals, y_valid.values),
    verbose=2,
    shuffle=False
)

def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

plot_history(basic_history)


### the hybrid one
complex_history = complex_model.fit(
    X_train_vals, 
    y_train.values, 
    epochs=20, 
    batch_size=70,
    validation_data=(X_valid_vals, y_valid.values),
    verbose=2,
    shuffle=False
)

plot_history(complex_history)




### Evaluating model predictions

def model_eval(model, X_test, y_test):
    
    ### Evaluate (step-by-step) model predictions from X_test and return predictions and real values in comparable format.
    
    # prepare data
    sales_x_cols = [col for col in X_test.columns if 'sales' in col]
    sales_x_idxs = [X_test.columns.get_loc(col) for col in sales_x_cols]
    sales_y_cols = [col for col in y_test.columns if 'sales' in col]
    sales_y_idxs = [y_test.columns.get_loc(col) for col in sales_y_cols]
    n_samples = y_test.shape[0]
    y_pred = np.zeros(y_test.shape)
    # iterate
    x_next = X_test.iloc[0].values
    
    for i in range(0, n_samples):
        
        x_arr = np.array([x_next])
        x_arr = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        y_pred[i] = model.predict(x_arr)[0] # input for prediction must be 2d, output is immediately extracted from 2d to 1d
        
        try:
            
            x_next = X_test.iloc[i+1].values
            x_next[sales_x_idxs] = y_pred[i][sales_y_idxs]
        except IndexError:
            
            pass  # this happens on last iteration, and x_next does not matter anymore
        
    return y_pred, y_test.values



def unscale(y_arr, scaler, template_df, toint=False):
    
    ### Unscale array y_arr of model predictions, based on a scaler fitted 
    ### to template_df.
    
    tmp = template_df.copy()
    tmp[y_cols] = pd.DataFrame(y_arr, index=tmp.index)
    tmp[cols_to_scale] = scaler.inverse_transform(tmp[cols_to_scale])
    
    if toint:
        
        return tmp[y_cols].astype(int)
    
    return tmp[y_cols]



def vector_smape(y_pred, y_real):
    nom = np.abs(y_pred-y_real)
    denom = (np.abs(y_pred) + np.abs(y_real)) / 2
    results = nom / denom
    return 100*np.mean(results)  # in percent



y_pred_basic, y_real = model_eval(basic_model, X_valid, y_valid)
y_pred_complex = model_eval(complex_model, X_valid, y_valid)[0]



### this is just for unscaling

scaler = MinMaxScaler()
scaled_cols = scaler.fit_transform(df_total[cols_to_scale])


template_df = pd.concat([X_valid, y_valid], axis=1)
template_df['is_test'] = np.repeat(True, template_df.shape[0])

basic_pred = unscale(y_pred_basic, scaler, template_df, toint=True)
complex_pred = unscale(y_pred_complex, scaler, template_df, toint=True)
real = unscale(y_real, scaler, template_df, toint=True)

basic_smapes = [vector_smape(basic_pred[col], real[col]) for col in basic_pred.columns]
complex_smapes = [vector_smape(complex_pred[col], real[col]) for col in complex_pred.columns]



sns.distplot(basic_smapes, label='Basic model')
sns.distplot(complex_smapes, label='Complex model')
plt.legend(loc='upper right')
plt.savefig('smape_basic_vs_complex.svg')
plt.show()



describe(basic_smapes)
describe(complex_smapes)


### Visualizing model prediction
### plot predictions of 2 models for a sample store and item.


store, item = 1,1
plot_lengths = [7, 30, 90]
rolling_mean_windows = [1, 1, 2]  # in order to make plots more readable

storeitem_col = f'item_{item}_store_{store}_sales'

for pl, mw in zip(plot_lengths, rolling_mean_windows):
    plt.plot(basic_pred[storeitem_col].rolling(mw).mean().values[:pl],
             color_scheme['blue'],
             lw=2,
             label='Basic model prediction')
    plt.plot(complex_pred[storeitem_col].rolling(mw).mean().values[:pl],
             color_scheme['green'],
             lw=2, 
             label='Complex model prediction')
    plt.plot(real[storeitem_col].rolling(mw).mean().values[:pl],
             color_scheme['black'],
             lw=2, 
             label='Real values')
    plt.legend(loc='upper left')
    plt.savefig(f'plot_prediction_{pl}_{mw}.svg')
    plt.show()

filename = 'neuralNetwork.sav'
filename1 = 'basicmodel.sav'


### lastly, save our models 

pickle.dump(complex_model, open(filename, 'wb'))

pickle.dump(basic_model, open(filename1, 'wb'))

pd.DataFrame(complex_pred).to_csv("file.csv")

pd.DataFrame(real).to_csv("file1.csv")