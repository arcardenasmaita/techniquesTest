# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Lime%20with%20Recurrent%20Neural%20Networks.ipynb
# This focuses on keras-style "stateless" recurrent neural networks. These networks expect input with a shape (n_samples, n_timesteps, n_features)
# as opposed to the more normal (n_samples, n_features) input that most other machine learning algorithms expect.
# To explain the neural network models, we use a variant on the TabularExplainer that takes care of reshaping data appropriately.

# Imports
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam  # from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical  # from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from lime import lime_tabular
####################################
'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite 
computationally intensive.

Author: Niek Tax
'''

from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
# from keras.optimizers import Nadam #old version
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.layers.normalization import BatchNormalization ### old
from tensorflow.keras.layers import BatchNormalization
from keras.layers.normalization import layer_normalization

from collections import Counter
import unicodecsv
import numpy as np
import pandas as pd
import random
import sys
import os
import copy
import csv
import time
# from itertools import izip
from itertools import zip_longest as izip
from datetime import datetime
from math import log
from lime import lime_tabular

df = pd.read_csv('data/helpdesk_columnaas.csv', index_col=0, parse_dates=True)
####################################


# Data
#df = pd.read_csv('data/co2_data.csv', index_col=0, parse_dates=True)
#fig, (left, right) = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
#df[['co2']].plot(ax=left)
#df[['co2_detrended']].plot(ax=right)
#fig.savefig('results/data_lime')

# Reshaping the dataset to be appropriate for the model
def reshape_data(seq, n_timesteps):
    N = len(seq) - n_timesteps - 1
    nf = seq.shape[1]
    if N <= 0:
        raise ValueError('I need more data!')
    new_seq = np.zeros((N, n_timesteps, nf))
    for i in range(N):
        new_seq[i, :, :] = seq[i:i + n_timesteps]
    return new_seq


N_TIMESTEPS = 3  # Use 1 year of lookback
data_columns = ['timestamp', 'act',	'time']
target_columns = ['ontime']

scaler = MinMaxScaler(feature_range=(-1, 1))
X_original = scaler.fit_transform(df[data_columns].values)
X = reshape_data(X_original, n_timesteps=N_TIMESTEPS)
y = to_categorical((df[target_columns].values[N_TIMESTEPS:-1]).astype(int))

# Train on the first 2000, and test on the last 276 samples
X_train = X#[:2000]
y_train = y#[:2000]
X_test = X#[2000:]
y_test = y#[2000:]
print('SHAPE --> X: ', X.shape, ' Y: ', y.shape)

# Define the model
model = Sequential()
model.add(LSTM(32, input_shape=(N_TIMESTEPS, len(data_columns))))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(X_train, y_train, batch_size=100, epochs=50,  # old 500 epochs
          validation_data=(X_test, y_test),
          verbose=2)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))

# Explain the model with LIME:
explainer = lime_tabular.RecurrentTabularExplainer(X_train, training_labels=y_train, feature_names=data_columns,
                                                   discretize_continuous=True,
                                                   class_names=['out', 'ontime'],
                                                   discretizer='decile')
exp = explainer.explain_instance(X_test[50], model.predict, num_features=10, labels=(1,))
#exp.show_in_notebook()
fig = exp.as_pyplot_figure()
fig.savefig('results/exp_rnn_lstm_tax_simple.jpg')
