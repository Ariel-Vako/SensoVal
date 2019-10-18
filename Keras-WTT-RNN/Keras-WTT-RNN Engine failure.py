import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import BatchNormalization

from keras import backend as k
from keras import callbacks

from sklearn.preprocessing import normalize

import pandas as pd
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM, GRU
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import RMSprop, adam
from keras.callbacks import History

import wtte.weibull as weibull
import wtte.wtte as wtte

from wtte.wtte import WeightWatcher

# help(normalize)

from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tqdm
from tqdm import tqdm

np.random.seed(2)
pd.set_option("display.max_rows", 1000)


def weibull_loglik_discrete(y_true, ab_pred, name=None):
    """
        Discrete log-likelihood for Weibull hazard function on censored survival data
        y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
        ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
        For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
    """
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)


def weibull_loglik_continuous(y_true, ab_pred, name=None):
    """
        Not used for this model, but included in case somebody needs it
        For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
    """
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + 1e-35) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))


def activate(ab):
    """
        Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
    """

    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)


"""
    Load and parse engine data files into:
       - an (engine/day, observed history, sensor readings) x tensor, where observed history is 100 days, zero-padded
         for days that don't have a full 100 days of observed history (e.g., first observed day for an engine)
       - an (engine/day, 2) tensor containing time-to-event and 1 (since all engines failed)
    There are probably MUCH better ways of doing this, but I don't use Numpy that much, and the data parsing isn't the
    point of this demo anyway.
"""
pass

id_col = 'unit_number'
time_col = 'time'
feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_measurement_{}'.format(x) for x in range(1, 22)]
column_names = [id_col, time_col] + feature_cols

np.set_printoptions(suppress=True, threshold=10000)

train_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/train.csv', header=None, names=column_names)
test_x_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_x.csv', header=None, names=column_names)
test_y_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_y.csv', header=None, names=['T'])

test_x_orig.set_index(['unit_number', 'time'], verify_integrity=True)

# Combine the X values to normalize them,
all_data_orig = pd.concat([train_orig, test_x_orig])
# all_data = all_data[feature_cols]
# all_data[feature_cols] = normalize(all_data[feature_cols].values)

scaler = pipeline.Pipeline(steps=[
    #     ('z-scale', StandardScaler()),
    ('minmax', MinMaxScaler(feature_range=(-1, 1))),
    ('remove_constant', VarianceThreshold())
])

all_data = all_data_orig.copy()
all_data = np.concatenate([all_data[['unit_number', 'time']], scaler.fit_transform(all_data[feature_cols])], axis=1)

# then split them back out
train = all_data[0:train_orig.shape[0], :]
test = all_data[train_orig.shape[0]:, :]

# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test[:, 0:2] -= 1


# TODO: replace using wtte data pipeline routine
def build_data(engine, time, x, max_time, is_test, mask_value):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = []

    # number of features
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []

    n_engines = 100
    for i in tqdm(range(n_engines)):
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y.append(np.array((max_engine_time - j, 1), ndmin=2))

            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value
            #             xtemp = np.full((1, max_time, d), mask_value)

            xtemp[:, max_time - min(j, 99) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
            this_x.append(xtemp)

        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    return out_x, out_y


# # Configurable observation look-back period for each engine/day
max_time = 100
mask_value = -99

train_x, train_y = build_data(engine=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=max_time, is_test=False, mask_value=mask_value)
test_x, _ = build_data(engine=test[:, 0], time=test[:, 1], x=test[:, 2:], max_time=max_time, is_test=True, mask_value=mask_value)

# train_orig.groupby('unit_number')['time'].describe()

# always observed in our case
test_y = test_y_orig.copy()
test_y['E'] = 1

print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape)

tte_mean_train = np.nanmean(train_y[:, 0])
mean_u = np.nanmean(train_y[:, 1])

# Initialization value for alpha-bias
init_alpha = -1.0 / np.log(1.0 - 1.0 / (tte_mean_train + 1.0))
init_alpha = init_alpha / mean_u
print('tte_mean_train', tte_mean_train, 'init_alpha: ', init_alpha, 'mean uncensored train: ', mean_u)
