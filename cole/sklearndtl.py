# %% [markdown]
# # there is no stopping the cole train
#
# This exists so I can mess around with tf.keras until it successfully trains on the dataset we are using.

# %% [markdown]
# use tf.keras to train on the dataset found under ../data

# %%
# import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import RandomFourierFeatures
from sklearn import svm
# %% [markdown]
# import the dataset
#
# - `../data/attributes.dat` contains the `CSV_HEADER`
# - `../data/mushrooms.short.dat` contains csv entries

# %%

CSV_HEADER = open('../data/attributes.dat').readline().strip().split(',')
data = pd.read_csv('../data/mushrooms.dat', header=None, names=CSV_HEADER)

# convert all strings to integers
# convert classifications to 1 and 0 (binary for accuracy metric later)
data['classification'] = data['classification'].apply(
    lambda x: 1 if x == 'e' else 0)
data = data.applymap(lambda s: {i: o for o, i in enumerate(
    'abcdefghijklmnopqrstuvwxyz?')}[s] if type(s) == str else s)
# split into train and test data
split = 0.4
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)
del data

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

# %%
# build the model

# input is number of inputs; -1 because of the classification label
# each dense layer is a hidden layer
clf = svm.SVC(kernel='linear')

# %%
# prepare the data
(x_train, y_train), (x_test, y_test) = (train_data.drop(columns=[
    'classification']), train_data['classification']), (test_data.drop(columns=['classification']), test_data['classification'])

# convert all values to float32
x_train = x_train.apply(lambda s: s.astype('float32'))
x_test = x_test.apply(lambda s: s.astype('float32'))

# print shapes
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# %%
# train the model

clf.fit(x_train, y_train)

# %%
# check accuracy
print(f"Accuracy: {clf.score(x_test, y_test)}")
