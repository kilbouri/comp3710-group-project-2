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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

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
data['classification'] = data['classification'].apply(lambda x: 1 if x == 'e' else 0)
data = data.applymap(lambda s : {i:o for o,i in enumerate('abcdefghijklmnopqrstuvwxyz?')}[s] if type(s) == str else s)
# split into train and test data
split = 0.2
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)
del data

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

# %%
# build the model

# input is number of inputs; -1 because of the classification label
# each dense layer is a hidden layer
model = keras.Sequential(
    [
        keras.Input(shape=(len(CSV_HEADER)-1,), dtype='float32'), 
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
# %%
# prepare the data


# %%
# train the model
(x_train, y_train), (x_test, y_test) = (train_data.drop(columns=['classification']), train_data['classification']), (test_data.drop(columns=['classification']), test_data['classification'])

# print shapes
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# %%
# train the model

model.fit(x_train.values, y_train.values, epochs=20, batch_size=128, validation_split=0.2)

# %%
