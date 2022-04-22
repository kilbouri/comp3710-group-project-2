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
def loadData(dataPath, attributePath):
    # read the attribute name mapping so we can zip it later
    with open(attributePath, 'r') as file:
        attrs = file.readline().strip().split(',')
        attrs = map(lambda x: x.strip(), attrs)
        attrs = tuple(attrs)

    # read the actual data
    with open(dataPath, 'r') as file:
        lines = file.readlines()

    examples = map(lambda x: x.strip().split(','), lines)

    return attrs, tuple(
        dict(zip(attrs, values))
        for values in examples
    )

def issac_to_dfdict(data):
    """
    takes a df and converts all values to ints
    'classification' becomes 1 for edible and 0 for poisonous
    everything else is encoded as ascii using ord()
    """
    # if input is dict, it is a single input
    if isinstance(data, dict):
        dfdict = {}
        for k, v in data.items():
            out[k] = [v]
    # start by converting from a list of dicts to a dict of lists
    else:
        dfdict = {k: [] for k in data[0].keys()}
        for d in data:
            for k, v in d.items():
                dfdict[k].append(v)

    for k, v in dfdict.items():
        if k == 'index':
            continue
        if k == 'classification':
            dfdict[k] = [{'e':1, 'p':0}[x] for x in v]
        else:
            m = {i: o for o, i in enumerate('abcdefghijklmnopqrstuvwxyz?')}
            dfdict[k] = [m[x] for x in v]
    return dfdict


CSV_HEADER, data = loadData('../../data/mushrooms.dat', '../../data/attributes.dat')
data = pd.DataFrame(issac_to_dfdict(data))
data = data.apply(lambda s: s.astype('float32'))

# %%

# CSV_HEADER = open('../../data/attributes.dat').readline().strip().split(',')
# data = pd.read_csv('../../data/mushrooms.dat', header=None, names=CSV_HEADER)


# # convert all strings to integers
# # convert classifications to 1 and 0 (binary for accuracy metric later)
# data['classification'] = data['classification'].apply(
#     lambda x: 1 if x == 'e' else 0)
# data = data.applymap(lambda s: {i: o for o, i in enumerate(
#     'abcdefghijklmnopqrstuvwxyz?')}[s] if type(s) == str else s)

# %% 

# split into train and test data
split = 0.7
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)
# del data

# print(f"Train dataset shape: {train_data.shape}")
# print(f"Test dataset shape: {test_data.shape}")

# %%
# build the model

# input is number of inputs; -1 because of the classification label
# each dense layer is a hidden layer
model = keras.Sequential(
    [
        keras.Input(shape=(len(CSV_HEADER)-1,), dtype='float32', name='input'),
        # RandomFourierFeatures(
        #     output_dim=4096, scale=10.0, kernel_initializer="gaussian", name='random_fourier'
        # ),
        layers.Dense(units=16, name='middle1', activation='relu'),
        layers.Dense(units=4, name='middle2', activation='relu'),
        layers.Dense(units=1, name='output', activation='sigmoid'),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    # loss=keras.losses.binary_crossentropy,
    # metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    metrics=[keras.metrics.binary_accuracy],
)

# %%
# prepare the data
(x_train, y_train), (x_test, y_test) = (train_data.drop(columns=[
    'classification']), train_data['classification']), (test_data.drop(columns=['classification']), test_data['classification'])

# # convert all values to float32
# x_train = x_train.apply(lambda s: s.astype('float32'))
# x_test = x_test.apply(lambda s: s.astype('float32'))

# print shapes
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape}")
# print(f"y_test shape: {y_test.shape}")

# %%
# train the model

model.fit(x_train.values, y_train.values, epochs=20,
          batch_size=128, validation_split=0.5, verbose=0)


# %%
# print accuracy
pout = model.predict(x_test.values, verbose=0)
# for p, a in zip(pout, y_test.values):
#     p = 1 if p > 0.5 else 0
#     print(f"Predicted: {p} Actual: {int(a)} -> {p == int(a)}")
# print(y_test)
print(f"Accuracy: {sum(int(poo > 0.5) == yt for poo, yt in zip(pout, y_test.values))/len(y_test.values)}")

# %%
