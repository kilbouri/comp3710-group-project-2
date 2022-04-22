# %% [markdown]
# # there is no stopping the cole train
#
# This exists so I can mess around with tf.keras until it successfully trains on the dataset we are using.

# %% [markdown]
# use tf.keras to train on the dataset found under ../data

# %%
# import numpy as np
from progress.bar import IncrementalBar
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
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
            dfdict[k] = [{'e': 1, 'p': 0}[x] for x in v]
        else:
            m = {i: o for o, i in enumerate('abcdefghijklmnopqrstuvwxyz?')}
            dfdict[k] = [m[x] for x in v]
    return dfdict


CSV_HEADER, data = loadData(
    '../../data/mushrooms.dat', '../../data/attributes.dat')

# data = pd.DataFrame(data)
# data['classification'] = data['classification'].apply(lambda x: {'e': 1, 'p': 0}[x])
# data = data.applymap(lambda x: ord(x) if isinstance(x, str) else x)

data = pd.DataFrame(issac_to_dfdict(data))

data = data.apply(lambda s: s.astype('float32'))

# %%

# CSV_HEADER = open('../../data/attributes.dat').readline().strip().split(',')
# data = pd.read_csv('../../data/mushrooms.dat', header=None, names=CSV_HEADER)


# convert all strings to integers
# convert classifications to 1 and 0 (binary for accuracy metric later)
# data['classification'] = data['classification'].apply(
#     lambda x: {'e':1, 'p':0}[x])
# data = data.applymap(lambda s: {i: o for o, i in enumerate(
#     'abcdefghijklmnopqrstuvwxyz?')}[s] if type(s) == str else s)

# %%
def test(train_data, test_data):
    """train, test, return accuracy"""

    # print(f"Train dataset shape: {train_data.shape}")
    # print(f"Test dataset shape: {test_data.shape}")

    # build the model

    # one versus one decision shape
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    # clf = svm.LinearSVC()

    # prepare the data
    (x_train, y_train), (x_test, y_test) = (train_data.drop(columns=[
        'classification']), train_data['classification']), (test_data.drop(columns=['classification']), test_data['classification'])

    # convert all values to float32
    # x_train = x_train.apply(lambda s: s.astype('float32'))
    # x_test = x_test.apply(lambda s: s.astype('float32'))

    # # print shapes
    # print(f"x_train shape: {x_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"x_test shape: {x_test.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # train the model
    # print(y_train)
    clf.fit(x_train, y_train)

    # check accuracy
    # print(f"Accuracy: {clf.score(x_test, y_test)}")
    pout = clf.predict(x_test)
    acc = sum(int(poo > 0.5) == yt for poo, yt in zip(
        pout, y_test.values))/len(y_test.values)
    print(f"Accuracy: {acc}")
    return acc


# %%

# split into train and test data
split = 0.6
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)

# HOLDOUT
print(f'Holdout Result: {test(train_data, test_data)}')

# %%

# k-fold method, split training data into k parts
split = 0.8
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)
k = 8
train_blocks = []
for i in range(k-1, 0, -1):
    sample = train_data.sample(frac=i/k)
    train_blocks.append(sample)
    # drop sample from train_data
    train_data.drop(sample.index, inplace=True)
train_blocks.append(train_data)


print(f'cpu count is {cpu_count()}')
with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
    threads = [
        executor.submit(test, block, test_data)
        for block in train_blocks
    ]
# # progress bar wrapper lol
# bar = IncrementalBar(f"{k}-Fold Cross-Validation")
# wrappedFutures = bar.iter(threads)
# return the average from the evaluation tasks
print(f'{k}-Fold Cross-Validation Result: {sum(task.result() for task in threads) / len(threads)}')

# %%
