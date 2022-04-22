import os
from tabnanny import verbose

if os.getlogin() == 'Mat':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2" 

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

# convert all strings to integers
# convert classifications to 1 and 0 (binary for accuracy metric later)
data['classification'] = data['classification'].apply(lambda x: 1 if x == 'e' else 0)
# split into train and test data
split = 0.25
train_data = data.sample(frac=split)
test_data = data.drop(train_data.index)
#split data into inputs and outputs

train_inputs = train_data.drop(columns=['classification'])
train_outputs = train_data['classification']
test_inputs = test_data.drop(columns=['classification'])
test_outputs = test_data['classification']


classifier = keras.Sequential()
classifier.add(keras.layers.Dense(name="Input", units=32, activation='relu', input_shape=[len(CSV_HEADER)-1]))
classifier.add(keras.layers.Dense(units=16, activation='relu'))
classifier.add(keras.layers.Dense(units=4, activation='relu'))
classifier.add(keras.layers.Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(train_inputs,train_outputs, batch_size=5 , epochs=10, verbose=0)

pout = classifier.predict(test_inputs)
print(f"Accuracy: {sum(int(poo > 0.5) == yt for poo, yt in zip(pout, test_outputs.values))/len(test_outputs.values)*100}")
