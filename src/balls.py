import os
from learningMethods.DTL import DTLLearner
from learningMethods.NNL import NNLearner
from learningMethods.SVM import SVMLearner
from testingMethods.holdOut import testHoldout
from testingMethods.kFoldCross import kFoldCross
from util import randomize

# %%

attributePath = '../data/attributes.dat'
dataPath = '../data/mushrooms.short.dat'

# read the attribute name mapping so we can zip it later
with open(attributePath, 'r') as file:
    attrs = file.readline().strip().split(',')
    attrs = map(lambda x: x.strip(), attrs)
    attrs = tuple(attrs)
# read the actual data
with open(dataPath, 'r') as file:
    lines = file.readlines()
examples = map(lambda x: x.strip().split(','), lines)
cum, balls =  set(attrs), tuple(
    dict(zip(attrs, values))
    for values in examples
)


# %%
import pandas as pd
cock = pd.DataFrame(balls)
# %%
from util import data_str_to_int
cock = data_str_to_int(cock)
# %%
