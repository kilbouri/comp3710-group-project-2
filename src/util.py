from math import inf, log2
from typing import Counter, SupportsFloat
from random import uniform


def randomize(items):
    unvisited = list(items)
    while len(unvisited) != 0:
        index = int(uniform(0, len(unvisited)))
        item = unvisited.pop(index)
        yield item


def clamp(value, _min, _max):
    return max(_min, min(_max, value))


def mode(*items):
    return Counter(*items).most_common(1)[0][0]


def entropy(*probabilities: SupportsFloat):
    if 0 in probabilities:
        return -inf  # there will be a math error, entropy is undefined
    epy = sum(-p * log2(p) for p in probabilities)
    return epy


def getValues(attribute, data):
    return tuple(map(lambda x: x[attribute], data))


def getValueSet(attribute, data):
    return set(getValues(attribute, data))


def filterByAttribute(attr, attrValue, data):
    return tuple(filter(lambda x: x[attr] == attrValue, data))


########################################################################
# UTILITIES FOR TF DATA FORMATS
########################################################################

def transform_data(issacinput:list):
    # convert issac's data format into something that can be fed to a dataframe
    out = {col : [] for col in issacinput[0].keys()}
    for d in issacinput:
        for k,v in d.items():
            out[k].append(v)
    return out

def data_str_to_int(data:dict):
    # takes a dict from transform_data and converts all strings to ints
    m = {i:o for o,i in enumerate('abcdefghijklmnopqrstuvwxyz?')}
    for k,v in data.items():
        data[k] = [m[x] for x in v]
    return data
