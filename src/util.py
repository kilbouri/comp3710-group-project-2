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