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
# UTILITIES FOR KEEPING DETERMINISM
########################################################################


def removeDuplicates(s):
    return setToDeterministicTuple(set(s))


def setToDeterministicTuple(s):
    return sorted(tuple(s))


def removeItem(toRemove, tpl):
    return tuple(filter(lambda x: x != toRemove, tpl))