from cmath import inf
from collections import Counter
import os
from util import entropy, filterByAttribute, getValueSet, mode, getValues, removeItem, setToDeterministicTuple
from structures.DecisionTree import DecisionTree
from concurrent.futures import ThreadPoolExecutor


def DTL(data, attrs: set, className='classification'):
    if len(data) == 0:
        raise ValueError("Example list cannot be empty!")
    if len(attrs) == 0:
        raise ValueError("Attribute set cannot be empty!")

    attributes = setToDeterministicTuple(attrs - {className})
    return _DTL_Helper(data, attributes, className, 'p')


def _DTL_Helper(data, attrs, className, default):
    # are we out of examples?
    if len(data) == 0:
        return DecisionTree(default)

    # do all classifications match?
    if len(getValueSet(className, data)) == 1:
        return DecisionTree(data[0][className])

    # are we out of attributes to classify?
    if len(attrs) == 0:
        return DecisionTree(mode(getValues(className, data)))

    best = _selectAttribute(data, attrs, className)
    tree = DecisionTree(best)

    newAttributes = removeItem(best, attrs)
    default = mode(getValues(className, data))

    def threadTask(valueOfBest):
        newExamples = filterByAttribute(best, valueOfBest, data)
        subtree = _DTL_Helper(newExamples, newAttributes, className, default)
        return valueOfBest, subtree

    # split the work across some threads when there are more than 4 possible values
    if len(getValueSet(best, data)) > 2:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(threadTask, valueOfBest) for valueOfBest in getValues(best, data)]

        results = [future.result() for future in futures]
        for branchName, subtree in results:
            tree.addBranch(branchName, subtree)
    else:
        for valueOfBest in getValues(best, data):
            _, subtree = threadTask(valueOfBest)
            tree.addBranch(valueOfBest, subtree)

    return tree


def _selectAttribute(data, attrs, className):
    edibles = filterByAttribute(className, 'e', data)
    pEdible = len(edibles) / len(data)

    # calculate initial entropy
    eInitial = entropy(pEdible, 1 - pEdible)

    # find best splitting attribute based on information gain (IG)
    bestAttribute = None
    bestInfGain = -inf

    for attribute in attrs:
        eAttribute = _entropyOfSelectingAttr(attribute, data, className)
        gain = eInitial - eAttribute

        if gain > bestInfGain:
            bestInfGain = gain
            bestAttribute = attribute

    return bestAttribute


def _entropyOfSelectingAttr(attr, data, className):
    totalEntropy = 0
    branches = Counter(getValues(attr, data))

    for branchName, count in branches.most_common():
        attrMatches = filterByAttribute(attr, branchName, data)
        edibleUnderAttr = filterByAttribute(className, 'e', attrMatches)
        pEdible = len(edibleUnderAttr) / len(attrMatches)

        attrEntropy = entropy(pEdible, 1 - pEdible)
        branchProbability = count / len(data)
        totalEntropy += branchProbability * attrEntropy

    return totalEntropy
