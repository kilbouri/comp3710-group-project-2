from cmath import inf
from collections import Counter
from util import entropy, filterByAttribute, getValueSet, getValues, mode
from structures.DecisionTree import DecisionTree
from concurrent.futures import ThreadPoolExecutor


class DTLLearner():
    """
    A Decision Tree Learning Agent. Uses entropy-based attribute selection to maximize
    information gain at each branch.

    Available hyperparameters:
        - max depth: how many levels deep the tree can be. Lower values
                     are faster to train but less accurate. (default +inf)
    """

    def __init__(self, attributes, maxDepth=inf, className='classification') -> None:
        if len(attributes) == 0:
            raise ValueError("Attribute list cannot be empty!")

        self.maxDepth = maxDepth
        self.className = className
        self.attributes = attributes - {self.className}
        self.classifier = None

    def learn(self, data):
        if len(data) == 0:
            raise ValueError("Example list cannot be empty!")

        self.classifier = self._DTL_Helper(data, self.attributes, 0, 'e')

    def evaluate(self, data):
        if self.classifier == None:
            raise RuntimeError("DTL cannot be evaluated before learning!")

        return self.classifier.decide(data)

    def _DTL_Helper(self, data, attrs, depth, default):
        # are we out of attributes to classify, or at the depth limit?
        if len(attrs) == 0 or depth > self.maxDepth:
            return DecisionTree(mode(getValues(self.className, data)))

        # are we out of examples?
        if len(data) == 0:
            return DecisionTree(default)

        # do all classifications match?
        if len(getValueSet(self.className, data)) == 1:
            return DecisionTree(data[0][self.className])

        # select best attribute and create subtree root
        best = self._selectAttribute(data, attrs)
        tree = DecisionTree(best, defaultValue=default)

        # filter out attribute and determine new default
        newAttributes = attrs - {best}
        default = mode(getValues(self.className, data))

        # show a progress bar for the first split
        for valueOfBest in getValueSet(best, data):
            newExamples = filterByAttribute(best, valueOfBest, data)
            subtree = self._DTL_Helper(newExamples, newAttributes, depth + 1, default)
            tree.addBranch(valueOfBest, subtree)

        return tree

    def _selectAttribute(self, data, attrs):
        edibles = filterByAttribute(self.className, 'e', data)
        pEdible = len(edibles) / len(data)

        # calculate initial entropy
        eInitial = entropy(pEdible, 1 - pEdible)

        # find best splitting attribute based on information gain (IG)
        bestAttribute = None
        bestInfGain = -inf

        for attribute in attrs:
            eAttribute = self._entropyOfSelectingAttr(attribute, data)
            gain = eInitial - eAttribute

            if gain > bestInfGain:
                bestInfGain = gain
                bestAttribute = attribute

        return bestAttribute

    def _entropyOfSelectingAttr(self, attr, data):
        totalEntropy = 0
        branches = Counter(getValues(attr, data))

        for branchName, count in branches.most_common():
            attrMatches = filterByAttribute(attr, branchName, data)
            edibleUnderAttr = filterByAttribute(self.className, 'e', attrMatches)
            pEdible = len(edibleUnderAttr) / len(attrMatches)

            attrEntropy = entropy(pEdible, 1 - pEdible)
            branchProbability = count / len(data)
            totalEntropy += branchProbability * attrEntropy

        return totalEntropy
