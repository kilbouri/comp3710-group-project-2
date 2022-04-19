from time import time
from progress.bar import IncrementalBar

from util import randomize


def testHoldout(learningMethod, data, splitPortion=0.5, className='classification', showProgress=True):
    """
    Determines effectiveness of learning method based on hold-out method. Data is split
    into a test and training set. The first showProgress% of the values are used for testing.

    Learning method must contain a function learn(examples, *args) and return
    a structure containing an evaluate(data) function, returning the classification
    of `data`.
    """
    splitPoint = int(len(data) * splitPortion)
    splitPoint = max(1, splitPoint)
    splitPoint = min(splitPoint, len(data))

    data = tuple(randomize(data))
    trainingSet = data[:splitPoint]
    testSet = data[splitPoint:]

    classifier = learningMethod.learn(trainingSet)

    iterator = testSet
    if showProgress:
        iterator = IncrementalBar("Hold-Out Evaluation").iter(testSet)

    numCorrect = 0
    for testItem in iterator:
        testData = dict(testItem)
        correct = testData.pop(className)

        classifierResult = classifier.evaluate(testData)
        if classifierResult == correct:
            numCorrect += 1

    return (numCorrect / len(testSet)), classifier
