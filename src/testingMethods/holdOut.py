from progress.bar import IncrementalBar
from util import randomize


def testHoldout(learningMethod, data, splitPortion=0.5, className='classification', showProgress=True, bulkTest=False, randomizeData=True):
    """
    Determines effectiveness of learning method based on hold-out method. Data is split
    into a test and training set. The first showProgress% of the values are used for testing.

    Learning method must contain a function learn(examples) and an evaluate(data) 
    function, returning the classification of `data`.
    """

    # split data into training and testing sets
    splitPoint = int(len(data) * splitPortion)
    splitPoint = max(1, splitPoint)
    splitPoint = min(splitPoint, len(data))

    if randomizeData:
        data = tuple(randomize(data))
    trainingSet = data[:splitPoint]
    testSet = data[splitPoint:]

    learningMethod.learn(trainingSet)

    if not bulkTest:
        for testItem in iterator:
            testData = dict(testItem)
            correct = testData.pop(className)

            # learningMethod.evaluate() asks the learning method to classify
            # the given testData
            classifierResult = learningMethod.evaluate(testData)
            if classifierResult == correct:
                numCorrect += 1
    else:
        results = learningMethod.evaluate(testSet)
        numCorrect = sum(result == testItem[className] for result, testItem in zip(results, testSet))

    return (numCorrect / len(testSet))
