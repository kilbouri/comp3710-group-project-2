from progress.bar import IncrementalBar
from util import randomize


def testHoldout(learningMethod, data, splitPortion=0.5, className='classification', showProgress=True):
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

    data = tuple(randomize(data))
    trainingSet = data[:splitPoint]
    testSet = data[splitPoint:]

    # request the learning method to perform some learning on the training set
    learningMethod.learn(trainingSet)

    # wrapper to show progress bar for long-running training
    iterator = testSet
    if showProgress:
        iterator = IncrementalBar("Hold-Out Evaluation").iter(testSet)
    
    # evaluate the learning method based on test set
    numCorrect = 0
    for testItem in iterator:
        testData = dict(testItem)
        correct = testData.pop(className)

        # learningMethod.evaluate() asks the learning method to classify
        # the given testData
        classifierResult = learningMethod.evaluate(testData)
        #print(f"- {classifierResult}, {correct}")
        if classifierResult == correct:
            numCorrect += 1

    return (numCorrect / len(testSet))
