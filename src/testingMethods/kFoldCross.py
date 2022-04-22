from util import randomize
from concurrent.futures import ThreadPoolExecutor
from progress.bar import IncrementalBar


def kFoldCross(learningMethod, data, kSets=8, max_workers=1, className='classification', showProgress=True, bulkTest=False, randomizeData=True):
    """
    Determines effectiveness of learning method based on k-fold cross-validation method. 
    Data is split into a test and training set. The data is broken into `kSets` groups.
    A distinct instance of the learning method is trained on each of the k-1 sets, and 
    the remaining set is used for evaluation of each instance. The average accuracy is
    returned.

    Learning method must contain a function learn(examples) and an evaluate(data) 
    function, returning the classification of `data`.
    """

    if max_workers != 1 and bulkTest:
        raise ValueError("Bulk test is not supported with multiple workers")

    if randomizeData:
        data = tuple(randomize(data))
    bucketSize = len(data) // kSets

    # break into k-1 training sets and a test set
    trainingBuckets = []
    lastSplitPoint = 0
    for _ in range(kSets - 1):
        trainingBuckets.append(data[lastSplitPoint:lastSplitPoint + bucketSize])
        lastSplitPoint += bucketSize

    testSet = data[lastSplitPoint:]

    # multithreading function target
    def testClassifierTask(trainingSet):
        # ask the classifier to learn from the training set
        learningMethod.learn(trainingSet)

        # evaluate the classifier's performance based on the shared
        # test set
        numCorrect = 0
        if not bulkTest:
            for test in testSet:
                inputData = dict(test)
                correct = inputData.pop(className)

                classifierResult = learningMethod.evaluate(inputData)
                if classifierResult == correct:
                    numCorrect += 1
        else:
            results = learningMethod.evaluate(testSet)
            numCorrect = sum(result == test[className] for result, test in zip(results, testSet))

        return numCorrect / len(testSet)

    if not bulkTest:
        # dispatch evaluation task to threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            threads = [
                executor.submit(testClassifierTask, bucket)
                for bucket in trainingBuckets
            ]

        # progress bar wrapper lol
        if showProgress and not bulkTest:
            bar = IncrementalBar(f"{kSets}-Fold Cross-Validation")
            wrappedFutures = bar.iter(threads)
        else:
            wrappedFutures = threads

        # return the average from the evaluation tasks
        return sum(task.result() for task in wrappedFutures) / len(threads)
    else:
        return testClassifierTask(trainingBuckets[0])
