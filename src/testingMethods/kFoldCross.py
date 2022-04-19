import os
from util import randomize
from concurrent.futures import ThreadPoolExecutor
from progress.bar import IncrementalBar


def kFoldCross(learningMethod, data, kSets=8, className='classification', showProgress=True):
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
        classifier = learningMethod.learn(trainingSet)

        numCorrect = 0
        for test in testSet:
            inputData = dict(test)
            correct = inputData.pop(className)

            classifierResult = classifier.evaluate(inputData)
            if classifierResult == correct:
                numCorrect += 1

        return numCorrect / len(testSet)

    # dispatch evaluation task to threads
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        threads = [
            executor.submit(testClassifierTask, bucket)
            for bucket in trainingBuckets
        ]

    # progress bar wrapper lol
    if showProgress:
        bar = IncrementalBar(f"{kSets}-Fold Cross-Validation")
        wrappedFutures = bar.iter(threads)
    else:
        wrappedFutures = threads

    # return the average from the evaluation tasks
    return sum(task.result() for task in wrappedFutures) / len(threads)
