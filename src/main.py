import os
from learningMethods.DTL import DTLLearner
from learningMethods.NNL import NNLearner
from learningMethods.SVM import SVMLearnerSK, SVMLearnerKeras
from testingMethods.holdOut import testHoldout
from testingMethods.kFoldCross import kFoldCross
from util import randomize


def loadData(dataPath, attributePath):
    # read the attribute name mapping so we can zip it later
    with open(attributePath, 'r') as file:
        attrs = file.readline().strip().split(',')
        attrs = map(lambda x: x.strip(), attrs)
        attrs = tuple(attrs)

    # read the actual data
    with open(dataPath, 'r') as file:
        lines = file.readlines()

    examples = map(lambda x: x.strip().split(','), lines)

    return set(attrs), tuple(
        dict(zip(attrs, values))
        for values in examples
    )


def main():
    
    TEST = False
    randomizeData = True
    TESTDTL, SVMSK, SVMKERAS, NN, NNTUNE = True, True, True, True, False
    dataPath = '../data/mushrooms.short.dat' if TEST else '../data/mushrooms.dat'
    attrSet, examples = loadData(dataPath, '../data/attributes.dat')

    if TESTDTL:
        print("\n=== DTL ===")
        dtlLearner = DTLLearner(attrSet)
        accuracy = testHoldout(dtlLearner, examples, splitPortion=0.2, showProgress=False)
        print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
        accuracy = kFoldCross(dtlLearner, examples, max_workers=os.cpu_count(), showProgress=False)
        print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")
    
    if SVMSK:
        print("\n=== SVM(sklearn) ===")
        svmLearnerSK = SVMLearnerSK(attrSet)
        accuracy = testHoldout(svmLearnerSK, examples, splitPortion=0.2, bulkTest=True, showProgress=False)
        print(f"SVM(sklearn) Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
        accuracy = kFoldCross(svmLearnerSK, examples, max_workers=1, bulkTest=True, showProgress=False)
        print(f"SVM(sklearn) Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")

    if SVMKERAS:
        print("\n=== SVM(keras) ===")
        svmLearnerKeras = SVMLearnerKeras(attrSet)
        accuracy = testHoldout(svmLearnerKeras, examples, splitPortion=0.2, bulkTest=True, showProgress=False)
        print(f"SVM(keras) Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
        accuracy = kFoldCross(svmLearnerKeras, examples, max_workers=1, bulkTest=True, showProgress=False)
        print(f"SVM(keras) Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")


    if NN:
        print("\n=== NN ===")
        nnLearner = NNLearner(attrSet)
        accuracy = testHoldout(nnLearner, examples, splitPortion=0.2, bulkTest=True, showProgress=False)
        print(f"NN Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
        accuracy = kFoldCross(nnLearner, examples, bulkTest=True, showProgress=False)
        print(f"NN Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")

    if NNTUNE:
        print("\n=== NNTuner ===")
        nnLearnerTune = NNLearner(attrSet)
        nnLearnerTune.tuneHyperparameters(examples)


if __name__ == '__main__':
    main()

