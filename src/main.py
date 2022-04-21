import os
from learningMethods.DTL import DTLLearner
from learningMethods.NNL import NNLearner
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
    
    TEST = True
    dataPath = '../data/mushrooms.short.dat' if TEST else '../data/mushrooms.dat'
    attrSet, examples = loadData(dataPath, '../data/attributes.dat')

    print("=== DTL ===")
    dtlLearner = DTLLearner(attrSet)
    accuracy = testHoldout(dtlLearner, examples, splitPortion=0.2)
    print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
    accuracy = kFoldCross(dtlLearner, examples, max_workers=os.cpu_count())
    print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")
    
    print("\n=== NN ===")
    nnLearner = NNLearner(attrSet)
    accuracy = testHoldout(nnLearner, examples, splitPortion=0.2)
    print(f"NN Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
    accuracy = kFoldCross(nnLearner, examples)
    print(f"NN Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")

    nnLearner.tuneHyperparameters(examples)


if __name__ == '__main__':
    main()

