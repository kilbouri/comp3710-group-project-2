from learningMethods.DTL import DTLLearner
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

    dtlLearner = DTLLearner(attrSet)
    accuracy, _ = testHoldout(dtlLearner, examples, splitPortion=0.15)
    print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in Hold-Out testing")
    accuracy = kFoldCross(dtlLearner, examples)
    print(f"DTL Learner achieved {100 * accuracy:.2f}% accuracy in 8-fold cross-validation testing")


if __name__ == '__main__':
    main()
