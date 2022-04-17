from learningMethods.DTL import DTL


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

    tree = DTL(examples, attrSet)
    tree.prettyTree()


if __name__ == '__main__':
    main()
