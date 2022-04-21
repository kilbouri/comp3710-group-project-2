class DecisionTree:
    def __init__(self, label, branches=None, defaultValue=None):
        self.label = label
        self.branches = branches
        self.defaultValue = defaultValue

    def addBranch(self, branchLabel, subtree):
        # make sure the subtree is actually a tree, so that
        # issues won't be caused in prettyTree
        if not isinstance(subtree, DecisionTree):
            raise ValueError("Subtree must be a DecisionTree!")

        if self.branches == None:
            self.branches = dict()

        self.branches[branchLabel] = subtree

    def decide(self, data):
        if self.isLeaf():
            return self.label

        branchLabel = data[self.label]
        subtree = self.branches.get(branchLabel)

        if subtree == None:
            return self.defaultValue

        return subtree.decide(data)

    def isLeaf(self):
        return self.branches == None

    def prettyTree(self, indent=0, branchName=''):
        if branchName:
            print(' ' * indent + branchName + '->' + self.label)
        else:
            print(' ' * indent + self.label)

        if self.isLeaf():
            return

        for branchName, subtree in self.branches.items():
            subtree.prettyTree(indent + 4, branchName)
