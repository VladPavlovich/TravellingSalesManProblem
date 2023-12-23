class Matrix:
    def __init__(self, values, parent, index):
        self.values = values
        self.parent = parent
        self.children = []
        self.cost = 0
        self.index = index
        self.pastIndex = self.getPastIndex()

    def rowMinimums(self):
        minimums = []
        for x in range(0, len(self.values)):
            minimums.append(min(self.values[x]))
        return minimums

    def colMinimums(self):
        minimums = []
        for i in range(0, len(self.values[0])):
            col = []
            for x in range(0, len(self.values)):
                col.append(self.values[x][i])
            minimums.append(min(col))
        return minimums

    def setRowInf(self, row):
        for x in range(0, len(self.values[row])):
            self.values[row][x] = float('inf')

    def setColInf(self, col):
        for x in range(0, len(self.values)):
            self.values[x][col] = float('inf')

    def getPastIndex(self):
        currentParent = self.parent
        past = []
        while currentParent is not None:
            past.append(currentParent.index)
            currentParent = currentParent.parent

        return past
