from Implementations.Interfaces.IRunnableAlgorithm import IRunnableAlgorithm


class GreedySubsetSum(IRunnableAlgorithm):

    def __init__(self, label):
        self.label = label
    def run(self, values, target):
        opt = []
        values.sort()
        weight = 0
        for i in range(len(values)):
            if sum(opt) + values[i] <= target:
                opt.append(values[i])
        return opt