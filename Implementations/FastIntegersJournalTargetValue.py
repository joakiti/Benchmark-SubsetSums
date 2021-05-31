from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm
from Implementations.helpers.Helper import divideAndConquerSumSet, sumSetNotComplex
import numpy as np

class FastIntegersJournalTargetValue(IDeterministicAlgorithm):

    @classmethod
    def run(cls, values, target):
        return cls.subsetSum(values, target)

    @classmethod
    def subsetSum(self, xs, t):
        r = t ** (3/4)
        xs = np.array(xs)
        s_l = xs[xs < r]
        s_r = xs[xs >= r]
        return sumSetNotComplex(self.smallMax(s_l, t), t)

    @classmethod
    def smallMax(self, xs, t):
        M = max(xs)
        n = len(xs)
        allSum = sum(xs)
        if n <= M**(2/3):
            return divideAndConquerSumSet(xs, t)
        else: #The input is dense.
            L = 100*M*M/n

    @classmethod
    def largeMin(cls, xs, u):
        return 1

