import math
from operator import itemgetter

from numpy import zeros, nonzero, select
from scipy.signal import fftconvolve

from Implementations.DPRegularWithCount import DynamicProgrammingWithCount
from Implementations.helpers.Helper import eps, divideAndConquerSumSet, bruteForceSolve
from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm

smallSetSolver = DynamicProgrammingWithCount()

class FastIntegersFromJournal(IDeterministicAlgorithm):

    def __init__(self, label, b=lambda n: max(int(math.sqrt(n * math.log(n))), 1), benchmarkMode=False, thresholdValue=3, dnqThreshold=10, recursionLimit=10):
        self.label = label
        self.bFunc = b
        self.benchmarkMode = benchmarkMode
        self.thresholdValue = thresholdValue
        self.dnqThreshold = dnqThreshold
        self.recursionLimit = recursionLimit


    def run(cls, values, target):
        # Not certain why we have to add + 1 here, might bite me in the ass later.
        return sorted(list(cls.SS_SmallInput(values, target+1)))

    def minkowski_sum(self, a, b, u):

        def minkowski_sum_1D(a, b, u):
            result = minkowski_sum_2D([(x, 0) for x in a], [(x, 0) for x in b], u)
            return [x for (x, _) in result]

        def minkowski_sum_2D(a, b, u):
            def L2M(S):
                # list to matrix representation
                c1 = list(map(itemgetter(0), S))
                c2 = list(map(itemgetter(1), S))
                M = zeros((max(c1) + 1, max(c2) + 1))
                M[c1, c2] = 1
                return M

            def M2L(M):
                # matrix to list representation
                return zip(*nonzero(select([M > eps()], [1])))

            conv = fftconvolve(L2M(a), L2M(b))
            return [(x, y) for (x, y) in M2L(conv) if x < u]

        if type(a[0]) is tuple:
            return minkowski_sum_2D(a, b, u)
        else:
            return minkowski_sum_1D(a, b, u)

    # input is a list of *distinct* non-negative integers S, u is an upper bound
    # output all subset sums of S up to u together with the cardinality information
    def SSC_BoundSum(self, S, u):
        if len(S) == 1:
            return [(0, 0), (S[0], 1)]
        if len(S) <= self.recursionLimit:
            return smallSetSolver.subsetSumDP(S, u)
        return self.minkowski_sum(self.SSC_BoundSum(S[0::2], u), self.SSC_BoundSum(S[1::2], u), u)

    # input is a list of *distinct* non-negative integers S, u is an upper bound
    # output all subset sums of S up to u, as a set.
    def SS_SmallInput(self, S, u):
        greaterThanTHalf = [x for x in S if x > math.floor(u/2)]
        greaterThanTHalf.append(0)
        usingLessThanTHalf = [x for x in S if x <= math.floor(u/2)]
        if len(usingLessThanTHalf) == 0:
            return greaterThanTHalf
        n = len(usingLessThanTHalf)
        b = self.bFunc(n)  # int(math.sqrt(n * math.log(n)))
        # compute S_l, need to do this in a single loop in order to do partition in linear time.
        partitioned_S = dict()
        divideAndConquerSet = [0]
        for x in usingLessThanTHalf:
            if x < b:
                divideAndConquerSet.append(x)
                continue
            if x % b not in partitioned_S:
                partitioned_S[x % b] = []
            partitioned_S[x % b].append(x)
        result = [0]
        for l in range(b):
            if l in partitioned_S:
                realVals = partitioned_S[l]
                if len(realVals) <= self.thresholdValue:
                    R_l = list(bruteForceSolve(realVals, u))
                elif len(realVals) <= self.dnqThreshold:
                    R_l = list(divideAndConquerSumSet(realVals, u))
                else:
                    Q_l = [x // b for x in realVals]
                    R_l = [z * b + l * j for (z, j) in self.SSC_BoundSum(Q_l, u // b + 1)]
                result = self.minkowski_sum(result, R_l, u)
                if self.benchmarkMode and result[-1] == u-1:
                    return [result[-1]]
        return self.minkowski_sum(self.minkowski_sum(divideAndConquerSumSet(divideAndConquerSet, u), result, u), greaterThanTHalf, u)
        return set(result)
