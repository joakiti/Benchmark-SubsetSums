import math

from scipy.signal import fftconvolve
import numpy as np

from Implementations.DPRegularWithCount import DynamicProgrammingWithCount
from Implementations.helpers.Helper import MatrixToList, ListToMatrix, divideAndConquerSumSet, sumSetNotComplex, \
    bruteForceSolve
from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm

smallSetSolver = DynamicProgrammingWithCount()
class FastIntegersPersonal(IDeterministicAlgorithm):

    def __init__(self, label, benchmarkMode =False, thresholdValue = 3, dnqThreshold = 10, recursionLimit = 10):
        self.label = label
        self.benchmarkMode = benchmarkMode
        self.thresholdValue = thresholdValue
        self.dnqThreshold = dnqThreshold
        self.recursionLimit = recursionLimit

    def run(cls, values, target):
        return cls.lemma2_16(values, target)

    def lemma2_15(self, S, r0, t):
        Si = self.lemma2_14(S, r0, t)
        sumsets = list()
        sumsets.append(divideAndConquerSumSet(Si[0], t))
        if self.benchmarkMode and sumsets[0][-1] == t:
            return [t]
        for i in range(1, len(Si)):
            if len(Si[i]) > 0:
                if len(Si[i]) <= self.thresholdValue:
                    sumsets.append(list(bruteForceSolve(Si[i], t)))
                elif len(Si[i] <= self.dnqThreshold):
                    sumsets.append(divideAndConquerSumSet(Si[i], t))
                else:
                    sumsets.append(self.lemma2_13(Si[i], r0 * 2 ** (i - 1) + 1, r0 * 2 ** i, t))
            if self.benchmarkMode and sumsets[-1][-1] == t:
                return [t]
        return sumsets

    def lemma_experiment(self, S, t):
        minval = min(S)
        maxval = max(S)
        return [0] + list(set(self.lemma2_13(S, minval, maxval, t)))

    def lemma2_16(self, S, t):
        greaterThanTHalf = [x for x in S if x > math.floor(t/2)]
        greaterThanTHalf.append(0)
        usingLessThanTHalf = [x for x in S if x <= math.floor(t/2)]
        if len(usingLessThanTHalf) == 0:
            return greaterThanTHalf
        n = len(usingLessThanTHalf)
        r0 = round(min(t ** (2 / 3), t / math.sqrt(n)))
        sums = self.lemma2_15(usingLessThanTHalf, r0, t)
        if self.benchmarkMode and sums == [t]:
            return [t]
        for i in range(1, len(sums)):
            sums[0] = sumSetNotComplex(sums[0], sums[i], t)
            if self.benchmarkMode and sums[0][-1] == t:
                return [t]
        return sumSetNotComplex(sums[0], greaterThanTHalf, t)

    def lemma2_14(self, S, r0, t):
        """
        This could likely be done faster, but as of now it is not a problem
        :param S:
        :param r0:
        :param t:
        :return:
        """
        S = np.array(S)
        n = len(S)
        Si = list()
        Si.append(S[(0 <= S) & (S <= r0)])
        r_last = r0
        for i in range(1, math.ceil(math.log2(t)) - 1):
            r_i = 2 ** i * r0
            Si.append(S[(r_last + 1 <= S) & (S <= r_i)])
            r_last = r_i
        return Si

    def lemma2_13(self, S, low, high, target):
        removeOffset = lambda x: x - low
        S = np.array(list(map(removeOffset, S)))
        addOffsetBack = lambda ij: (ij[0] + low * ij[1], ij[1])

        a = math.ceil(target / low)

        sums = np.array([addOffsetBack(x)[0] for x in self.lemma2_12(S, high, a, target)])

        return sums[sums <= target]

    def lemma2_12(self, S, high, a, target):
        if len(S) == 1:
            return [(S[0], 1)]
        if len(S) <= self.recursionLimit:
            return smallSetSolver.subsetSumDP(S, target)
        leftSum, rightSum = self.lemma2_12(S[1::2], high, a, target), self.lemma2_12(S[0::2], high, a, target)
        return self.lemma2_11(leftSum, rightSum, a, target)

    def lemma2_11(self, B, C, a, t):
        if len(B) == 0:
            return C
        if len(C) == 0:
            return B
        X, Y = ListToMatrix(B), ListToMatrix(C)
        #convolver = fftconvolver()
        #invMAMB = convolver(X, Y)
        #pyfftw.FFTW(X, Y, axes=(1,2), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD')
        sumset = fftconvolve(X, Y)
        #add 1:: to remove (0,0)
        return list(MatrixToList(sumset, a, int(t)))[1::]
