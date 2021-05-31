import math
import random
import time
from collections import defaultdict
from itertools import takewhile

import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int
from scipy.signal import fftconvolve

from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.helpers.Helper import ListToPolynomial, toNumbers, padWithZero, sumSetIntMult, \
    divideAndConquerSumSet, bruteForceSolve


class NearLinearBase:
    eps = 0.0001  # account for floating error

    def __init__(self, debug, label, repetitions=0, delta = 0.0001, benchmarkMode = False, bruteForceThreshold = 3, dnQThreshold = 5):
        """
        It is possible to use default number of repetitions by setting to 0.
        :param debug:
        :param repetitions:
        """
        self.debug = debug
        self.layerInformation = list()
        self.bucketInformation = list()
        self.benchmarkMode = benchmarkMode
        self.bruteForceThreshold = bruteForceThreshold
        self.dnQThreshold = dnQThreshold
        self.repetitions = repetitions
        self.label = label
        self.delta = delta
        self.n = 0

    def getLayerInformation(self):
        try:
            return self.layerInformation
        except NameError as e:
            return []

    def getBucketInformation(self):
        try:
            return self.bucketInformation
        except NameError as e:
            return []

    def partitionSetIntoKRegularNumbers(self, Z, k):
        k = math.ceil(k)
        partition = defaultdict(list)
        listUsed = set()
        for i in Z:  # Ignore 0 component with 1:
            goesTo = np.random.randint(0, k)
            partition[goesTo].append(i)
            listUsed.add(goesTo)
        return [partition[x] for x in listUsed]

    def color_coding(self, Z, t, k, delta):
        if len(Z) == 1:
            return [1]
        if self.repetitions == 0:
            # if math.log(t, 1.05) >= self.n:
            #     repetitions = 5
            # else:
                #repetitions = 1
            repetitions = math.log(1.0 / delta, 4.0 / 3.0)
        else:
            repetitions = min(self.repetitions, math.log(1.0 / delta, 4.0 / 3.0))
        S = []
        newValueFound = 0
        j = math.ceil(repetitions)
        realValues = toNumbers(Z)[1:]
        if len(realValues) <= self.bruteForceThreshold:
            return ListToPolynomial(list(bruteForceSolve(realValues, t)))
            # solveForLargePartition = FastIntegersFromGit()
        if len(realValues) <= self.dnQThreshold:
            return ListToPolynomial(divideAndConquerSumSet(realValues, t))
        low = min(realValues)
        if False: #low <= t/4
            removeOffset = lambda x: (x - low, 1)
            addOffsetBack = lambda ij: ij[0] + low * ij[1]
            j = math.ceil(repetitions)
            while j >= 0 and not newValueFound == 5:
                start = time.time()
                removedOffset = list(map(removeOffset, realValues))
                partition = self.partitionSetIntoKRegularNumbers(removedOffset, k * k)
                sumset = []
                for i in range(0, len(partition)):
                    nextSet = FastIntegersPersonal.lemma2_11(list(map(removeOffset, sumset)), partition[i], k, t)
                    sumset = takewhile(lambda x: x <= t, list(map(addOffsetBack,nextSet)))
                    end = time.time()
                    print('used time to combine all sumsets USING MINBOUND', end - start)
                if len(S) == 0:
                    S = set(sumset)
                else:
                    newValues = set(sumset)
                    hasAddedNewValue = False
                    for v in newValues:
                        if v not in S:
                            S.add(v)
                            hasAddedNewValue = True
                    if hasAddedNewValue:
                        newValueFound = 0
                    if not hasAddedNewValue:
                        newValueFound += 1
                j -= 1
        else:
            while j >= 0: #and newValueFound <= 5:
                start = time.time()
                partition = self.partitionSetIntoK(Z, k * k)  # max(int(k*k//2), 2))
                partitionA = partition[0::2]
                partitionB = partition[1::2]
                sumsetA = partitionA[0]
                for i in range(1, len(partitionA)):
                    sumsetA = self.sumSet(sumsetA, partitionA[i], t)

                if len(partitionB) > 0:
                    sumsetB = partitionB[0]
                    for i in range(1, len(partitionB)):
                        sumsetB = self.sumSet(sumsetB, partitionB[i], t)
                    sumsetA = self.sumSet(sumsetA, sumsetB, t)
                end = time.time()
                if len(S) == 0:
                    S = sumsetA
                else:
                    newValues = toNumbers(sumsetA)
                    oldValues = set(toNumbers(S))
                    hasAddedNewValue = False
                    for v in newValues:
                        if v not in oldValues:
                            oldValues.add(v)
                            hasAddedNewValue = True
                    if hasAddedNewValue:
                        newValueFound = 0
                    if not hasAddedNewValue:
                        newValueFound += 1
                    S = ListToPolynomial(list(oldValues))
                j -= 1
        return list(S)

    def fasterSubsetSum(self, Z, t, delta=0.0001):
        if self.delta != 0.0001:
            delta = self.delta
        n = len(Z)
        self.n = n
        Z = np.array(Z)
        Zi = self.partitionIntoLayers(Z, n, t)
        S = [1]
        if len(Zi[0]) > 1:
            S = Zi[0]
            if len(Zi) == 1:
                S = self.ColorCodingLayer(S, t, len(Z), delta / (math.ceil(math.log2(n))))
        for i in range(1, len(Zi)):
            z = np.array(Zi[i])
            if len(z) > 1:
                start = time.time()
                Si = self.ColorCodingLayer(z, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                           high=pow(2, i) if i != len(Zi) - 1 else (2 ** i, "Last is zero"))
                S = self.sumSet(Si, S, t)
                if self.benchmarkMode and len(S) == t +1 and S[t] == 1:
                    return [t]
                end = time.time()
        return toNumbers(S)

    def partitionSetIntoK(self, Z, k):
        k = math.ceil(k)
        partition = np.zeros((k, len(Z)), dtype=np.dtype('u1')) #Otherwise we use too much memory.
        listUsed = set()
        for i in np.nonzero(Z)[0][1:]:  # Ignore 0 component with 1:
            goesTo = np.random.randint(0, k)
            partition[goesTo][i] = 1
            partition[goesTo][0] = 1
            listUsed.add(goesTo)
        return [partition[x][:max(np.nonzero(partition[x])[0])+1] for x in listUsed]

    def ColorCodingLayer(self, Z, t, l, delta, high=(1, 0)):
        if len(Z) <= 1:
            return [1]
        divisor = math.log2(l / delta)
        if l < divisor:
            return self.color_coding(Z, t, l, delta)
        m = self.roundToPowerOf2(l / divisor)
        Z = self.partitionSetIntoK(Z, m)
        m = self.roundToPowerOf2(len(Z))
        while len(Z) < m:
            Z.append([1])
        gamma = math.ceil(6 * divisor)
        threshold = math.ceil(min(t, 2*gamma*t / l))
        if gamma > l:
            gamma = l
        if self.debug:
            if type(high) is not tuple:
                steps = (t // high - (t // (l + 1))) / m
                self.bucketInformation.append(
                    [(len(toNumbers(Z[j])), j * steps + steps / 2, t // high) for j in range(len(Z))])
            if type(high) is tuple:
                steps = t // high[0] / m
                self.bucketInformation.append(
                    [(len(toNumbers(Z[j])), j * steps + steps / 2, t // high[0]) for j in range(len(Z))])
        S = [self.color_coding(Z[j], threshold, round(gamma), delta / l) for j in range(m)]

        for h in range(1, int(math.log2(m)) + 1):
            threshold = min(pow(2, h) * 2 * gamma * t / l, t)
            for j in range(1, int(m / pow(2, h)) + 1):
                S[j - 1] = self.sumSet(S[2 * j - 1 - 1], S[2 * j - 1], threshold)
        S[0] = np.array(S[0])
        return S[0]

    @staticmethod
    def roundToPowerOf2(m):
        return pow(2, math.ceil(math.log2(m)))

    def partitionIntoLayers(self, Z, n, t):
        Zi = [Z[(t / pow(2, i) <= Z) & (Z < t / pow(2, i - 1))] for i in
              range(1, math.ceil(math.log2(n)))]
        Zi.append(Z[(0 <= Z) & (Z < t / pow(2, math.ceil(math.log2(n)) - 1))])
        if self.debug:
            self.layerInformation = list()
            for i in range(len(Zi)):
                self.layerInformation.append((len(Zi[i]), t / pow(2, i)))
            self.layerInformation.append((len(Zi[len(Zi) - 1]), 0))
        for i in range(len(Zi)):
            if len(Zi[i]) < 1:
                Zi[i] = padWithZero([])
        Zi = np.array(list(map(ListToPolynomial, Zi)))
        return Zi

    @staticmethod
    def sumSet(A, B, threshold):
        eps = 0.0001  # account for floating error
        AsumsetB = fftconvolve(A, B)
        return np.select([AsumsetB[:int(threshold + 1)] > eps], [1])

    @staticmethod
    def sumSetIntMult(A, B, t):
        valsA = np.nonzero(A[:t + 1])[0]
        valsB = np.nonzero(B[:t + 1])[0]
        t = max(max(A)+max(B), t)
        if len(A) <= 1:
            return B
        if len(B) <= 1:
            return A

        log2T = int(math.ceil(math.log2(t)))
        lenA = (log2T + 1) * len(A)
        lenB = (log2T + 1) * len(B)
        a = [0] * lenA
        for i in valsA * (log2T + 1):
            a[i] = 1
        b = [0] * lenB
        for i in valsB * (log2T + 1):
            b[i] = 1
        f = ba2int(bitarray(a))
        g = ba2int(bitarray(b))
        z = bitarray(np.binary_repr(f * g)[:(t + 1) * (log2T + 1)]).to01()
        sumsets = [1 if int(z[x:x + log2T + 1]) > 0 else 0 for x in range(0, len(z), log2T + 1)]
        return sumsets
