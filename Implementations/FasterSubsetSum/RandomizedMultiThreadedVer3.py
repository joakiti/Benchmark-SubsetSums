import math
import os
from collections import defaultdict

import threading
import concurrent.futures
import time
from concurrent.futures._base import as_completed
from multiprocessing import Process
from parfor import parfor

import numpy as np
import shutil
from joblib import Parallel, delayed, dump, load, parallel_backend
from joblib.externals.loky import set_loky_pickler
from scipy.signal import fftconvolve

from Implementations.helpers.Helper import toNumbers, ListToPolynomial
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase

class RandomizedMultiThreadedVer3(NearLinearBase):

    def __init__(self, debug, repetitions, threads):
        super().__init__(debug, repetitions)
        self.threads = threads

    def sumSet(self, A, B, threshold):
        Apoly = ListToPolynomial(A)
        Bpoly = ListToPolynomial(B)
        eps = 0.0001  # account for floating error
        AsumsetB = fftconvolve(Apoly, Bpoly)
        return toNumbers(np.select([AsumsetB[:int(threshold + 1)] > eps], [1]))

    def color_coding(self, Z, t, k, delta):
        if len(Z) == 1:
            return [0, Z[0]]
        if self.repetitions == 0:
            # if math.log(t, 1.05) >= self.n:
            #     repetitions = 5
            # else:
            # repetitions = 1
            repetitions = math.log(1.0 / delta, 4.0 / 3.0)
        else:
            repetitions = self.repetitions
        S = [[] for _ in range(math.ceil(repetitions))]
        for j in range(0, math.ceil(repetitions)):
            partitions = self.partitionSetIntoK(Z, k * k)
            if len(partitions) == 1:
                return partitions[0]
            sumset = partitions[0]
            for i in range(1, len(partitions)):
                sumset = self.sumSet(sumset, partitions[i], t)
            S[j] = sumset
            # partitionPerThread = divmod(len(partitions), self.threads)
            # index = 0
            # threadListWork = list()
            # for i in range(self.threads):
            #     if i == self.threads - 1:
            #         threadListWork.append((index, index + partitionPerThread[0] + partitionPerThread[1]))
            #         continue
            #     else:
            #         threadListWork.append((index, index + partitionPerThread[0]))
            #     index = index + partitionPerThread[0]
            #
            # #(list for pair in threadListWork for list in partitions[pair[0], pair[1])
            # @parfor(threadListWork, nP=self.threads, rP=1, serial=1)
            # def combinePartitions(x):
            #     start = partitions[x[0]]
            #     for o in range(x[0], x[1]):
            #         start = self.sumSet(start, partitions[o], t)
            #     return start
            # partialSumsets = combinePartitions
            # sumset = partialSumsets[0]
            # for x in range(1, len(partialSumsets)):
            #     sumset = self.sumSet(partialSumsets[x], sumset, t)
            # S[j] = sumset
        union = set(S[0])
        for j in range(1, len(S)):
            for s in S[j]:
                union.add(s)
            # if len(S[j]) > len(union):
            #     S[j][np.nonzero(union)[0]] = 1
            #     union = S[j]
            # else:
            #     union[np.nonzero(S[j])[0]] = 1
        return list(union)

    def partitionSetIntoK(self, Z, k):
        k = math.ceil(k)
        partition = defaultdict(list)
        listUsed = set()
        for i in Z:  # Ignore 0 component with 1:
            goesTo = np.random.randint(0, k)
            partition[goesTo].append(i)
            listUsed.add(goesTo)
        return [partition[x] for x in listUsed]

    def ColorCodingLayer(self, Z, t, l, delta, high=(1, 0)):
        if len(Z) == 1:
            return [0, Z[0]]
        divisor = math.log2(l / delta)
        if l < divisor:
            return self.color_coding(Z, t, l, delta)
        m = self.roundToPowerOf2(l / divisor)
        Z = self.partitionSetIntoK(Z, m)
        m = self.roundToPowerOf2(len(Z))
        while len(Z) < m:
            Z.append([1])
        gamma = 6 * divisor
        if gamma > l:
            gamma = l

        @parfor(range(m), nP=self.threads, rP=1, serial=1)
        def combinePartitions(i):
            return self.color_coding(Z[i], 2 * t * gamma / l, round(gamma), delta / l)
        S = combinePartitions

        for h in range(1, int(math.log2(m)) + 1):
            threshold = min(pow(2, h) * 2 * gamma * t / l, t)
            for j in range(1, int(m / pow(2, h)) + 1):
                S[j - 1] = self.sumSet(S[2 * j - 1 - 1], S[2 * j - 1], threshold)
        S[0] = np.array(S[0])
        return S[0]

    def partitionIntoLayers(self, Z, n, t):
        Zi = [Z[(t / pow(2, i) <= Z) & (Z < t / pow(2, i - 1))] for i in
              range(1, math.ceil(math.log2(n)))]
        Zi.append(Z[(0 <= Z) & (Z < t / pow(2, math.ceil(math.log2(n)) - 1))])
        if self.debug:
            self.layerInformation = list()
            for i in range(len(Zi)):
                self.layerInformation.append((len(Zi[i]), t / pow(2, i)))
            self.layerInformation.append((len(Zi[len(Zi) - 1]), 0))
        return Zi

    def fasterSubsetSum(self, Z, t, delta):
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
            if len(z) > 0:
                start = time.time()
                Si = self.ColorCodingLayer(z, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                           high=pow(2, i) if i != len(Zi) - 1 else (2 ** i, "Last is zero"))
                S = self.sumSet(Si, S, t)
                end = time.time()
                print('solved layer ', i, 'in ', end - start)
        return toNumbers(S)
