from abc import ABC, abstractmethod
import numpy as np

from Implementations.helpers.Helper import toNumbers, padWithZero, ListToPolynomial
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class IRandomizedWithPartitionFunction(NearLinearBase, ABC):

    def __init__(self, debug, layerFunction, intervalFunction, solutionSizeFunction, repetitions):
        super().__init__(debug, 'adaptive', repetitions)
        self.layerFunction = layerFunction
        self.intervalFunction = intervalFunction
        self.solutionSizeFunction = solutionSizeFunction

    @abstractmethod
    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        Z = np.array(Z)
        Zi = self.partitionIntoLayers(Z, n, t)
        dontDoLast = 0
        S = [1]
        if len(Zi[len(Zi) - 1]) > t // 2:
            S = Zi[len(Zi) - 1]
            dontDoLast = 1
        for i in range(0, len(Zi) - dontDoLast):
            z = np.array(Zi[i])
            if len(z) > 1:
                Si = self.ColorCodingLayer(z, t, self.solutionSizeFunction(Z, n, t, i), delta / len(Zi),
                                           high=pow(2, i) if i != len(Zi) - 1 else (
                                               2 ** i, "Last is zero"))  # Just adding some garbage, does not matter.
                S = self.sumSet(Si, S, t)
        return toNumbers(S)

    @abstractmethod
    def partitionIntoLayers(self, Z, n, t):
        Zi = [self.layerFunction(Z, n, t, sample) for sample in self.intervalFunction(Z, n, t)]

        if self.debug:
            self.layerInformation = list()
            sampleKey = 0
            for sample in self.intervalFunction(Z, n, t):
                self.layerInformation.append((len(Zi[sampleKey]), sample[1]))
                sampleKey += 1
            self.layerInformation.append((0, 0))

        for i in range(len(Zi)):
            if len(Zi[i]) < 1:
                Zi[i] = padWithZero([])

        Zi = np.array(list(map(ListToPolynomial, Zi)))
        return Zi
