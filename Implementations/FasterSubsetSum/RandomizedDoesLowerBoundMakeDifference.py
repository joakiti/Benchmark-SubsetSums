import math

import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers, padWithZero
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedLowerBoundDifference(NearLinearBase):

    def __init__(self, debug):
        super().__init__(debug)

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        Z = np.array(Z)
        Z = Z[Z <= t]
        Zi, minimals = self.partitionIntoLayers(Z, n, t)
        S = [1]
        for i in range(0, len(Zi)):
            z = np.array(Zi[i])
            if len(z) > 1:
                z = ListToPolynomial(z)
                Si = self.ColorCodingLayer(z, t, int(t // max(minimals[i], 1)), delta / len(Zi))
                S = self.sumSet(Si, S, t)
        return toNumbers(S)

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
        minimals = [min(z) for z in Zi]
        return Zi, minimals
