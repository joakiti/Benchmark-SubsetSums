import math

import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers, padWithZero
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedLinearLayers(NearLinearBase):

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        Z = np.array(Z)
        Zi, largerThanTHalf = self.partitionIntoLayers(Z, n, t)
        S = ListToPolynomial(largerThanTHalf)
        step = t / math.log2(n)
        for i in range(0, len(Zi)):
            z = np.array(Zi[i])
            if len(z) > 1 and i == 0:
                z = ListToPolynomial(z)
                Si = self.ColorCodingLayer(z, t, len(Zi[i]), delta / len(Zi))
                S = self.sumSet(Si, S, t)
            elif len(z) > 1:
                z = ListToPolynomial(z)
                Si = self.ColorCodingLayer(z, t, min(t/step*i, len(Zi[i])), delta / len(Zi))
                S = self.sumSet(Si, S, t)

        return toNumbers(S)

    def partitionIntoLayers(self, Z, n, t):
        step = t / math.log2(n)
        Zi = list()
        last = -1
        for i in range(0, math.ceil(math.log2(n))):
            if step*(i+1) < t // 2:
                Zi.append(Z[(step * i <= Z) & (Z < step * (i + 1))])
            else:
                last = i
                break
        Zi.append(Z[(step * last <= Z) & (Z < t//2)])
        remainder = Z[(t//2 <= Z) & (Z <= t)]
        if len(remainder) < 1:
            remainder = [0]
        if self.debug:
            self.layerInformation = list()
            for i in range(len(Zi) - 2):
                self.layerInformation.append((len(Zi[i]), step * (i + 1)))
            self.layerInformation.append((len(Zi[len(Zi) - 2]), t // 2))
            self.layerInformation.append((len(remainder), t))
        for i in range(len(Zi)):
            if len(Zi[i]) < 1:
                Zi[i] = padWithZero([])
        return Zi, remainder
