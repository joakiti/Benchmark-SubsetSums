import math

import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedBaseWithShortcuts(NearLinearBase):

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        Z = np.array(Z)

        Zi = self.partitionIntoLayers(Z, n, t)
        S = [1]
        if len(Zi[0]) > 1:
            S = Zi[0]
        for i in range(0, min(len(Zi), 5)):
            z = np.array(Zi[i])
            if len(z) > 1:
                Si = self.ColorCodingLayer(z, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                           high=pow(2, i) if i != len(Zi) - 1 else (2 ** i, "Last is zero"))
                S = self.sumSet(Si, S, t)
        z = set([item for remainder in Zi[5:] for item in toNumbers(remainder)])
        if len(z) > 1:
            z = ListToPolynomial(z)
            i = 5
            Si = self.ColorCodingLayer(z, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                       high=pow(2, i) if i != len(Zi) - 1 else (2 ** i, "Last is zero"))
            S = self.sumSet(Si, S, t)
        return toNumbers(S)

