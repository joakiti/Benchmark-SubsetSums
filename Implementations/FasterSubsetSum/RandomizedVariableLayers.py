import math

import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers, padWithZero
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedVariableExponentialLayers(NearLinearBase):

    def __init__(self, debug, variable, label, repetitions):
        super().__init__(debug, label, repetitions)
        self.exponent = variable

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        Z = np.array(Z)
        Zi, S = self.partitionIntoLayers(Z, n, t)
        S = ListToPolynomial(S)
        for i in range(0, len(Zi)):
            z = np.array(Zi[i])
            if len(z) > 1:
                Si = self.ColorCodingLayer(z, t, pow(self.exponent, i + 1) - 1,
                                           delta / (math.ceil(math.log(n, self.exponent))),
                                           high=pow(self.exponent, i) if i != len(Zi) - 1 else (
                                           self.exponent ** i, "Last is zero"))
                S = self.sumSet(Si, S, t)
        return toNumbers(S)

    def partitionIntoLayers(self, Z, n, t):
        # Have to remove the values greater than tHalf..
        greaterThanTHalf = list()
        lessThanTHalf = list()

        for z in Z:
            if z > math.ceil(t/2):
                greaterThanTHalf.append(z)
            else:
                lessThanTHalf.append(z)

        greaterThanTHalf = np.array(greaterThanTHalf)
        lessThanTHalf = np.array(lessThanTHalf)

        Zi = [lessThanTHalf[(t / pow(self.exponent, i) <= lessThanTHalf) & (lessThanTHalf < t / pow(self.exponent, i - 1))] for i in
              range(1, math.ceil(math.log(n, self.exponent)))]

        Zi.append(Z[(0 <= Z) & (Z < t / pow(self.exponent, math.ceil(math.log(n, self.exponent)) - 1))])

        if self.debug:
            self.layerInformation = list()
            for i in range(len(Zi)):
                self.layerInformation.append((len(Zi[i]), min(t / pow(self.exponent, i), t/2)))
            self.layerInformation.append((0, 0))
            self.layerInformation.append((len(greaterThanTHalf), t))
        for i in range(len(Zi)):
            if len(Zi[i]) < 1:
                Zi[i] = padWithZero([])
        if len(greaterThanTHalf) < 1:
            greaterThanTHalf = padWithZero(greaterThanTHalf)

        Zi = np.array(list(map(ListToPolynomial, Zi)))
        return Zi, greaterThanTHalf
