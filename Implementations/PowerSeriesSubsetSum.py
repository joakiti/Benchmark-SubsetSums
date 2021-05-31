import functools
import math
import numpy as np

from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm
from Implementations.Interfaces.IRunnableAlgorithm import IRunnableAlgorithm
from Implementations.helpers import Helper
from Implementations.helpers.Helper import toNumbers, sumSetNotComplex, convolution, sumSet, ListToPolynomial


class PowerSeriesSubsetSum(IRunnableAlgorithm):

    def __init__(self, label):
        self.label = label

    @classmethod
    def run(self, values, target):
        return self.polynomialEncoding(values, target)

    @classmethod
    def polynomialEncoding(cls, xs, t):
        xs = set([x for x in xs if x > 0])
        B_t = [0] * (t + 1)
        # B_t[0] = 1
        for k in xs:
                for j in range(1, math.floor(t / k) + 1):  # Add one to include
                    B_t[k * j] += ((-1) ** (j - 1)) / j
        ans = cls.expCoefficients(t, B_t, min(xs))
        return [x for x in np.nonzero([np.array(ans) >= 0.5])[1]]

    @classmethod
    def expCoefficients(cls, T, f, lowerBound):
        g_i = [0] * (T + 1)
        g_i[0] = 1
        # g_i[lowerBound] = 1
        smalLSetSolver = FastIntegersFromGit()
        def Fx(r, l):
            F = [0] * (r-l+1)
            for k in range(0, r-l + 1):
                F[k] = k*f[k]
            return np.array(F)

        def Gx(m, l):
            G = [0] * (m-l+1)
            for j in range(0, m-l + 1):
                G[j] = g_i[j+l]
            return np.array(G)

        def computeRecFFT(l, r):
            if l < r:
                m = math.floor((l + r) / 2)
                if m > lowerBound:
                    if m-l < 20:
                        computeRec(l, m)
                    else:
                        computeRecFFT(l, m)
                G = Gx(m, l)
                F = Fx(r, l)
                H = convolution(F, G)
                for i in range(m + 1, r + 1):
                    g_i[i] = g_i[i] + H[i-l].real / i
                if r - m < 20:
                    computeRec(m + 1, r)
                else:
                    computeRecFFT(m + 1, r)

        def computeRec(l, r):
            if l < r:
                m = math.floor((l + r) / 2)
                computeRec(l, m)
                for i in range(m + 1, r + 1):
                    g_i[i] = g_i[i] + sum([(i - j) * f[i - j] * g_i[j] for j in range(l, m + 1)]) / i
                computeRec(m + 1, r)

        computeRecFFT(0, T)
        return g_i
