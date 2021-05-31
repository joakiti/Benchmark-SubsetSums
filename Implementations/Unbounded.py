import math

from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.helpers.Helper import sumSetNotComplex, padWithZero
from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm


class Unbounded(IDeterministicAlgorithm):

    @classmethod
    def run(cls, values, target):
        return cls.unboundedSumSet(values, target)

    @classmethod
    def unboundedSumSet(cls, values, t):
        n = len(values)
        S = [[] for i in range(math.ceil(math.log2(n)) + 1)]
        DPAlg = DynamicProgramming('.', benchmarkMode=True)
        S[0] = DPAlg.run(values, t // n)
        for i in range(1, math.ceil(math.log2(n)) + 1):
            t_i = math.pow(2, i) * t / n
            S[i] = sumSetNotComplex(sumSetNotComplex(S[i - 1], S[i - 1], t_i), padWithZero(values), t_i)
        return [x for x in S[math.ceil(math.log2(n))] if x <= t]
