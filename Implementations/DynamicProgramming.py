import math

from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm
from Implementations.helpers.Helper import ListToPolynomial, toNumbers, sumSet


class DynamicProgramming(IDeterministicAlgorithm):

    def __init__(self, label, benchmarkMode):
        self.label = label
        self.benchmarkMode = benchmarkMode

    def run(cls, values, target):
        xs = set([x for x in values if x > 0 and x <= math.floor(target/2)])
        greaterThanTHalf = [x for x in values if x > math.floor(target/2)]
        usingLessThanTHalf = cls.subsetSumDP(xs, target)
        if cls.benchmarkMode and target in usingLessThanTHalf:
            return [target]
        if len(greaterThanTHalf) > 0:
            coefficientsGreaterThanTHalf = ListToPolynomial(greaterThanTHalf)
            return toNumbers(sumSet(ListToPolynomial(usingLessThanTHalf), coefficientsGreaterThanTHalf, target))
        else:
            return usingLessThanTHalf

    def subsetSumDP(self, vals, T):
        vals = list(filter(lambda v: v <= T, vals))
        OPT = [[None for i in range(T + 1)] for i in range(len(vals) + 1)]
        # Can never reach a target value >= 1, if not allowed to use any values
        for i in range(1, T + 1):
            OPT[0][i] = False
        # Can always reach zero no matter what, because you simply dont use any weight.
        for i in range(len(vals) + 1):
            OPT[i][0] = True

        for i in range(1, len(vals) + 1):
            tempI = OPT[i]
            tempIMinusOne = OPT[i-1]
            for j in range(1, T + 1):
                #           We use the value   Or we pick it, if the target value is greater than or equal
                tempI[j] = tempIMinusOne[j] or (j >= vals[i - 1] and tempIMinusOne[j - vals[i - 1]])
            if tempI[T] == True and self.benchmarkMode:
                return [T]

        return [target for target, achieved in enumerate(OPT[len(vals)]) if achieved]
