from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm


class DynamicProgrammingRegular(IDeterministicAlgorithm):

    def __init__(self):
        self.label = 'DP Regular'

    @classmethod
    def run(cls, values, target):
        return cls.subsetSumDP(values, target)

    @classmethod
    def subsetSumDP(cls, vals, T):
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

        return [target for target, achieved in enumerate(OPT[len(vals)]) if achieved]
