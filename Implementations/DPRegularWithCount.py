from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm


class DynamicProgrammingWithCount(IDeterministicAlgorithm):

    def __init__(self):
        self.label = 'DP Regular'

    @classmethod
    def run(cls, values, target):
        return cls.subsetSumDP(values, target)

    # def runWithAlreadyKnown(self, values, target, alreadyKnown):
    #     return self.subsetSumDPWithKnown(values, target, alreadyKnown)
    #
    @classmethod
    def subsetSumDP(cls, vals, T):
        vals = list(filter(lambda v: v <= T, vals))
        OPT = [[None for i in range(T + 1)] for i in range(len(vals) + 1)]
        # Can never reach a target value >= 1, if not allowed to use any values
        for i in range(1, T + 1):
            OPT[0][i] = [(False, 0)]
        # Can always reach zero no matter what, because you simply dont use any weight.
        for i in range(len(vals) + 1):
            OPT[i][0] = [(True, 0)]

        for i in range(1, len(vals) + 1):
            tempI = OPT[i]
            tempIMinusOne = OPT[i-1]
            for j in range(1, T + 1):
                #           We use the value   Or we pick it, if the target value is greater than or equal
                # If value is already know, but we can also access it, we have to add to list
                if tempIMinusOne[j] is not None and tempIMinusOne[j][0][0]:
                    if tempIMinusOne[j - vals[i - 1]] is not None and (j >= vals[i - 1] and tempIMinusOne[j - vals[i - 1]][0]):
                        tempI[j] = list()
                        for valuesThatAttainedNeededSum in tempIMinusOne[j - vals[i - 1]]:
                            tempI[j].append((valuesThatAttainedNeededSum[0], valuesThatAttainedNeededSum[1] + 1))
                        for previousValuesThatAlsoAttainedSum in tempIMinusOne[j]:
                            tempI[j].append(previousValuesThatAlsoAttainedSum)
                    else:
                        tempI[j] = tempIMinusOne[j]
                elif tempIMinusOne[j - vals[i - 1]] is not None and (j >= vals[i - 1] and tempIMinusOne[j - vals[i - 1]][0][0]):
                    tempI[j] = list()
                    for valuesThatAttainedNeededSum in tempIMinusOne[j - vals[i - 1]]:
                        tempI[j].append((valuesThatAttainedNeededSum[0], valuesThatAttainedNeededSum[1] + 1))
        result = list()
        for value, listOfAttainment in enumerate(OPT[len(vals)]):
            if listOfAttainment is None:
                continue
            else:
                for sums in listOfAttainment:
                    result.append((value, sums[1]))
        return result#[(target, achieved[1]) for target, achieved in enumerate(OPT[len(vals)]) if achieved is not None and achieved[0]]

    # @classmethod
    # def subsetSumDPWithKnown(cls, vals, T, alreadyKnown):
    #     vals = list(filter(lambda v: v <= T, vals))
    #     OPT = [[None for i in range(T + 1)] for i in range(len(vals) + 1)]
    #     # Can never reach a target value >= 1, if not allowed to use any values
    #     for i in range(1, T + 1):
    #         OPT[0][i] = (False, 0)
    #     # Can always reach zero no matter what, because you simply dont use any weight.
    #     for i in range(len(vals) + 1):
    #         OPT[i][0] = (True, 0)
    #     for val in alreadyKnown:
    #         if val != 0:
    #             OPT[0][val] = (True, 1)
    #     for i in range(1, len(vals) + 1):
    #         tempI = OPT[i]
    #         tempIMinusOne = OPT[i-1]
    #         for j in range(1, T + 1):
    #             #           We use the value   Or we pick it, if the target value is greater than or equal
    #             if tempIMinusOne[j] is not None and tempIMinusOne[j][0] :
    #                 tempI[j] = tempIMinusOne[j]
    #             elif tempIMinusOne[j - vals[i - 1]] is not None and (j >= vals[i - 1] and tempIMinusOne[j - vals[i - 1]][0]):
    #                 tempI[j] = (tempIMinusOne[j - vals[i - 1]][0], tempIMinusOne[j - vals[i - 1]][1]+1)
    #
    #     return [(target, achieved[1]) for target, achieved in enumerate(OPT[len(vals)]) if achieved is not None and achieved[0]]
