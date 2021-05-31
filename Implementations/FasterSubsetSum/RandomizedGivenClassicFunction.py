import math

from Implementations.Interfaces.IRandomizedWithPartitionFunction import IRandomizedWithPartitionFunction


class RandomizedGivenClassicFunction(IRandomizedWithPartitionFunction):

    def __init__(self, debug, repetitions):
        # Define functions here
        layerFunction = self.layerFunction
        intervalFunction = self.intervalFunction
        solutionSizeFunction = self.solutionSizeFunction
        super().__init__(debug, layerFunction, intervalFunction, solutionSizeFunction, repetitions)

    def layerFunction(self, Z, n, t, sample):
        if sample != math.ceil(math.log2(n)):
            return Z[(t / pow(2, sample) <= Z) & (Z < t / pow(2, sample - 1))]
        else:
            return Z[(0 <= Z) & (Z < t / pow(2, math.ceil(math.log2(n)) - 1))]

    def solutionSizeFunction(self, Z, n, t, i):
        return pow(2, i + 1) - 1

    def intervalFunction(self, Z, n, t):
        return range(math.ceil(math.log2(n)) + 1)

    def fasterSubsetSum(self, Z, t, delta):
        return super().fasterSubsetSum(Z, t, delta)

    def partitionIntoLayers(self, Z, n, t):
        return super().partitionIntoLayers(Z, n, t)
