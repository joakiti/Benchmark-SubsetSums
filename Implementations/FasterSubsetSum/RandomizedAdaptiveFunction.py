from Implementations.helpers.ComputeLayersByMass import ComputeLayersByMass
from Implementations.helpers.Helper import padWithZero, ListToPolynomial
from Implementations.Interfaces.IRandomizedWithPartitionFunction import IRandomizedWithPartitionFunction
import numpy as np

class RandomizedAdaptiveFunction(IRandomizedWithPartitionFunction):

    def __init__(self, debug, repetitions):
        self.analyzer = None
        super().__init__(debug,
                         None,
                         None,
                         None,
                         repetitions)

    def fasterSubsetSum(self, Z, t, delta):
        self.analyzer = ComputeLayersByMass(Z, t)
        super().__init__(self.debug,
                         self.analyzer.layerFunction(),
                         self.analyzer.intervalFunction(),
                         self.analyzer.retrieveSolutionSizeFunction(),
                         self.repetitions)
        return super().fasterSubsetSum(Z, t, delta)

    def partitionIntoLayers(self, Z, n, t):
        Zi = [self.layerFunction(Z, n, t, sample) for sample in self.intervalFunction(Z, n, t)]

        if self.debug:
            self.layerInformation = list()
            sampleKey = 0
            for sample in self.intervalFunction(Z, n, t):
                self.layerInformation.append((len(Zi[sampleKey]), sample[1]))
                sampleKey += 1
            self.layerInformation.append((0, 0))

        for i in range(len(Zi)):
            if len(Zi[i]) < 1:
                Zi[i] = padWithZero([])

        Zi = np.array(list(map(ListToPolynomial, Zi)))
        return Zi