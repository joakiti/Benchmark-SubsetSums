import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.helpers.ComputeKMeansGrouping import ComputeKMeansGrouping
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase
from benchmarks.test_distributions import Distributions


class RandomizedClusters(NearLinearBase):

    def __init__(self, debug, repetitions, kMeansSolver=None, i=-1, delta=0.0001, bruteForceThreshold=5, dnQThreshold=20):
        super().__init__(debug, 'k-means', repetitions, bruteForceThreshold=bruteForceThreshold, dnQThreshold=dnQThreshold)
        self.clustering = kMeansSolver
        self.i = i
        self.delta = delta

    def fasterSubsetSum(self, Z, t, delta=0.0001):
        if self.delta != 0.0001:
            delta = self.delta
        n = len(Z)
        Z = np.array(Z)
        Z = Z[Z <= t]
        Zi, minimals = self.partitionIntoLayers(Z, n, t)
        S = [1]
        for i in range(0, len(Zi)):
            z = np.array(Zi[i])
            if len(z) > 2: #There is a single number, and zero. Only do something then
                z = ListToPolynomial(z)
                Si = self.ColorCodingLayer(z, t, int(t // max(minimals[i], 1)), delta / len(Zi))
                S = self.sumSet(Si, S, t)
            else:
                z = ListToPolynomial(z)
                S = self.sumSet(z, S, t)
        return toNumbers(S)

    def partitionIntoLayers(self, Z, n, t):
        if self.clustering is None:
            self.clustering = ComputeKMeansGrouping(Z)
            self.clustering.computeClusters(Distributions.noClusters(self.i) - 1)
        clusterCount = self.i
        Zi = self.clustering.clusters(clusterCount)
        minimals = [min(z) for z in Zi]
        for z in Zi:
            z.append(0)
        if self.debug:
            maximals = [max(z) for z in Zi]
            self.layerInformation = set()
            for i in range(len(Zi)):
                self.layerInformation.add((len(Zi[i]), maximals[i]))
        return Zi, minimals
