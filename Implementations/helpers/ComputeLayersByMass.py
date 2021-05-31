import math

import numpy as np
import scipy as scipy
from scipy import stats


class ComputeLayersByMass:

    def __init__(self, distribution, T):
        self.input = list(filter(lambda x: x < T // 2, distribution))
        if (len(self.input) < 1):
            self.input.append(0)
        self.layerValues = []
        n = len(self.input)
        hist = np.histogram(self.input, bins=T)
        self.hist_dist = scipy.stats.rv_histogram(hist)

        samples = max(math.ceil(math.log2(n)), 1)
        massPerSample = (n / samples) / n
        self.lowerBoundOfPartition = [massPerSample * i for i in range(samples)]
        self.UpperBoundOfPartition = [massPerSample * i for i in range(1, samples + 1)]
        self.layerValues = self.hist_dist.ppf(self.lowerBoundOfPartition)
        self.layerValues = self.layerValues.tolist()
        self.layerValues = np.append(self.layerValues, (T // 2, T))

    def retrieveSolutionSizeFunction(self):
        def solutionSizeFunction(Z, n, t, i):
            return math.ceil(t / max(self.layerValues[i], 1))

        return solutionSizeFunction

    def layerFunction(self):
        def layering(Z, n, t, sample):
            return Z[(sample[0] <= Z) & (Z <= sample[1])]

        return layering

    def intervalFunction(self):
        # getting data of the histogram
        def intervalFunction(Z, n, t):
            intervalsLowThenUpperTuples = list(zip(
                self.hist_dist.ppf(self.lowerBoundOfPartition),
                self.hist_dist.ppf(self.UpperBoundOfPartition)))
            intervalsLowThenUpperTuples.append((t // 2, t))
            return intervalsLowThenUpperTuples

        return intervalFunction
        # return range(math.ceil(math.log2(n) + 1))
