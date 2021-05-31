import math
from collections import defaultdict
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import numpy as np


class ComputeKMeansGrouping:

    def __init__(self, values):
        for i in range(len(values)):
            values[i] = int(values[i])
        self.values = sorted(values)
        self.OPT = None
        self.T = None

    def computeClusters(self, noClusters):
        # Compute prefix sums:
        noClusters = int(math.ceil(noClusters))
        m = len(self.values) - 1
        n = len(self.values)
        xSquared = [0] * n
        xSquared[0] = self.values[0] ** 2
        for i in range(1, len(self.values)):
            xSquared[i] = xSquared[i - 1] + self.values[i] ** 2

        x = [0] * n
        x[0] = self.values[0]
        for i in range(1, len(self.values)):
            x[i] = x[i - 1] + self.values[i]

        def µ(i, j):
            if i > 0:
                ans = (x[j] - x[i - 1]) / (j - i + 1)
            else:
                ans = x[j] / (j - i + 1)
            return ans

        def CC(i, j):
            if i > 0:
                ans = (j - i + 1) * (µ(i, j) ** 2) - 2 * µ(i, j) * (x[j] - x[i-1]) + (xSquared[j] - xSquared[i-1])
            else:
                ans = (j - i + 1) * (µ(i, j) ** 2) - 2 * µ(i, j) * x[j] + xSquared[j]
            return ans

        OPT = defaultdict(dict)
        T = defaultdict(dict)
        for j in range(0, n):
            OPT[1][j] = CC(0, j)

        for i in range(2, noClusters + 1):
            OPT[i][0] = 0.0
            T[i][0] = 0
            for j in range(1, n):
                mins = list()
                for arg in range(1, j + 1):
                    mins.append(OPT[i - 1][arg - 1] + CC(arg, j))
                OPT[i][j] = min(mins)
                argmin = np.argmin(mins) + 1
                T[i][j] = argmin
        self.T = T

    def clusters(self, numberOfClusters):
        if self.T is None:
            raise Exception("Clusters not computed")
        else:
            result = []
            end = len(self.values) - 1
            while (numberOfClusters > 1):
                leftMostStartingPoint = self.T[numberOfClusters][end]
                result.append(self.values[leftMostStartingPoint:end+1])
                numberOfClusters -= 1
                end = leftMostStartingPoint - 1
            result.append(self.values[0:end + 1])
            result.reverse()
            return result

    def computerClustersLibrary(self, a, numberOfClusters):
        """
        Based upon https://pyclustering.github.io/docs/0.10.1/html/index.html
        :param a:
        :param numberOfClusters:
        :return:
        """
        # Prepare initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(a, numberOfClusters).initialize()

        # Create instance of K-Means algorithm with prepared centers.
        kmeans_instance = kmeans(a, initial_centers)

        # Run cluster analysis and obtain results.
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()


