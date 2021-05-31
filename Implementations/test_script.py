import time
import sys

sys.path.append('C:\\Users\\mikke\\PycharmProjects\\Benchmark-SubsetSum\\pythonSol')

from Implementations.FasterSubsetSum.RandomizedMultiThreaded import RandomizedMultiThreaded
import numpy as np

from Implementations.FasterSubsetSum.RandomizedMultiThreadedVer2 import RandomizedMultiThreadedVer2
from benchmarks.test_distributions import Distributions

runner = RandomizedMultiThreadedVer2(False, 1, 8)

if __name__ == '__main__':

    a, T = Distributions.evenDistribution(1000)
    start = time.time()
    runner.fasterSubsetSum(a, T, 0.00001)
    end = time.time()
    print('solved in', end- start)