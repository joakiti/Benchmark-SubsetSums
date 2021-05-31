import operator
import time
from _operator import itemgetter
from collections import defaultdict
from datetime import datetime

from matplotlib import pyplot as plt

from Implementations.FasterSubsetSum.RandomizedMultiThreaded import RandomizedMultiThreaded
from Implementations.FasterSubsetSum.RandomizedMultiThreadedVer2 import RandomizedMultiThreadedVer2
from benchmarks.benchmark_randomized_linear_layers import mkdir_p, plotRunningTimes, plotDistributions
from benchmarks.test_distributions import Distributions


def timeIt(func):
    start = time.time()
    val = func()
    end = time.time()
    clock = end - start
    return clock, val


def test_compare_all_algorithms():
    title = "../plots/" + datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
    multiThreaded2ThreadsFSS = RandomizedMultiThreadedVer2(True, 0, 8)
    multiThreaded3ThreadsFSS = RandomizedMultiThreaded(True, 0, 3)
    multiThreaded4ThreadsFSS = RandomizedMultiThreaded(True, 0, 4)
    multiThreaded6ThreadsFSS = RandomizedMultiThreaded(True, 0, 6)
    fasterSubsetSum = [
        # Define algoritmhs
        # multiThreaded6ThreadsFSS,
        multiThreaded2ThreadsFSS
    ]
    runningTimes = defaultdict(lambda: defaultdict(list))
    layers = defaultdict(lambda: defaultdict(list))
    testInput = defaultdict(list)
    testInputRatio = defaultdict(list)
    solutionFail = defaultdict(lambda: defaultdict(list))
    distributionFunctions = Distributions().allDistributions()
    sizeOfInput = defaultdict(list)
    tInput = defaultdict(list)
    for i in range(2, 4):
        for distributionFunction in distributionFunctions:
            a, T = distributionFunction(i)
            delta = 0.00001
            testInput[distributionFunction] = a, T
            testInputRatio[distributionFunction.__name__].append((a, T))
            print("Test {}, distribution {}: List of size {}, T={}".format(i, distributionFunction.__name__, len(a),
                                                                           T))
            solutions = defaultdict()
            # cluster = ComputeKMeansGrouping(a)
            # cluster.computeClusters(Distributions.noClusters(i))
            # self.clusterFSS = RandomizedClusters(True, kMeansSolver=cluster)
            # self.clusterFSS.i = i
            for algorithm in fasterSubsetSum:
                runner = algorithm
                totalClock = 0
                results = []
                for j in range(3):
                    clock, results = timeIt(lambda: runner.fasterSubsetSum(a, T, delta))
                    totalClock += clock
                totalClock /= 3
                print("%s solved on avg in %.2f" % (algorithm, totalClock))
                runningTimes[distributionFunction.__name__][algorithm].append(totalClock)
                solutions[algorithm] = results
                if runner.debug:
                    layers[algorithm][distributionFunction.__name__] = runner.getLayerInformation()

            sizeOfInput[distributionFunction].append(len(a))
            tInput[distributionFunction].append(T)
    mkdir_p(title + "/runningTimes")
    plotRunningTimesWork(title, distributionFunctions, runningTimes, sizeOfInput, tInput)


def plotRunningTimesWork(title, distributionFunctions, runningTimes, sizeOfInput, tInput):
    for i in range(len(distributionFunctions)):
        fig, ax = plt.subplots()
        distributionFunction = distributionFunctions[i]
        ax.set_title(distributionFunction.__name__, fontsize=12)
        for algorithm in runningTimes[distributionFunction.__name__].keys():
            sizeWithTime = list(
                zip(sizeOfInput[distributionFunction], runningTimes[distributionFunction.__name__][algorithm]))
            sizeWithTime.sort(key=operator.itemgetter(0))
            ax.plot(list(map(itemgetter(0), sizeWithTime)),
                    list(map(itemgetter(1), sizeWithTime)), label='p = 2')
            ax.set_ylabel('Time (seconds)')
            ax.set_xlabel('Size of input')
            # ax2 = ax.twinx()
            # ax2.set_ylabel('T')
            # ax2.tick_params(axis='y', colors='gray')
            # ax2.spines['right'].set_color('gray')
            # ax2.yaxis.label.set_color('gray')
            # ax2.scatter(sizeOfInput[distributionFunction], tInput[distributionFunction], c='gray')
        ax.grid()
        ax.legend()
        fig.tight_layout()
        fig.savefig(title + "/runningTimes/" + distributionFunction.__name__)
        fig.show()


if __name__ == '__main__':
    test_compare_all_algorithms()
