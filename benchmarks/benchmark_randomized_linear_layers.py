import datetime
import math
import operator
import sys
import time
import unittest
from collections import defaultdict
from operator import itemgetter
from random import random
from unittest import TestCase
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import gridspec

from Implementations.DPRegular import DynamicProgrammingRegular
from Implementations.DivideAndConquerRunner import DivideAndConquerRunner
from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.FastIntegersFromJournal import FastIntegersFromJournal
from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.FasterSubsetSum.RandomizedBaseLessRepetitions import RandomizedBaseLessRepetitions
from Implementations.Greedy import GreedySubsetSum
from Implementations.ILPRandomRounding import ILPRandomRoundingSubsetSum
from Implementations.ILPSubsetSum import ILPSubsetSum
from Implementations.Interfaces.IRunnableAlgorithm import IRunnableAlgorithm
from Implementations.PowerSeriesSubsetSum import PowerSeriesSubsetSum
from Implementations.helpers.ComputeKMeansGrouping import ComputeKMeansGrouping
from Implementations.FasterSubsetSum.RandomizedAdaptiveFunction import RandomizedAdaptiveFunction
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase
from Implementations.FasterSubsetSum.RandomizedClusters import RandomizedClusters
from Implementations.FasterSubsetSum.RandomizedDoesLowerBoundMakeDifference import RandomizedLowerBoundDifference
from Implementations.FasterSubsetSum.RandomizedLinearLayers import RandomizedLinearLayers
from Implementations.FasterSubsetSum.RandomizedMultiThreaded import RandomizedMultiThreaded
from Implementations.FasterSubsetSum.RandomizedMultiThreadedVer2 import RandomizedMultiThreadedVer2
from Implementations.FasterSubsetSum.RandomizedMultiThreadedVer3 import RandomizedMultiThreadedVer3
from Implementations.FasterSubsetSum.RandomizedUsingDeterministic import MixOfDeterministicAndRandomized
from Implementations.FasterSubsetSum.RandomizedVariableLayers import RandomizedVariableExponentialLayers
from benchmarks.test_distributions import Distributions as dist, Distributions


def timeIt(func):
    start = time.time()
    val = func()
    end = time.time()
    clock = end - start
    return clock, val


def resetInstances():
    algorithms = [
        # PowerSeriesSubsetSum('PSSS'),
        ILPSubsetSum('ILP'),
        # ILPRandomRoundingSubsetSum('RandomRounding'),
        # DynamicProgramming('DP', benchmarkMode=True),
        # GreedySubsetSum('G'),
        # NearLinearBase(False, 'FSS, 0.0001', delta=0.0001, repetitions=8, benchmarkMode=True),
        # NearLinearBase(False, 'FSS, 0.1', delta=0.1, repetitions=8, benchmarkMode=True),
        # NearLinearBase(False, 'FSS, 1', delta=1, repetitions=8, benchmarkMode=True),
        # FastIntegersFromGit(),
        # FastIntegersPersonal('FIP', benchmarkMode=True, recursionLimit=3, dnqThreshold=3, thresholdValue=3),
        # FastIntegersFromJournal('FIJ', benchmarkMode=True, recursionLimit=3, dnqThreshold=3, thresholdValue=3),
        # FastIntegersFromJournal('FIJ, b/2', b=lambda n: int(math.sqrt(n * math.log(n))) // 2, benchmarkMode=True, recursionLimit=3, dnqThreshold=3, thresholdValue=3),
        # DivideAndConquerRunner('DivideAndConquer')
    ]
    return algorithms

class Test(TestCase):

    # ALGORITHM GRAVEYARD
    # cls.unbounded = Unbounded()

    @classmethod
    def setUp(cls):
        cls.linearLayersFSS = RandomizedLinearLayers(True, 'linear layers', 0)
        cls.fiveRepetitionFSS = NearLinearBase(True, 'normal')
        # cls.fastIntegersGit = FastIntegersFromGit()
        # cls.DP = DynamicProgramming('DP', benchmarkMode=False)
        # cls.powerSeries = PowerSeriesSubsetSum('PSSS')
        # cls.fastIntegerGit = FastIntegersFromGit()
        # cls.fastIntegerPersonal = FastIntegersPersonal('FIP', benchmarkMode=False, dnqThreshold=10, thresholdValue=5, recursionLimit=20)
        # cls.fij = FastIntegersFromJournal('FIJ', benchmarkMode=False, dnqThreshold=10, thresholdValue=5, recursionLimit=20)
        # cls.fij2 = FastIntegersFromJournal('FIJ, b/2', b=lambda n: max(int(math.sqrt(n * math.log(n))) // 2, 1), benchmarkMode=False, dnqThreshold=10, thresholdValue=5, recursionLimit=20)
        # cls.nearLinear1FSS = NearLinearBase(False, 'FSS, 0.0001', delta=0.0001, repetitions=1, benchmarkMode=False, bruteForceThreshold=5, dnQThreshold=20)
        # cls.nearLinear2FSS = NearLinearBase(False, 'FSS, 0.1', delta=0.1, repetitions=1, benchmarkMode=False, bruteForceThreshold=5, dnQThreshold=20)
        # cls.nearLinear3FSS = NearLinearBase(False, 'FSS, 1', delta=1, repetitions=1, benchmarkMode=False, bruteForceThreshold=5, dnQThreshold=20)
        # cls.fastIntegersPersonal = FastIntegersPersonal('Koiliaris and Xu', benchmarkMode=True)
        # cls.fastIntegersJournal = FastIntegersFromJournal('journal', benchmarkMode=True)
        # cls.clusterFSS = RandomizedClusters(True, 1, 0, delta=0.0001, bruteForceThreshold=5, dnQThreshold=20)
        # cls.adaptiveFSS = RandomizedAdaptiveFunction(True, 0)  # Just add some junk here
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 2)
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 1)
        # cls.OneThreadFSS = RandomizedMultiThreadedVer2(False, 1, 1)
        # cls.TwoThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 2)
        # cls.OneThreadFSS = RandomizedMultiThreadedVer2(False, 1, 1)
        # cls.FourThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 4)
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.AOneThreadFSS = RandomizedMultiThreadedVer2(False, 1, 1)
        # cls.BTwoThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 2)
        # cls.CFourThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 4)
        # cls.DEightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.ESixteenThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 16)
        # cls.SixteenThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 16)
        # cls.EightThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.SixteenThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 16)
        # cls.FourThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 4)
        # cls.SixThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 6)
        # cls.SixThreadsFSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.multiThreaded4ThreadsOriginalFSS = RandomizedMultiThreaded(False, 1, 4)
        # cls.multiThreaded4ThreadsVer2FSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.multiThreaded8ThreadsVer2FSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.multiThreaded8ThreadsVer2FSS = RandomizedMultiThreadedVer3(False, 1, 8)
        # cls.multiThreaded4ThreadsVer3FSS = RandomizedMultiThreadedVer3(False, 1, 4)
        # cls.multiThreaded2ThreadsVer2FSS = RandomizedMultiThreadedVer2(False, 1, 2)
        # cls.multiThreaded8ThreadsVer2FSS = RandomizedMultiThreadedVer2(False, 1, 8)
        # cls.multiThreaded2ThreadsVer2FSS = RandomizedMultiThreadedVer2(False, 1, 4)
        # cls.multiThreaded3ThreadsFSS = RandomizedMultiThreaded(True, 0, 3)
        # cls.multiThreaded4ThreadsFSS = RandomizedMultiThreaded(False, 1, 2)
        # cls.multiThreaded6ThreadsFSS = RandomizedMultiThreaded(True, 0, 6)
        # cls.AvariableLayers1dot5FSS = RandomizedVariableExponentialLayers(True, 1.5, 'b=1.5', 0)
        # cls.BvariableLayersPointSeventyFiveFSS = RandomizedVariableExponentialLayers(True, 1.75, 'b=1.75', 0)
        # cls.CvariableLayers2dot5FSS = RandomizedVariableExponentialLayers(True, 2.5, 'b=2.5', 0)
        # cls.DvariableLayers2dot75FSS = RandomizedVariableExponentialLayers(True, 2.75, 'b=2.75', 0)
        # cls.EvariableLayers3FSS = RandomizedVariableExponentialLayers(True, 3, 'b=3', 0)
        # cls.FvariableLayers3dot25FSS = RandomizedVariableExponentialLayers(True, 3.25, 'b=3.25', 0)
        # cls.GvariableLayers3dot5FSS = RandomizedVariableExponentialLayers(True, 3.5, 'b=3.5', 0)
        # cls.HvariableLayers4FSS = RandomizedVariableExponentialLayers(True, 4, 'b=4', 0)
        # cls.lessRepetitionsFSS = RandomizedBaseLessRepetitions(True, 'less repetitions', 0)
        # cls.lowerBoundChangeFSS = RandomizedLowerBoundDifference(True)  # Just add some junk here
        # cls.MixOfDeterministicAndRandomizedFSS = MixOfDeterministicAndRandomized(True)
        # cls.fastIntegersJournalSqrtNLogNTimesTwo = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))*2)
        # cls.fastIntegersJournalSqrtNLogNTimesFour = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))*4)
        # cls.fastIntegersJournalSqrtNLogNTimesEight = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))*8)
        # cls.fastIntegersJournalSqrtNLogNDividedTwo = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))//2)
        # cls.fastIntegersJournalSqrtNLogNDividedFour = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))//4)
        # cls.fastIntegersJournalSqrtNLogNDividedSix = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))//6)
        # cls.fastIntegersJournalSqrtNLogNDividedEight = FastIntegersFromJournal(lambda n: int(math.sqrt(n * math.log(n)))//8)

    @unittest.skip("skip")
    def test_me(self):
        delta = 0.0001
        i = 5
        a, T = dist.clusteredDistributionEven(i)
        algorithm = RandomizedLinearLayers(True)
        fast = algorithm.fasterSubsetSum(a, T, delta)
        # visualize_distribution(a, algorithm.getLayerInformation(), algorithm.getBucketInformation())

    def test_compare_all_algorithms(self):
        this = Test()
        self.title = "../plots/" + datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
        fasterSubsetSum = [attr for attr in dir(this) if not callable(getattr(this, attr)) and not attr.startswith(
            "_") and not attr == 'longMessage' and not attr == 'maxDiff' and attr.endswith('FSS')]
        fasterSubsetSum.sort()
        otherAlgorithms = [attr for attr in dir(this) if not callable(getattr(this, attr)) and not attr.startswith(
            "_") and not attr == 'longMessage' and not attr == 'maxDiff' and not attr.endswith('FSS')]
        runningTimes = defaultdict(lambda: defaultdict(list))
        layers = defaultdict(lambda: defaultdict(list))
        testInput = defaultdict(list)
        testInputRatio = defaultdict(list)
        solutionFail = defaultdict(lambda: defaultdict(list))
        distributionFunctions = Distributions().allDistributions()
        sizeOfInput = defaultdict(list)
        tInput = defaultdict(list)
        for i in range(1, 4):
            i = 2**i
            for distributionFunction in distributionFunctions:
                a, T = distributionFunction(i)
                delta = 0.00001
                testInput[distributionFunction] = a, T
                testInputRatio[distributionFunction.__name__].append((a, T))
                print("Test {}, distribution {}: List of size {}, T={}".format(i, distributionFunction.__name__, len(a),
                                                                               T))
                solutions = defaultdict()
                if distributionFunction is Distributions.clusteredDistributionRandom:
                    cluster = ComputeKMeansGrouping(a)
                    cluster.computeClusters(Distributions.noClusters(i))

                    self.clusterFSS = RandomizedClusters(False, 1, kMeansSolver=cluster, i=Distributions.noClusters(i), delta=0.0001, bruteForceThreshold=5, dnQThreshold=20)
                for algorithm in fasterSubsetSum:
                    runner = eval("self." + algorithm)
                    print('Running ', algorithm)
                    totalClock = 0
                    results = []
                    # for j in range(3):
                        # np.random.seed(i)
                    clock, results = timeIt(lambda: runner.fasterSubsetSum(a, T, delta))
                    totalClock += clock
                    totalClock
                    print("%s solved on avg in %.2f" % (algorithm, totalClock))
                    runningTimes[distributionFunction.__name__][algorithm].append(totalClock)
                    solutions[algorithm] = results
                    if runner.debug:
                        layers[algorithm][distributionFunction.__name__] = runner.getLayerInformation()

                for algorithm in otherAlgorithms:
                    runner = eval("self." + algorithm)
                    print('Running ', algorithm)
                    totalClock = 0
                    results = []
                    # for j in range(3):
                    clock, results = timeIt(lambda: runner.run(a, T))
                    totalClock += clock
                    # totalClock /= 3
                    print("%s solved in %.2f" % (algorithm, totalClock))
                    runningTimes[distributionFunction.__name__][algorithm].append(clock)
                    solutions[algorithm] = results

                for deterministic in otherAlgorithms:
                    for randomized in solutions.keys():
                        try:
                            solutions[deterministic].sort()
                            solutions[randomized].sort()
                            self.assertListEqual(solutions[deterministic], solutions[randomized])
                        except AssertionError as e:
                            # if solutions[deterministic] == [T] or solutions[randomized] == [T]:
                            #     continue
                            print("ASSERTION ERROR: {} and {} did not return same result".format(deterministic,
                                                                                                 randomized))
                            print("Algorithm missed {} values".format(
                                abs(len(solutions[deterministic]) - len(solutions[randomized]))))
                            solutionFail[randomized][distributionFunction.__name__].append(
                                abs(len(solutions[deterministic]) - len(solutions[randomized])))
                            print(e)
                            # for value in solutions[randomized]:
                            #     if value not in solutions[deterministic]:
                            #         print('Randomized generated value not in determinstic!', value)
                            # exit(1)
                        else:
                            solutionFail[randomized][distributionFunction.__name__].append(0)
                sizeOfInput[distributionFunction].append(len(a))
                tInput[distributionFunction].append(T)
        mkdir_p(self.title + "/layerings")
        mkdir_p(self.title + "/runningTimes")
        mkdir_p(self.title + "/errorPlots")
        self.plotRunningTimes(self.title, distributionFunctions, runningTimes, sizeOfInput, tInput)
        plotDistributions(self.title, distributionFunctions, fasterSubsetSum, layers, testInput)
        # printRatios(self.title, runningTimes)

    def test_evaluateQualityOfSolutions(self):
        algorithms = resetInstances()
        DistributionGenerator = Distributions()
        runningTimeAndSolution = defaultdict(lambda: defaultdict(list))
        # print('Running P3')
        runner = FastIntegersPersonal('FIP', benchmarkMode=True, recursionLimit=3, dnqThreshold=3, thresholdValue=3)
        for i in range(1, 26):
            # Generate P3 problems
            a, t = DistributionGenerator.P(3, i * 4, 1)
            a2, t2 = DistributionGenerator.P(3, i * 4, 2)
            a3, t3 = DistributionGenerator.P(3, i * 4, 3)
            instances = [(a, t), (a2, t2), (a3, t3)]
            optimalSolution = t == max(runner.run(a, t))
            optimalSolution2 = t2 == max(runner.run(a2, t2))
            optimalSolution3 = t3 == max(runner.run(a3, t3))
            runningTimeAndSolution['possible'][i] = (optimalSolution, optimalSolution2, optimalSolution3)
            # runningTimeAndSolution['possible'][i] = (True, True, True)
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)
        algorithms = resetInstances()
        # print('Running P6')
        for i in range(26, 51):
            # Generate P4 problems
            a, t = DistributionGenerator.P(5, (i - 25) * 4, 1)
            a2, t2 = DistributionGenerator.P(5, (i - 25) * 4, 2)
            a3, t3 = DistributionGenerator.P(5, (i - 25) * 4, 3)
            # optimalSolution = max(FastIntegersFromGit().run(a, t))
            instances = [(a, t), (a2, t2), (a3, t3)]
            optimalSolution = t == max(runner.run(a, t))
            optimalSolution2 = t2 == max(runner.run(a2, t2))
            optimalSolution3 = t3 == max(runner.run(a3, t3))
            runningTimeAndSolution['possible'][i] = (optimalSolution, optimalSolution2, optimalSolution3)
            # runningTimeAndSolution['possible'][i] = (True, True, True)
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)
        # print('Running P12')
        algorithms = resetInstances()

        for i in range(51, 76):
            # Generate P12 problems
            a, t = DistributionGenerator.P(6, (i - 50) * 4, 1)
            a2, t2 = DistributionGenerator.P(6, (i - 50) * 4, 2)
            a3, t3 = DistributionGenerator.P(6, (i - 50) * 4, 3)
            instances = [(a, t), (a2, t2), (a3, t3)]
            # if len(algorithms) > 1:
            optimalSolution = t == max(runner.run(a, t))
            optimalSolution2 = t2 == max(runner.run(a2, t2))
            optimalSolution3 = t3 == max(runner.run(a3, t3))
            runningTimeAndSolution['possible'][i] = (optimalSolution, optimalSolution2, optimalSolution3)
            # else:
            # runningTimeAndSolution['possible'][i] = (True, True, True)
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)
        algorithms = resetInstances()
        print('Running EVEN/ODD')
        for i in range(76, 101):
            # Generate EVEN/ODD problems
            a, t = DistributionGenerator.EVEN_ODD(3, (i - 75) * 4)
            a2, t2 = DistributionGenerator.EVEN_ODD(3, (i - 75) * 4)
            a3, t3 = DistributionGenerator.EVEN_ODD(3, (i - 75) * 4)
            instances = [(a, t), (a2, t2), (a3, t3)]
            runningTimeAndSolution['possible'][i] = (False, False, False)
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)
        algorithms = resetInstances()
        print('Running TODD')
        for i in range(101, 126):
            # Generate TODD problems
            a, t = DistributionGenerator.TODD(i - 97)
            runningTimeAndSolution['possible'][i] = tuple([False])
            instances = [(a, t)]
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)
        algorithms = resetInstances()
        print('Running AVIS')
        for i in range(126, 151):
            # Generate AVIS problems.
            a, t = DistributionGenerator.AVIS((i - 125) * 4)
            instances = [(a, t)]
            runningTimeAndSolution['possible'][i] = tuple([False])
            self.evaluateAlgorithms(instances, algorithms, i, runningTimeAndSolution)

        algorithms = resetInstances()
        # for a in instance:

        # for algorithm in algorithms:
        #     for i in range(min(runningTimeAndSolution['possible'].keys()), max(runningTimeAndSolution['possible'].keys()) + 1):
        #         if i in runningTimeAndSolution[algorithm.label]:
        #             for choice in range(len(runningTimeAndSolution[algorithm.label][i][1])):
        #                     agree = runningTimeAndSolution[algorithm.label][i][1][choice] == runningTimeAndSolution['possible'][i][choice]
        #                     if not agree:
        #                         dict[algorithm.label] += 1
        names = ['P3', 'P5', 'P6', 'EVENODD', 'TODD', 'AVIS']
        for a in range(0, 6):
            instanceIndex = (a*25 + 1, ((a+1)*25))
            fig, ax = plt.subplots()
            dict = defaultdict(int)
            outOf = defaultdict(int)
            for algorithm in algorithms:
                instanceAndTime = list()
                for index in range(instanceIndex[0], instanceIndex[1]+1):
                    if index in runningTimeAndSolution[algorithm.label]:
                        instanceAndTime.append((index, runningTimeAndSolution[algorithm.label][index][0])) #Use 1 to access decision
                        for choice in range(len(runningTimeAndSolution[algorithm.label][index][1])):
                                agree = runningTimeAndSolution[algorithm.label][index][1][choice] == runningTimeAndSolution['possible'][index][choice]
                                outOf[algorithm.label] += 1
                                if not agree:
                                    dict[algorithm.label] += 1
                    else:
                        break
                print(algorithm.label, 'got', dict[algorithm.label], 'incorrect decisions out of',
                      outOf[algorithm.label])
                p = ax.plot(list(map(itemgetter(0), instanceAndTime)), list(map(itemgetter(1), instanceAndTime)),
                             '^')
                colorUsed = p[-1].get_color()
                ax.plot(list(map(itemgetter(0), instanceAndTime)), list(map(itemgetter(1), instanceAndTime)),
                         label=algorithm.label, c=colorUsed)
            ax.set_ylabel('Time (seconds)')
            ax.set_xlabel('Instance index')
            ax.grid()
            ax.set_xlim(instanceIndex[0], instanceIndex[1])
            ax.legend(prop={'size': 13})
            fig.tight_layout()
            fig.savefig(names[a], dpi=300)
            fig.show()
        # for algorithm in algorithms:
        #     instanceAndTime = list(zip(runningTimeAndSolution[algorithm.label].keys(),
        #                                map(lambda x: x[0], runningTimeAndSolution[algorithm.label].values())))
        #     p = plt.plot(list(map(itemgetter(0), instanceAndTime)), list(map(itemgetter(1), instanceAndTime)), '^')
        #     colorUsed = p[-1].get_color()
        #     plt.plot(list(map(itemgetter(0), instanceAndTime)), list(map(itemgetter(1), instanceAndTime)),
        #              label=algorithm.label, c=colorUsed)
        # plt.ylabel('Time (seconds)')
        # plt.xlabel('Instance index')
        # plt.grid()
        # plt.xlim([min(runningTimeAndSolution['possible'].keys()), max(runningTimeAndSolution['possible'].keys())])
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.legend(prop={'size': 13})
        # plt.tight_layout()
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()
        # fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(7, 5)
        # fig.savefig('AVIS.png', dpi=100)
        # fig.show()

    def evaluateAlgorithms(self, instances, algorithms, i, runningTimeAndSolution):
        toRemoveAlgorithm = list()
        for algorithm in algorithms:
            solution = -1
            print('Running', algorithm.label)
            clock = 0
            for input in instances:
                a, t = input
                if isinstance(algorithm, NearLinearBase):
                    runningTime, results = timeIt(lambda: algorithm.fasterSubsetSum(a, t))
                    solution = max(results)
                if isinstance(algorithm, PowerSeriesSubsetSum):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, DivideAndConquerRunner):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, FastIntegersFromGit):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, FastIntegersPersonal):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, FastIntegersFromJournal):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, DynamicProgramming):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, DynamicProgrammingRegular):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = max(results)
                if isinstance(algorithm, GreedySubsetSum):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = sum(results)
                if isinstance(algorithm, ILPSubsetSum):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = sum(results)
                if isinstance(algorithm, ILPRandomRoundingSubsetSum):
                    runningTime, results = timeIt(lambda: algorithm.run(a, t))
                    solution = sum(results)
                if solution == -1:
                    print('SOMETHING WENT WRONG')
                    exit(-1)
                clock += runningTime
                runningTimeAndSolution[algorithm.label][i].append(solution == t)
            # Update clock to reflect average running time
            clock /= len(instances)
            # Add the average running time, and if found solution
            runningTimeAndSolution[algorithm.label][i] = (clock, runningTimeAndSolution[algorithm.label][i])
            print(clock)
            if clock > 20:
                toRemoveAlgorithm.append(algorithm)
        for algorithm in toRemoveAlgorithm:
            print('Removing ', algorithm.label)
            algorithms.remove(algorithm)

    def test_barplot_even(self):
        def a(T, n):
            vals = np.random.randint(0, T, n, dtype=np.int64)
            vals = list(set(vals))
            return vals

        self.barplot_errors_fss(a, 'Even Distribution')

    def test_barplot_low(self):
        def a(T, n):
            vals = np.random.randint(0, T // 3, n, dtype=np.int64)
            vals = list(set(vals))
            return vals

        self.barplot_errors_fss(a, 'Low Distribution')

    def test_barplot_normal(self):
        def a(T, n):
            vals = np.random.normal(T // 2, T / 6, n)
            vals = map(lambda x: int(abs(x)), vals)
            vals = list(set(vals))
            return vals

        self.barplot_errors_fss(a, 'Normal Distribution')

    def barplot_errors_fss(self, funcA, title):
        # set width of bars
        matplotlib.rcParams['text.usetex'] = True
        fig = plt.figure(figsize=(6, 8))
        fig.suptitle("Error rate using different repetitions, {}".format(title))
        outer = gridspec.GridSpec(1, 1)
        ts = ['2 ** 13', '2 ** 16', '2 ** 19']
        stringTs = ['2^{13}', '2^{16}', '2^{19}']
        ns = [25, 50, 75, 100, 150, 300]
        repetitions = 8
        for i in range(len(ts)):
            bars = defaultdict(lambda: list())
            for n in ns:
                np.random.seed(213353)
                T = eval(ts[i])
                a = funcA(T, n)
                for r in range(1, repetitions + 1):
                    alg = NearLinearBase(False, repetitions=r)
                    deterministicAlg = FastIntegersFromGit()
                    deterministic = deterministicAlg.run(a, T)
                    errorRate = 0
                    print("running for T = {}, n = {}".format(T, n))
                    for j in range(0, 10):
                        randomized = alg.fasterSubsetSum(a, T, 0.00001)
                        errorRate += len(deterministic) - len(randomized)
                    errorRate /= 10
                    errorRate = math.ceil(errorRate)
                    bars[n].append(errorRate)

            inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                                                     subplot_spec=outer[0], wspace=0.1, hspace=0.1)
            figure = plt.Subplot(fig, inner[i])
            barWidth = 0.1
            x = np.arange(len(ns))
            for r in range(0, repetitions):
                bar = list(map(itemgetter(r), bars.values()))
                bar = list(map(lambda x: int(x), bar))
                if max(bar) == 0:
                    continue
                # Make the plot
                figure.set_yscale('log')
                figure.bar(list(map(str, ns)), bar, label='r = {}'.format(r + 1))
            # Add xticks on the middle of the group bars
            # figure.set_title(r'T = ${}$'.format(stringTs[i]), fontweight='bold')
            # figure.set_xticks([r + barWidth * 3.3 for r in range(len(ns))])
            # figure.set_xticklabels(list(map(lambda x: 'n={}'.format(x), ns)))
            # import matplotlib.patches as mpatches
            # red_patch = mpatches.Patch(color='w', label=r'T = ${}$'.format(stringTs[i]))
            # plt.legend(handles=[red_patch])
            # figure.text(0.9, 0.1, 'T = {}'.format(ts[i]), ha='center', va='center', transform=figure.transAxes)
            # figure.set_xticks(x, list(map(str, ns)))
            figure.legend(title=r'T = ${}$'.format(stringTs[i]), bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.add_subplot(figure)  # Create legend & Show graphic
        fig.show()

    def plotRunningTimes(self, title, distributionFunctions, runningTimes, sizeOfInput, tInput):
        for i in range(len(distributionFunctions)):
            fig, ax = plt.subplots()
            distributionFunction = distributionFunctions[i]
            # color_cycle = ax._get_lines.prop_cycler
            # next(color_cycle)['color']
            ax.set_title(distributionFunction.__name__, fontsize=12)
            for algorithm in runningTimes[distributionFunction.__name__].keys():
                sizeWithTime = list(
                    zip(sizeOfInput[distributionFunction], runningTimes[distributionFunction.__name__][algorithm]))
                sizeWithTime.sort(key=operator.itemgetter(0))
                algorithm = eval("self." + algorithm)
                ax.plot(list(map(itemgetter(0), sizeWithTime)),
                        list(map(itemgetter(1), sizeWithTime)), label=algorithm.label)
                ax.set_ylabel('Time (seconds)')
                ax.set_xlabel('Size of input')
                ax.set_yscale('log')
                ax.set_xscale('log')
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

    def test_compare_failure_according_to_n(self):
        this = Test()
        fasterSubsetSum = [attr for attr in dir(this) if not callable(getattr(this, attr)) and not attr.startswith(
            "_") and not attr == 'longMessage' and not attr == 'maxDiff' and attr.endswith('FSS')]
        otherAlgorithms = [attr for attr in dir(this) if not callable(getattr(this, attr)) and not attr.startswith(
            "_") and not attr == 'longMessage' and not attr == 'maxDiff' and not attr.endswith('FSS')]
        runningTimes = defaultdict(lambda: defaultdict(list))
        layers = defaultdict(lambda: defaultdict(list))
        testInput = defaultdict(list)
        solutionFail = defaultdict(lambda: defaultdict(list))
        distributionFunctions = [Distributions.evenDistributionGivenT]
        sizeOfInput = defaultdict(list)
        tInput = defaultdict(list)
        delta = 0.0001
        for T in [2 ** 20]:
            self.title = "../plots/" + datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
            for i in range(10, 300):
                for distributionFunction in distributionFunctions:
                    a, T = distributionFunction(i, T)
                    testInput[distributionFunction] = a, T
                    print("Test {}, distribution {}: List of size {}, T={}".format(i, distributionFunction.__name__,
                                                                                   len(a), T))
                    solutions = defaultdict()
                    for algorithm in fasterSubsetSum:
                        runner = eval("self." + algorithm)
                        totalClock = 0
                        results = []
                        # for j in range(3):
                        clock, results = timeIt(lambda: runner.fasterSubsetSum(a, T, delta))
                        totalClock += clock
                        # totalClock /= 3
                        print("%s solved on avg in %.2f" % (algorithm, totalClock))
                        runningTimes[distributionFunction.__name__][algorithm].append(totalClock)
                        solutions[algorithm] = results
                        if runner.debug:
                            layers[algorithm][distributionFunction.__name__] = runner.getLayerInformation()

                    for algorithm in otherAlgorithms:
                        runner = eval("self." + algorithm)
                        totalClock = 0
                        results = []
                        for i in range(3):
                            clock, results = timeIt(lambda: runner.run(a, T))
                            totalClock += clock
                        totalClock /= 3
                        print("%s solved in %.2f" % (algorithm, totalClock))
                        runningTimes[distributionFunction.__name__][algorithm].append(clock)
                        solutions[algorithm] = results

                    for deterministic in otherAlgorithms:
                        for randomized in fasterSubsetSum:
                            try:
                                self.assertListEqual(solutions[deterministic], solutions[randomized])
                            except AssertionError as e:
                                print("ASSERTION ERROR: {} and {} did not return same result".format(deterministic,
                                                                                                     randomized))
                                print("Algorithm missed {} values".format(
                                    abs(len(solutions[deterministic]) - len(solutions[randomized]))))
                                solutionFail[randomized][distributionFunction.__name__].append(
                                    abs(len(solutions[deterministic]) - len(solutions[randomized])))
                                # print(e)
                                # exit(1)
                            else:
                                solutionFail[randomized][distributionFunction.__name__].append(0)
                    sizeOfInput[distributionFunction].append(len(a))
                    tInput[distributionFunction].append(T)
            # mkdir_p(self.title + "/layerings")
            # mkdir_p(self.title + "/runningTimes")
            mkdir_p(self.title + "/errorPlots")
            # self.plotRunningTimes(distributionFunctions, runningTimes, sizeOfInput, tInput)
            # self.plotDistributions(distributionFunctions, fasterSubsetSum, layers, testInput)
            self.plotErrorRates(distributionFunctions, solutionFail, sizeOfInput, T)

    def test_findOptimalBParameter(self):
        distributionFunctions = Distributions().allDistributions()
        divisors = list()
        times = list()
        i = 40  # Fix I for suitable instance to measure
        prevClockTime = sys.maxsize
        a, T = Distributions.clusteredDistributionLow(i)
        n = len(a)
        divisor = 2
        b = int(math.sqrt(n * math.log(n)))
        algorithm = FastIntegersFromJournal(lambda n: b)
        clock, results = timeIt(lambda: algorithm.run(a, T))
        for i in np.arange(1, 4, 0.1):
            divisors.append(i)
            times.append(clock)
            print(clock)
            prevClockTime = clock
            b = int(int(math.sqrt(n * math.log(n))) / i)
            divisor *= 2
            algorithm = FastIntegersFromJournal(lambda n: b)
            clock, results = timeIt(lambda: algorithm.run(a, T))
        plt.xlabel("Divisor")
        plt.ylabel("Time")
        plt.plot(divisors, times)
        plt.show()


def plotDistributions(title, distributionFunctions, fasterSubsetSum, layers, testInput):
    for algorithm in fasterSubsetSum:
        Tot = len(distributionFunctions)
        Cols = max(Tot // 2, 1)
        Rows = Tot // Cols
        Rows += Tot % Cols
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(Rows, Cols)
        fig.suptitle(algorithm)
        for i in range(len(distributionFunctions)):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                     subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            layerInformation = layers[algorithm][distributionFunctions[i].__name__]
            distributionIndex = 0
            layerIndex = 1
            distribution = plt.Subplot(fig, inner[distributionIndex])
            a, t = testInput[distributionFunctions[i]]
            distribution.set_title(distributionFunctions[i].__name__ + ', T = {}'.format(t))
            # maxLayerHeight = max(list(map(itemgetter(0), layerInformation)))
            yvalues = np.zeros_like(a)
            yvalues = list(map(lambda x: x + random(), yvalues))
            bins = len(a)
            hist = np.histogram(a, bins=bins)

            baseArea = (len(a) / (t // 100)) // 2
            area = list(map(lambda x: x[0] * baseArea, hist))
            hist = list(filter(lambda x: x[0] != 0, list(zip(hist[0], hist[1]))))
            maxElement = max(list(map(lambda x: x[0], hist)))
            minElement = min(list(map(lambda x: x[0], hist)))
            colors = list(map(lambda x: 1 - (x[0] - minElement) / (maxElement - minElement), hist))

            # for i in range(len(hist)):
            #     color = str(colors[i])
            #     distribution.plot([hist[i][1], hist[i][1]], [0, hist[i][0]], color=color)
            # distribution.hist(a, bins=t//100, color=colors)
            # distribution.histogram(list(map(itemgetter(1), hist)), list(map(itemgetter(0), hist)), s=area)
            # distribution.plot(list(map(itemgetter(1), hist)), list(map(itemgetter(0), hist)))
            distribution.hist(a, bins=110)
            distribution.label_outer()
            distribution.set_xlim([-t // 50, t + t // 50])  # Make a bit room here
            # distribution.set_ylim([0, max(list(map(lambda x: x // baseArea, area))) * 1.3]) #Make a bit room here
            layerDistribution = plt.Subplot(fig, inner[layerIndex])
            layerDistribution.set_xlim([-t // 50, t + t // 50])
            layerDistribution.vlines([x[1] for x in layerInformation], 0, [max(x[0], 2) for x in layerInformation],
                                     color='r')
            fig.add_subplot(distribution)
            fig.add_subplot(layerDistribution)
        fig.savefig(title + "/layerings/" + algorithm)
        fig.show()


def plotErrorRates(self, distributionFunctions, failures, sizeOfInput, T):
    for i in range(len(distributionFunctions)):
        fig, ax = plt.subplots()
        distributionFunction = distributionFunctions[i]
        ax.set_title(distributionFunction.__name__ + ", T = {}".format((T)), fontsize=12)
        for algorithm in failures.keys():
            size = sizeOfInput[distributionFunction]
            ax.bar(size,
                   failures[algorithm][distributionFunction.__name__], label=algorithm)
            ax.set_ylabel('# solutions missed')
            ax.set_xlabel('Size of input')
        ax.grid()
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.title + "/errorPlots/" + distributionFunction.__name__)
        fig.show()


def printRatios(runningTimes, testInput):
    for distribution in runningTimes.keys():
        print('Distribution:', distribution, )
        for algorithm in runningTimes[distribution].keys():
            print('Algorithm: ', algorithm, ':')
            inp = testInput[distribution]
            for i in range(0, len(runningTimes[distribution][algorithm])):
                if i == 0:
                    print('Size: ', len(inp[i][0]), '. T =', inp[i][1], '. Elapsed time:',
                          runningTimes[distribution][algorithm][i])
                else:
                    ratio = runningTimes[distribution][algorithm][i] / runningTimes[distribution][algorithm][i - 1]
                    print('Size: ', len(inp[i][0]), '. T = ', inp[i][1], '. Elapsed time:',
                          runningTimes[distribution][algorithm][i], ' ratio: ', ratio)


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def visualize_distribution(a, layerInformation, bucketInformation):
    # Set up the plot
    # ax = plt.subplot(2, 2, i + 1)
    # a = np.random.random_integers(0, 100000, 100)

    # Draw the plot
    # An "interface" to matplotlib.axes.Axes.hist() method
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[1].hist(x=a, bins=int(len(a) / 3), color='#0504aa',
               alpha=0.7, rwidth=0.85)
    ax[1].set_title('Buckets (yellow) and intervals (red)')
    ax[0].vlines([x[1] for x in layerInformation], 0, [max(x[0], 10) for x in layerInformation], color='r')
    ax[0].set_title('Distribution of numbers')
    bucketValuesX = list()
    bucketValuesY = list()
    for x in bucketInformation:
        for y in x:
            bucketValuesX.append(y[2] - y[1] - 2)
            bucketValuesY.append(y[0])
    ax[0].vlines(bucketValuesX, 0, bucketValuesY, color='orange')
    fig.show()


def visualize_distribution_normal(a, layerInformation, bucketInformation):
    # Set up the plot
    # ax = plt.subplot(2, 2, i + 1)
    # a = np.random.random_integers(0, 100000, 100)

    # Draw the plot
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[1].hist(x=a, bins=int(len(a) / 3), color='#0504aa',
               alpha=0.7, rwidth=0.85)
    ax[1].set_title('Buckets (yellow) and intervals (red)')
    ax[0].vlines([x[1] for x in layerInformation if x[0] > 0], 0, [x[0] for x in layerInformation if x[0] > 0],
                 color='r')
    ax[0].set_title('Distribution of numbers')
    bucketValuesX = list()
    bucketValuesY = list()
    for x in bucketInformation:
        for y in x:
            bucketValuesX.append(y[2] - y[1] - 2)
            bucketValuesY.append(y[0])
    ax[0].vlines(bucketValuesX, 0, bucketValuesY, color='orange')
    plt.show()
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
