import math

import numpy as np

from Implementations.helpers.Helper import toNumbers


def T(i):
    return N(i) * 5


def N(i):
    return 100 * i


def flatmap(a):
    return [value for ls in a for value in ls]


class Distributions:

    def allDistributions(self):
        return [  # self.perfectlyEvenlyDistributed,
            self.evenDistribution,
            # self.staticT_increasingN,
            self.clusteredDistributionEven,
            self.normalDistribution,
            # self.someHighManyLow,
            # self.clusteredDistributionHigh,
            # self.clusteredDistributionRandom,
            # self.clusteredDistributionLow,
            self.LowEndDistribution,
            # self.someLowManyHigh,
            # self.HighEndDistribution,
            # self.TODD
            # lambda i: self.P(6, i, 1)
        ]

    def EVEN_ODD(self, E, n):
        np.random.seed(n)
        t = n*10**E/4 + 1
        vals = set()
        while len(vals) < n:
            value = np.random.randint(1, 10**E)
            if value & 1 or value in vals:
                continue
            vals.add(value)
        return list(vals), int(t)


    def EVEN_ODD_fixed(self, E, n, seed):
        np.random.seed(seed)
        t = 100001
        vals = set()
        while len(vals) < n:
            value = np.random.randint(1, t)
            if value & 1 or value in vals:
                continue
            vals.add(value)
        return list(vals), int(t)

    def P(self, E, n, seed):
        np.random.seed(seed)
        t = n*10**E/4
        vals = set()
        while len(vals) < n:
            value = np.random.randint(1, 10**E)
            if value in vals:
                continue
            vals.add(value)
        return list(vals), int(t)

    def AVIS(self, n):
        a = [int(n*(n+1) + j) for j in range(1, n + 1)]
        t = math.floor((n-1) / 2 * n*(n+1) + math.comb(n, 2))
        return a, int(t)

    def TODD(self, n):
        k = math.log2(n)
        a = [int(2**(k+n+1) + 2**(k+j) + 1) for j in range (1, n + 1)]
        t = (n+1)*2**(k+n)-2**k+math.floor(n/2)
        return a, int(t)

    @staticmethod
    def single_solution_to_all(n):
        """
        Realized that this is an incredibly stupid way to generate the series 2^i
        :param n:
        :return:
        """
        solution = list()
        solution.append(1)
        lastIndexOf1 = 1
        while (len(solution) < n):
            nextIndex = lastIndexOf1 + 1
            solution.append(nextIndex)
            lastIndexOf1 = nextIndex + lastIndexOf1
        return solution, max(solution) + 1


    @staticmethod
    def staticT_increasingN(i):
        np.random.seed(i)
        t = 200000
        n = i
        a = np.random.randint(1, t, n)
        return list(set(a)), t

    @staticmethod
    def perfectlyEvenlyDistributed(i):
        np.random.seed(i)
        t = T(i)
        n = N(i)
        a = [np.random.randint(t // 2 ** (i + 1), t // 2 ** i, n // 2 ** i) for i in
             range(math.ceil(math.log2(n)))]
        a = flatmap(a)
        return list(set(a)), t

    @staticmethod
    def normalDistribution(i):
        np.random.seed(i)
        t = T(i)
        n = N(i)
        a = np.random.normal(t / 2, t / 6, n)
        a = list(map(lambda x: abs(int(x)), a))
        return list(set(a)), t

    @staticmethod
    def someLowManyHigh(i):
        np.random.seed(i)
        t = T(i)
        n = N(i)
        a = [np.random.randint(0, t // 2, n // 10), np.random.randint(t // 2, t, n - n // 10)]
        a = flatmap(a)
        return list(set(a)), t

    @staticmethod
    def someHighManyLow(i):
        np.random.seed(i)
        t = T(i)
        n = N(i)
        a = [np.random.randint(t // 2, t, n // 10), np.random.randint(0, t // 2, n - n // 10)]
        a = flatmap(a)
        return list(set(a)), t

    @staticmethod
    def evenDistribution(i):
        np.random.seed(i)
        t = T(i)
        n = N(i)
        a = np.random.randint(0, t, n, dtype=np.int64)
        return list(set(a)), t

    @staticmethod
    def evenDistributionGivenT(i, t):
        np.random.seed(i)
        n = N(i)
        a = np.random.randint(0, t, n, dtype=np.int64)
        return list(set(a)), t

    @staticmethod
    def clusteredDistributionEven(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        clusterCount = int(math.log(n))
        size = int(n // clusterCount) // 2
        incrementer = int(t // clusterCount)
        clusters = [incrementer * i for i in range(1, clusterCount)]
        a = [range(max(cluster - size, 0), min(cluster + size, t)) for cluster in clusters]
        a = list(set(flatmap(a)))
        return list(set(a)), t

    @staticmethod
    def noClusters(i):
        n = N(i)
        return int(math.log(n))

    @staticmethod
    def clusteredDistributionRandom(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        clusterCount = int(math.log(n))
        size = int(n // clusterCount) // 2
        clusters = np.random.randint(0, t, clusterCount)
        a = [range(max(cluster - size, 0), min(cluster + size, t)) for cluster in clusters]
        a = list(set(flatmap(a)))
        return list(set(a)), t

    @staticmethod
    def singleSolution(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)

    @staticmethod
    def clusteredDistributionLow(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        clusterCount = int(math.log(n))
        size = int(n // clusterCount) // 2
        incrementer = int((t // 2) // clusterCount)
        clusters = [incrementer * i for i in range(1, clusterCount)]
        a = [range(max(cluster - size, 0), min(cluster + size, t)) for cluster in clusters]
        a = list(set(flatmap(a)))
        return list(set(a)), t

    @staticmethod
    def clusteredDistributionHigh(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        clusterCount = int(math.log(n))
        size = int(n // clusterCount) // 2
        incrementer = int((t // 2) // clusterCount)
        clusters = [t // 2 + incrementer * i for i in range(1, clusterCount)]
        a = [range(max(cluster - size, 0), min(cluster + size, t)) for cluster in clusters]
        a = list(set(flatmap(a)))
        return list(set(a)), t

    @staticmethod
    def LowEndDistribution(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        a = np.random.randint(0, t // 3, n)
        return list(set(a)), t

    @staticmethod
    def HighEndDistribution(i):
        np.random.seed(i)
        n = N(i)
        t = T(i)
        a = np.random.randint(t - t // 3, t, n)
        return list(set(a)), t
