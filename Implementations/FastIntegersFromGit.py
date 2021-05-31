import math

from numpy import transpose, nonzero, select, zeros
from scipy.signal import fftconvolve
from operator import itemgetter
from itertools import takewhile, product

# to tests, try the following
# > import random
# > xs = list(set(random.sample(xrange(10000), 500)))
# > subset_sums_up_to_u(xs,50000) == naive_subset_sums_up_to_u(xs,50000)
from Implementations.DPRegularWithCount import DynamicProgrammingWithCount
from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm


def naive_subset_sums_up_to_u(xs, u):
    sums = {0}
    for x in xs:
        sums2 = set(sums)
        for y in sums:
            if x + y < u:
                sums2.add(x + y)
        sums = sums2
    return sums


class FastIntegersFromGit(IDeterministicAlgorithm):


    def __init__(self):
        self.label = 'FIG'
    @classmethod
    def run(cls, values, target):
        if len(values) == 1:
            return values
        return cls.deterministic_integers(values, target)

    # output is list of subset sums
    @classmethod
    def deterministic_integers(cls, xs, u):
        eps = 0.0001  # account for floating error

        # list to matrix representation
        def L2M(S):
            c1 = list(map(itemgetter(0), S))
            c2 = list(map(itemgetter(1), S))
            M = zeros((max(c1) + 1, max(c2) + 1))
            M[c1, c2] = 1
            return M

        # matrix to list representation
        def M2L(M):
            return zip(*nonzero(select([M > eps], [1])))

        def minkowski(a, b):
            (A, kA, aA, bA) = a
            (B, kB, aB, bB) = b
            if kB == 0:
                return (A, kA, aA, bA)
            # info of the output
            occurences = kA + kB
            lowerBound = min(aA, aB)
            upperBound = max(bA, bB)
            return minkowski2(A, B, occurences, lowerBound, upperBound)

        # taking minkowski sum of two sets A and B, such that
        # it is known that A, B, A+B are contained in the union of intervals
        # [i*a, i*b] for all 0<=i<=k
        def minkowski2(A, B, occurences, a, b):
            mins = int(min(1 + u // a, occurences))
            multiply = int(b-a)
            if a <= mins * multiply:
                h = lambda x: (0, x)

                def invh1(a):
                    (_, x) = a
                    return x

                invh = invh1
            else:
                h = lambda x: (x // a, x % a)

                def invh2(v):
                    (i, j) = v
                    return i * a + j

                invh = invh2
            ARemoveOffset = list(map(h, A))
            BRemoveOffset = list(map(h, B))
            MA, MB = L2M(ARemoveOffset), L2M(BRemoveOffset)
            invMC = fftconvolve(MA, MB)
            C = list(takewhile(lambda x: x <= u, list(map(invh, M2L(invMC)))))
            return (C, occurences, a, b)

        smallRecursionSolver = DynamicProgrammingWithCount()
        # combine a list, where each element of the list is
        # (set of subset sums, number of generators, lower bound of generators, upper bound of generators)
        def combine(xs):
            if len(xs) == 1:
                return xs[0]
            if len(xs) % 2 != 0:
                xs.append(([], 0, 0, 0))
            # if len(xs[0][0]) <= 2: #Basic case
            #     return combine([([0, X[0][0][1], X[1][0][1], X[0][0][1] + X[1][0][1]],
            #                      X[0][1] + X[1][1], min(X[0][2], X[1][2]), max(X[0][3], X[1][3])) for X in zip(xs[0::2], xs[1::2])])
            #


                # start = xs[0]
                # for x in xs[1:]:
                #     (A, kA, aA, bA) = start
                #     (B, kB, aB, bB) = x
                #     if kB == 0:
                #         start = (A, kA, aA, bA)
                #         continue
                #     # info of the output
                #     occurences = kA + kB
                #     lowerBound = min(aA, aB)
                #     upperBound = max(bA, bB)
                #     sums = smallRecursionSolver.run(B, u)
                #     start = ([value[0] for value in sums], occurences, lowerBound, upperBound)
                # return start

            # if xs[0][-1] == u:
            #     return xs
            return combine([minkowski(*X) for X in zip(xs[0::2], xs[1::2])])

        ys = [([0, x], 1, x, x) for x in sorted(xs) if x <= u and x > 0]

        return list(combine(ys)[0])
