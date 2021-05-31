import math

import numpy as np

from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedBaseLessRepetitions(NearLinearBase):

    def color_coding(self, Z, t, k, delta):
        if len(Z) == 1:
            return [1]
        if self.repetitions == 0:
            # if math.log(t, 1.05) >= self.n:
            #     repetitions = 5
            # else:
            # repetitions = 1
            repetitions = math.log(1.0 / delta, 17.0 / 7.0)
        else:
            repetitions = self.repetitions
        S = [[] for _ in range(math.ceil(repetitions))]
        for j in range(0, math.ceil(repetitions)):
            partition = self.partitionSetIntoK(Z, k * k * k)  # max(int(k*k//2), 2))
            sumset = partition[0]
            for i in range(1, len(partition)):
                sumset = self.sumSet(sumset, partition[i], t)
            S[j] = sumset
        union = np.array(S[0])
        for j in range(1, len(S)):
            if len(S[j]) > len(union):
                S[j][np.nonzero(union)[0]] = 1
                union = S[j]
            else:
                union[np.nonzero(S[j])[0]] = 1
        return list(union)
