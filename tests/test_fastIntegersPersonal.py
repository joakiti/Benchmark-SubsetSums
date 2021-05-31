from unittest import TestCase
import numpy as np

from Implementations.FastIntegersPersonal import FastIntegersPersonal
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.fastIntegers = FastIntegersPersonal('does not matter')

    def test_multidimensionalFFT(self):
        # Remember to accord for the fact that if the minimal element is 5,
        # Then we can subtract it from the sum the amount of times, without getting negative.
        A, B = [(15, 2), (16, 3)], [(38, 2), (27, 5), (100, 10)]

        expected = [(15, 2), (16, 3), (38, 2), (27, 5), (43, 8), (42, 7), (54, 5), (53, 4)]
        res = self.fastIntegers.lemma2_11(A, B, 10, 70)

        self.assertCountEqual(res, expected)

    def test_cardinality_bounded_sum_set(self):
        S = np.array([1, 2, 7, 8])
        res = self.fastIntegers.lemma2_12(S, 9, 10, 20)
        print(res)

    def test_partitioning_numbers(self):
        S = [2, 5, 8, 11, 14, 39, 12, 25, 53, 83, 125, 201]
        self.fastIntegers.lemma2_14(S, 1, 202)

    def test_fast_integer_returns_correct_sumset(self):
        vals = [1, 15, 11, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        expected = [0, 1, 3, 4, 8, 9, 11]
        T = 11
        sums = self.fastIntegers.lemma2_16(vals, T)
        self.assertListEqual(sums, expected)


    def test_benchmark(self):
        vals, t = Distributions.clusteredDistributionEven(9)
        self.fastIntegers.run(vals, t)
