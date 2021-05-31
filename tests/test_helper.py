from unittest import TestCase

from numpy.fft import fft, ifft
import numpy as np

from Implementations.DPRegularWithCount import DynamicProgrammingWithCount
from Implementations.helpers.Helper import reduceToMultisetWithCardinalityAtMostTwo, divideAndConquerSumSet, sumSetNotComplex, \
    ListToPolynomial, toNumbers


class Test(TestCase):

    def test_cardinality_at_most_two_function(self):
        vals = [1, 1, 1, 10, 10, 25, 25, 25, 31]
        expected = [1, 2, 10, 10, 25]
        T = 30
        filtered = reduceToMultisetWithCardinalityAtMostTwo(vals, T)
        self.assertListEqual(filtered, expected)

    def test_DP_solver(self):
        alreadyKnown = [7]
        vals = [1,2,3]
        runner = DynamicProgrammingWithCount()
        proposedSolution = runner.subsetSumDP(vals, 12)
        print()

    def test_cardinality_at_most_two_function_with_mult_4(self):
        vals = [1, 1, 1, 10, 10, 25, 25, 25, 25, 25, 25, 31]
        expected = [1, 2, 10, 10, 25, 25, 31, 50, 50]
        T = 100
        filtered = reduceToMultisetWithCardinalityAtMostTwo(vals, T)
        self.assertListEqual(filtered, expected)

    def test_cardinality_at_most_two_handles_cardinality_overlap(self):
        vals = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 8]
        expected = [1, 2, 2, 4, 8, 16]
        T = 100
        filtered = reduceToMultisetWithCardinalityAtMostTwo(vals, T)
        self.assertListEqual(filtered, expected)

    def test_all_sumsets_in_S(self):
        vals = [1, 2, 3, 4]
        sums = divideAndConquerSumSet(vals, 11)
        self.assertListEqual(sums, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_sum_set_not_complex(self):
        A, B = [3, 2, 1, 0], [0, 10, 20]
        sumset = sumSetNotComplex(A, B)
        self.assertListEqual(sumset, [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])

    def test_sum_set_withThreshold(self):
        A, B = [3, 2, 1, 0], [0, 10, 20]
        sumset = sumSetNotComplex(A, B, 11)
        self.assertListEqual(sumset, [0, 1, 2, 3, 10, 11])

    def test_convertIFFTSumsToRealValues(self):
        eps = 0.00001
        A = [3, 2, 1]
        polyA = ListToPolynomial(A)
        ptWiseA = fft(polyA)
        ShouldBeA = ifft(ptWiseA)
        threshold = 10
        sums = toNumbers(np.select([ShouldBeA[:int(threshold + 1)] > eps], [1]))
        self.assertListEqual(sums, toNumbers(polyA))
