from unittest import TestCase

import numpy as np
from numpy import polymul
from numpy.fft import fft

from FFT.FastFourierTransform import FFT
from Implementations.DPRegularWithCount import DynamicProgrammingWithCount
from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.FastIntegersFromJournal import FastIntegersFromJournal
from Implementations.helpers.Helper import sumSetNotComplex, sumSetIntMult, ListToPolynomial, toNumbers
from Implementations.Unbounded import Unbounded
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.fasterSubset = NearLinearBase(False, 'fast')

    def test_unbounded_sumset_returns_correct_sumset(self):
        vals = [13, 21, 32, 49]
        T = 27
        sums = Unbounded().run(vals, T)
        self.assertListEqual(sums, [0, 13, 21, 26])

    def test_onlyUniqueSolutions(self):
        vals = Distributions.single_solution_to_all(10)
        print(vals)

    # def test_DP_Solution_returns_correct_sumset(self):
    #     vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
    #     expected = [0, 1, 3, 4, 8, 9, 11]
    #     T = 11
    #     sums = DynamicProgrammingWithCount().run(vals, T)
    #     self.assertListEqual(sums, expected)

    def test_Deterministic_returns_correct_sumset(self):
        vals = [1, 2, 3, 4]
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        T = 11
        sums = FastIntegersFromGit.run(vals, T)
        self.assertListEqual(expected, sums)

    def test_deterministic_integers_journal(self):
        vals = [1, 2, 3, 4]
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        T = 11
        sums = FastIntegersFromJournal('.').run(vals, T)
        self.assertListEqual(expected, list(sums))

    def test_everyone_agrees(self):
        i = 1
        np.random.seed(i)
        delta = 0.2
        # generate some integers
        a = np.random.random_integers(0, 1000 * i, 100 * i)
        a = list(set(a))
        T = 2000
        fast = self.fasterSubset.fasterSubsetSum(a, T, delta)
        dp = DynamicProgramming('.', benchmarkMode=False).run(a, T)
        deterministic = FastIntegersFromGit.run(a, T)
        self.assertListEqual(fast, dp)
        self.assertListEqual(deterministic, dp)

    def test_compute_sum_set_versus_n2(self):
        A, B = [3, 2, 1, 0], [0, 5, 4]
        sumset = sumSetNotComplex(A, B)
        self.assertListEqual(sb_mul(A, B), sumset)

    def test_FFT(self):
        x = np.random.random(1024)
        self.assertTrue(np.allclose(FFT(x), np.fft.fft(x)))

    def test_toPolynomial(self):
        A = [6, 3, 2, 1]
        #Redefined threshold to be responsibility of caller
        polyA = ListToPolynomial(A)
        expected = np.array([1, 1, 1, 1, 0, 0, 1])
        self.assertListEqual(list(polyA), list(expected))

    def test_sum_set_compared_to_polynomial_mult(self):
        A, B = [3, 2, 1, 0], [5, 4, 0]
        sumset = sumSetNotComplex(A, B)
        polyA = ListToPolynomial(A)
        polyB = ListToPolynomial(B)
        # Numpy assumes coefficients arrive in reversed order
        p1 = np.poly1d(np.flip(polyA))
        p2 = np.poly1d(np.flip(polyB))
        # And then we have to flip again, and here we pass on the coefficients of the polynomial.
        self.assertEqual(toNumbers(np.flip(polymul(p1, p2).coeffs)), sumset)

    def test_sum_set_integer_multiplication(self):
        A, B = [3, 2, 1, 0], [0, 10, 20]
        sumset = sumSetIntMult(ListToPolynomial(A), ListToPolynomial(B), 23)
        self.assertListEqual(toNumbers(sumset), [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])

    def test_divide_and_conquer(self):
        res = divideandconquer([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertTrue(res == 32 + 13)



def test_mul(arr_a, arr_b):
    """Runs polynomial multiplication from numpy. This is assumed to be working perfectly"""
    c = polymul(arr_a, arr_b)
    return c


def sb_mul(arr_a, arr_b):  # naive impl
    vals = set()
    for i in range(len(arr_a)):
        for j in range(len(arr_b)):
            vals.add(arr_a[i] + arr_b[j])
    return list(vals)


def rec(bits, frm, to):
    """
    Returns integer value of bits in bits[frm:to]
    :param bits:
    :param frm:
    :param to:
    :return:
    """
    if frm == to:
        return bits[frm]
    size = (to - frm) // 2
    mid = frm + size
    a, b = rec(bits, frm, mid), rec(bits, mid + 1, to)
    return (a << (to - mid)) | b


def divideandconquer(bits):
    return rec(bits, 0, len(bits) - 1)
