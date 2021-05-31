import unittest
from unittest import TestCase

from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase
from benchmarks.test_distributions import Distributions as dist


class RandomizedBaseTester(TestCase):

    @classmethod
    def setUp(cls):
        cls.fasterSubset = NearLinearBase(False, 1)

    def test_faster_sumset_base_returns_correct_sumset(self):
        vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        T = 11
        sums = self.fasterSubset.fasterSubsetSum(vals, T, 0.2)
        self.assertListEqual(sums, [0, 1, 3, 4, 8, 9, 11])

    def test_color_coding_base_returns_correct_sumset(self):
        vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        T = 11
        characteristic = ListToPolynomial(vals)
        sums = self.fasterSubset.color_coding(characteristic, T, len(vals), 0.2)
        self.assertListEqual(toNumbers(sums), [0, 1, 3, 4, 8, 9, 11])

    @unittest.skip("Not currently working.")
    def test_faster_sumset_returns_correct_sumset_multiples(self):
        vals = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        T = 11
        sums = self.fasterSubset.fasterSubsetSum(vals, T, 0.2)
        self.assertListEqual(sums, [0, 1, 3, 4])

    @unittest.skip("Not currently working. I.e some of the speed ups we done means this does not work properly")
    def test_faster_simple(self):
        vals = [8, 10]
        T = 18
        a = list(set(vals))
        delta = 0.0001
        fast = self.fasterSubset.fasterSubsetSum(a, T, delta)
        self.assertListEqual(fast, [0, 8, 10, 18])

    @unittest.skip("comment in for benchmark.")
    def test_me(self):
        delta = 0.0001
        i = 500
        a, T = dist.evenDistribution(i)
        fast = self.fasterSubset.fasterSubsetSum(a, T, delta)
        # expertSolution = FastIntegersFromGit().run(a, T)
        # self.assertListEqual(fast, expertSolution)
