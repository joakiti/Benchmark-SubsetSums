from unittest import TestCase
from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.FasterSubsetSum.RandomizedVariableLayers import RandomizedVariableExponentialLayers
from benchmarks.test_distributions import Distributions as dist


class Test(TestCase):

    @classmethod
    def setUp(cls):
        cls.fasterSubset = RandomizedVariableExponentialLayers(False, 3, 'variable layers', 1)

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

    def test_me(self):
        delta = 0.0001
        i = 20
        a, T = dist.clusteredDistributionEven(i)
        fast = self.fasterSubset.fasterSubsetSum(a, T, delta)
        expertSolution = FastIntegersFromGit().run(a, T)
        self.assertListEqual(fast, expertSolution)
