from unittest import TestCase
import numpy as np

from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.ILPSubsetSum import ILPSubsetSum
from Implementations.PowerSeriesSubsetSum import PowerSeriesSubsetSum
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.ilpSolver = ILPSubsetSum()


    def test_fast_integer_returns_correct_sumset(self):
        vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        T = 11
        sums = self.ilpSolver.run(vals, T)
        self.assertEqual(sum(sums), 11)

    def test_fast_integer_returns_not_found(self):
        vals = [1, 2, 4, 15, 17]
        T = 26
        sums = self.ilpSolver.run(vals, T)
        self.assertEqual(sum(sums), 24)


