from unittest import TestCase
import numpy as np

from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.PowerSeriesSubsetSum import PowerSeriesSubsetSum
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.powerSeries = PowerSeriesSubsetSum('Powerseries')


    def test_fast_integer_returns_correct_sumset(self):
        vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        T = 11
        expected =  [0, 1, 3, 4, 8, 9, 11]
        sums = self.powerSeries.run(vals, T)
        self.assertListEqual(sums, expected)


    def test_benchmark(self):
        vals, t = Distributions.clusteredDistributionEven(9)
        self.powerSeries.run(vals, t)
