from unittest import TestCase
import numpy as np

from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.FastIntegersFromJournal import FastIntegersFromJournal
from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.ILPSubsetSum import ILPSubsetSum
from Implementations.PowerSeriesSubsetSum import PowerSeriesSubsetSum
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.fip = FastIntegersPersonal('FIP', benchmarkMode=True)


    def test_fast_integer_returns_correct_sumset(self):
        vals = [1, 15, 3, 8, 120, 290, 530, 420, 152, 320, 150, 190]
        T = 11
        sums = self.fip.run(vals, T)
        self.assertEqual(max(sums), 11)

    def test_fast_integer_returns_not_found(self):
        vals = [1, 2, 4, 15, 17]
        T = 26
        sums = self.fip.run(vals, T)
        self.assertEqual(max(sums), 24)

    def test_bad_performance(self):
        DistributionGenerator = Distributions()
        a, t = DistributionGenerator.AVIS(100)
        sums = self.fip.run(a, t)
        self.assertEqual(max(sums), t)




