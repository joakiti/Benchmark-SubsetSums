from unittest import TestCase
import numpy as np

from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.FastIntegersFromJournal import FastIntegersFromJournal
from Implementations.FastIntegersPersonal import FastIntegersPersonal
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase
from benchmarks.test_distributions import Distributions


class Test(TestCase):

    @classmethod
    def setUp(self):
        self.fastIntegers = FastIntegersFromJournal('does not matter')
        self.fastIntegersFromGit = FastIntegersFromGit()
        self.RandomizedFasterSubsetSum = NearLinearBase(False, 'near_linear', 1)

    def test_find_bug(self):
        i = 5
        delta = 0.0001
        a, t = Distributions().HighEndDistribution(i)
        tested = self.fastIntegers.run(a, t)
        tested.sort()
        expected = self.fastIntegersFromGit.run(a, t)
        self.assertListEqual(tested, expected)
