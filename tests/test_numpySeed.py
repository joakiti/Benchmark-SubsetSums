import unittest
from unittest import TestCase

import numpy as np
from numpy import polymul
from numpy.fft import fft

from FFT.FastFourierTransform import FFT
from Implementations.DynamicProgramming import DynamicProgramming
from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.FastIntegersFromJournal import FastIntegersFromJournal
from Implementations.helpers.Helper import sumSetNotComplex, sumSetIntMult, ListToPolynomial, toNumbers
from Implementations.Unbounded import Unbounded
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class Test(TestCase):

    @unittest.expectedFailure
    def test_no_seed(self):
        x = np.random.randint(0, 1000, 10)
        y = np.random.randint(0, 1000, 10)
        x.sort()
        y.sort()
        self.assertListEqual(x.tolist(), y.tolist())

    @unittest.expectedFailure
    def test_numpy_seed_works_once(self):
        np.random.seed(10)
        x = np.random.randint(0, 1000, 10)
        y = np.random.randint(0, 1000, 10)
        x.sort()
        y.sort()
        self.assertListEqual(x.tolist(), y.tolist())

    def test_numpy_seed_for_all(self):
        np.random.seed(10)
        x = np.random.randint(0, 1000, 10)
        np.random.seed(10)
        y = np.random.randint(0, 1000, 10)
        x.sort()
        y.sort()
        self.assertListEqual(x.tolist(), y.tolist())

    # Run this twice, assert contents are the same.
    def test_numpy_seed_working_progressively(self):
        np.random.seed(10)
        x = np.random.randint(0, 1000, 10)
        y = np.random.randint(0, 1000, 10)
        x.sort()
        y.sort()
        print(x)
        print(y)
