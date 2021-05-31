from Implementations.FastIntegersFromGit import FastIntegersFromGit
from Implementations.helpers.Helper import ListToPolynomial, toNumbers
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase

class MixOfDeterministicAndRandomized(NearLinearBase):

    def __init__(self, debug):
        super().__init__(debug)

    def color_coding(self, Z, t, k, delta):
        color_coder = FastIntegersFromGit()
        ZPrime = toNumbers(Z)
        ZPrime = color_coder.run(ZPrime, t)
        return ListToPolynomial(ZPrime)
