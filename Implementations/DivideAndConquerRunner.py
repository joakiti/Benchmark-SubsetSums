import math

from Implementations.Interfaces.IDeterministicAlgorithm import IDeterministicAlgorithm
from Implementations.helpers import Helper
from Implementations.helpers.Helper import ListToPolynomial, toNumbers, sumSet


class DivideAndConquerRunner(IDeterministicAlgorithm):

    def __init__(self, label, benchmarkMode=False):
        self.label = label
        self.benchmarkMode = benchmarkMode

    def run(cls, values, target):
        return Helper.divideAndConquerSumSet(values, target)
