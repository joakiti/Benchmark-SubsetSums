import heapq as hp
import itertools
import math
from operator import itemgetter

import numpy as np
from bitarray._bitarray import bitarray
from bitarray.util import ba2int
from scipy.signal import fftconvolve


def eps():
    return 0.0001


def ListToPolynomial(S):
    M = np.zeros(max(S) + 1, dtype=np.dtype('u1'))
    M[S] = 1
    M[0] = 1
    return M


def toNumbers(A):
    """Gives as input the polynomial A in coefficient representation, where the value of a coefficient
        denotes if the exponent is a part of the polynomial. If coefficient is greater than 0, then value is included
    """
    return [x for x in np.nonzero(A)[0]]


def sumSetNotComplex(A, B, threshold=2 ** 31 - 2):
    MA, MB = ListToPolynomial(A), ListToPolynomial(B)
    invMC = fftconvolve(MA, MB)
    invMC = invMC[:int(threshold + 1)]

    return [x for x in np.nonzero([invMC > eps()])[1]]


def convolution(A, B, threshold=2 ** 31 - 2):
    invMC = fftconvolve(A, B)
    return [x.real for x in invMC]


def padWithZero(S, amount=1):
    if len(S) == 0:
        return np.zeros(amount, dtype=int)
    return np.pad(S, (0, amount), 'constant')


def sumSet(A, B, threshold):
    eps = 0.0001  # account for floating error
    AsumsetB = fftconvolve(A, B)
    return np.select([AsumsetB[:int(threshold + 1)] > eps], [1])


def bruteForceSolve(vals, t):
    for i in range(1, len(vals) + 1):
        for subset in itertools.combinations(vals, i):
            subsetSum = sum(subset)
            if subsetSum <= t:
                yield subsetSum
    yield 0


def sumSetIntMult(A, B, t):
    """
    This is the second version of int mult, that runs slower, but slightly easier to read.
    :param A:
    :param B:
    :param t:
    :return:
    """
    if len(A) <= 1:
        return B
    if len(B) <= 1:
        return A
    a = list()
    log2T = int(math.ceil(math.log2(t)))
    for i in range(0, len(A) - 1):
        a.append(A[i])
        for _ in range(log2T):
            a.append(0)
    a.append(A[len(A) - 1])

    b = list()
    for i in range(0, len(B) - 1):
        b.append(B[i])
        for _ in range(log2T):
            b.append(0)
    b.append(B[len(B) - 1])
    """
        def rec(bits, frm, to):
            Returns integer value of bits in bits[frm:to]
            :param bits:
            :param frm:
            :param to:
            :return:
            if frm == to:
                return bits[frm]
            size = (to - frm) // 2
            mid = frm + size
            a, b = rec(bits, frm, mid), rec(bits, mid + 1, to)
            return (a << (to - mid)) | b

        def divideandconquer(bits):
            return rec(bits, 0, len(bits) - 1)

        def bitsToBytes(a):
            returnInts = []
            for i in range(0, len(a), 8):
                returnInts.append(int(''.join(str(x) for x in a[i:i + 8]), 2))  # goes 8 bits at a time to save as ints
            return returnInts

        def iterative(bits):
            step = 1
            while step < len(bits):
                for i in range(-1, -len(bits) + step - 1, -2 * step):
                    bits[i] += bits[i - step] << step
                step *= 2
            return bits[-1]
    """
    f = ba2int(bitarray(a))
    g = ba2int(bitarray(b))
    z = np.binary_repr(f * g)
    # np.select([z[:int(threshold + 1)] > eps], [1])
    sumsets = [1 if int(z[x:x + log2T + 1]) > 0 else 0 for x in range(0, len(z), log2T + 1)]
    return np.array(sumsets[:t + 1])


def reduceToMultisetWithCardinalityAtMostTwo(vals, T):
    def countCardinality(heap, value):
        cardinality = 1
        while len(heap) > 0 and heap[0][0] == value:
            (val, card) = hp.heappop(heap)
            if card != -1:
                cardinality += card
            else:
                cardinality += 1
        return cardinality

    heap = []
    for value in vals:
        # Initialize cardinality to -1, because we simply dont know it at this point
        hp.heappush(heap, (value, -1))
    result = []
    while len(heap) > 0:
        (value, cardinality) = hp.heappop(heap)
        if value > T:
            return result
        # Now since we know the cardinality we should be using it
        # Check if cardinality is set
        if cardinality == -1:
            cardinality = countCardinality(heap, value)
        if cardinality <= 2:
            # Nothing we can do
            for i in range(cardinality):
                result.append(value)
            continue
        else:
            for i in range(cardinality - 2 * math.floor((cardinality - 1) / 2)):
                # Append it once or twice.
                result.append(value)
            hp.heappush(heap, (2 * value, math.floor((cardinality - 1) / 2)))
    return result


def divideAndConquerSumSet(S, t):
    if len(S) == 1:
        return [*S, 0]
    if len(S) == 0:
        return [0]
    L = S[1::2]
    R = S[0::2]
    sumL = divideAndConquerSumSet(L, t)
    sumR = divideAndConquerSumSet(R, t)
    LR = sumSetNotComplex(sumL, sumR, t)
    return LR


def ListToMatrix(S):
    c1 = list(map(itemgetter(0), S))
    c2 = list(map(itemgetter(1), S))
    M = np.zeros((max(c1) + 1, max(c2) + 1), dtype=int)
    M[c1, c2] = 1
    M[0, 0] = 1
    return M


def MatrixToList(M, a, t):
    return list(zip(*np.nonzero(np.select([M[:t + 1, :int(a) + 1] > eps()], [1]))))
