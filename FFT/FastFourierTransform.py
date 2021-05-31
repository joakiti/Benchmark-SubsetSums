import math
import random
import numpy as np
from numpy.fft import fft, ifft
from numpy import multiply

def dft(x):
    """
    Works fine, was taken from From: https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
    :param x:
    :return:
    """
    x = np.array(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def idft(x):
    """
    Does not work
    :param x:
    :return:
    """
    array = np.asarray(x, dtype=float)
    # array length
    N = array.shape[0]
    # new array of lenght N [0, N-1]
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Calculate the exponential of all elements in the input array.
    M = np.exp(2j * np.pi * k * n / N)
    return 1 / N * np.dot(M, array)


def FFT(P):
    """
    It works, after adding - to the root of unity.
    :param P: Coefficient representation of P
    :return: point value representation of P evaluated at roots of unity of order len(P)
    """
    n = len(P)  # perhaps pad to power of 2
    if n % 2 > 0:
        raise ValueError("must be a power of 2")
    if n <= 2:
        return dft(P)
    Peven, Podd = P[::2], P[1::2]
    valsEven = FFT(Peven)
    valsOdd = FFT(Podd)
    vals = [0] * n
    ω = pow(math.e, -2j * math.pi / n)
    return mergeSolutions(n, vals, valsEven, valsOdd, ω)

def mergeSolutions(n, vals, valsEven, valsOdd, ω):
    for i in range(int(n / 2)):
        vals[i] = valsEven[i] + pow(ω, i) * valsOdd[i]
        vals[i + int(n / 2)] = valsEven[i] - pow(ω, i) * valsOdd[i]
    return vals

def IFFT(P):
    """
    Does not work
    :param P: point value representation of P evaluated at roots of unity
    :return: P in coefficient representation
    """
    return np.multiply(private_IFFT(P), np.full((len(P)), 1 / len(P)))


def private_IFFT(P):
    """Runs FFT, but it doesnt work correctly, as it seems the output is different.
    """
    n = len(P)  # perhaps pad to power of 2
    if n % 2 > 0:
        raise ValueError("must be a power of 2")
    if n <= 2:
        return dft(P)
    Peven, Podd = P[::2], P[1::2]
    valsEven = FFT(Peven)
    valsOdd = FFT(Podd)
    vals = [0] * n
    ω = pow(math.e, 2j * math.pi / n)
    return mergeSolutions(n, vals, valsEven, valsOdd, ω)
