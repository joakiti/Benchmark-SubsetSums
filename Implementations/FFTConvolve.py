from pyfftw import pyfftw
import pyfftw.builders
import numpy as np

"""
This is used to test pyfftw, in comparison to scipy's fft.convolve
A similar approach was used to test Numpy's aswell.
"""
class fftconvolver:

    def convolve(self, A, B):
        rowsB = B.shape[0]
        colsB = B.shape[1]
        rowsA = A.shape[0]
        colsA = A.shape[1]

        a = np.pad(A, ((0, rowsB - 1), (0, colsB - 1)), mode='constant')
        b = np.pad(B, ((0, rowsA - 1), (0, colsA - 1)), mode='constant')

        rowsA = rowsA + rowsB - 1
        colsA = colsA + colsB - 1

        threads = 1
        fft_A_obj = pyfftw.builders.fft2(a, s=(rowsA, colsA), threads=2)
        fft_B_obj = pyfftw.builders.fft2(b, s=(rowsA, colsA), threads=2)
        ifft_obj = pyfftw.builders.ifft2(fft_A_obj.output_array, s=(rowsA, colsA), threads=threads)
        return ifft_obj(fft_A_obj(a) * fft_B_obj(b))
