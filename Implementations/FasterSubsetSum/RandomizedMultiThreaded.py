import math
import os
import threading
import concurrent.futures
import time
from concurrent.futures._base import as_completed
from multiprocessing import Process

import numpy as np
import shutil
from joblib import Parallel, delayed, dump, load, parallel_backend
from joblib.externals.loky import set_loky_pickler

from Implementations.helpers.Helper import toNumbers
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase


class RandomizedMultiThreaded(NearLinearBase):

    def __init__(self, debug, repetitions, threads):
        super().__init__(debug, repetitions)
        self.threads = threads

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        self.n = n
        Z = np.array(Z)
        Zi = self.partitionIntoLayers(Z, n, t)
        S = [1]
        if len(Zi[0]) > 1:
            S = Zi[0]
            if len(Zi) == 1:
                S = self.ColorCodingLayer(S, t, len(Z), delta / (math.ceil(math.log2(n))))
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(Zi, data_filename_memmap)
        memmap = load(data_filename_memmap, mmap_mode='r')
        # 2. Parallelization
        # set_loky_pickler('pickle')
        # byThreads = [i for i in range()]
        # First, schedule 1 thread to create more processes
        # And schedule another thread to compute the other layers.
        with concurrent.futures.ProcessPoolExecutor(self.threads) as executor:
            futures = [executor.submit(self.ColorCodingLayerMulti, memmap, i,
                                       t,
                                       pow(2, i + 1) - 1,
                                       delta / (math.ceil(math.log2(n)))
                                       ) for i in range(1, len(Zi))]
            for f in as_completed(futures):
                S = self.sumSet(S, f.result(), t)
        # with Parallel(self.threads, prefer='processes', backend='multiprocessing', batch_size=1) as parallel:
        #     jobs = parallel(
        #         delayed(self.ColorCodingLayerMulti)
        #         (memmap,
        #          i,
        #          t,
        #          pow(2, i + 1) - 1,
        #          delta / (math.ceil(math.log2(n)))
        #          ) for i in range(1, len(Zi)))
        # range(min(2, len(Zi)), min(6, len(Zi))))  # Run threads for 3,4
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_hard_layers = executor.submit(self.run_hard_layers, memmap, t, delta, n)
        #     # for i in list(range(1, min(2, len(Zi)))) + list(range(6, len(Zi))):
        #     #     s = self.ColorCodingLayerMulti(Zi,
        #     #                                    i,
        #     #                               t,
        #     #                               pow(2, i + 1) - 1,
        #     #                               delta / (math.ceil(math.log2(n))))
        #     #     S = self.sumSet(S, s, t)
        #     # future_easy_layers = executor.submit(self.run_easy_layers, Zi, t, delta, n)
        #     # jobs.append(future_easy_layers.result())
        #     print('done for other layers')
        #     jobs = future_hard_layers.result()

        # with Parallel(2, verbose=10, prefer='threads') as scheduling:
        #     jobs = scheduling([delayed(self.run_hard_layers)(Zi, t, delta, n),
        #                        delayed(self.run_easy_layers)(Zi, t, delta, n)])
        # with ProcessPoolExecutor(max_workers=12) as executor:
        #     for i in range(1, len(Zi)):
        #         z = np.array(Zi[i])
        #         if len(z) > 1:
        #             ans = executor.submit(self.ColorCodingLayer, z, t, pow(2, i + 1) - 1,
        #                                   delta / (math.ceil(math.log2(n))), executor)
        #             futures.append(ans)
        #     for f in as_completed(futures):
        #         S = self.sumSet(S, f.result(), t)
        # for job in jobs:
        #     S = self.sumSet(S, job, t)
        # try:
        #     shutil.rmtree(folder)
        # except:  # noqa
        #     print('Could not clean-up automatically.')
        return toNumbers(S)

    def run_hard_layers(self, Zi, t, delta, n):
        with Parallel(6, prefer='processes', batch_size=1) as parallel:
            jobs = parallel(
                delayed(self.ColorCodingLayerMulti)
                (Zi,
                 i,
                 t,
                 pow(2, i + 1) - 1,
                 delta / (math.ceil(math.log2(n)))
                 ) for i in range(1, len(Zi)))  # range(min(2, len(Zi)), min(6, len(Zi))))  # Run threads for 3,4
        return jobs

    def color_coding_multi(self, Z, t, k, delta):
        if len(Z) == 1:
            return Z
        if self.repetitions == 0:
            repetitions = math.log(1.0 / delta, 4.0 / 3.0)
        else:
            repetitions = self.repetitions
        with Parallel(2, backend='loky', prefer='processes') as parallel:
            S = parallel((delayed(self.color_code)(Z, k, t) for j in range(0, math.ceil(repetitions))))

            # S = executor(delayed(self.color_code)(Z, k, t) for j in range(0, math.ceil(repetitions))):
            # self.color_code(S, Z, j, k, t)
            def unionValues(low, high):
                union = S[low]
                for j in range(low + 1, high):
                    if len(S[j]) > len(union):
                        S[j][np.nonzero(union)[0]] = 1
                        union = S[j]
                    else:
                        union[np.nonzero(S[j])[0]] = 1
                return union

            if repetitions < 10:
                return unionValues(0, len(S))
            else:
                mid = int(len(S) // 2)
                unions = parallel([
                    delayed(unionValues)(0, mid),
                    delayed(unionValues)(mid + 1, len(S))
                ])
                unions[0][np.nonzero(unions[1])[0]] = 1
                return unions[0]

    def color_code(self, Z, k, t):
        partition = self.partitionSetIntoK(Z, k * k)  # max(int(k*k//2), 2))
        sumset = partition[0]
        for i in range(1, len(partition)):
            sumset = self.sumSet(sumset, partition[i], t)
        return sumset

    def ColorCodingLayerMulti(self, Z, i, t, l, delta):
        layers = Z[i]
        divisor = math.log2(l / delta)
        if len(layers) <= 1:
            return [1]
        if l < divisor:
            vals = self.color_coding_multi(layers, t, l, delta)
            return vals
        m = self.roundToPowerOf2(l / divisor)
        layers = self.partitionSetIntoK(layers, m)

        m = self.roundToPowerOf2(len(layers))
        while len(layers) < m:
            layers.append([1])
        gamma = 6 * divisor
        if gamma > l:
            gamma = l

        S = [self.color_coding_multi(layers[j], t, round(gamma), delta / l) for j in range(m)]

        # with parallel_backend('loky'):
        #     S = Parallel(2)(delayed(self.color_coding)(layers[j], int(2 * gamma * t / l), round(gamma), delta / l) for j in range(m))
        def recSumSetCombine(left, right):
            if right - left == 1:  # base case:
                return self.sumSet(S[left], S[right], t)
            else:  # partition into left and right.
                mid = math.ceil((left + right) // 2)
                leftSum = recSumSetCombine(left, mid)
                rightSum = recSumSetCombine(mid + 1, right)
                return self.sumSet(leftSum, rightSum, t)

        # if tree is not large enough, default to regular computation.
        if m <= 2:
            for h in range(1, int(math.log2(m)) + 1):
                threshold = min(pow(2, h) * 2 * gamma * t / l, t)
                for j in range(1, int(m / pow(2, h)) + 1):
                    S[j - 1] = self.sumSet(S[2 * j - 1 - 1], S[2 * j - 1], threshold)
            S[0] = np.array(S[0])
            return S[0]
        with parallel_backend('loky'):
            S = Parallel(2, batch_size=1, prefer='processes')(
                [delayed(recSumSetCombine)(0, int(m//2)-1),
                 delayed(recSumSetCombine)(int(m//2), m-1)]
            )
            return self.sumSet(S[0], S[1], t)
    # return recSumSetCombine(0, len(S)).result()
