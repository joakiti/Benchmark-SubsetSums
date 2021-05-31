import math
import multiprocessing
import os
import queue
import time
from collections import defaultdict
from copy import copy
from multiprocessing import shared_memory
import copy

import numpy as np
from scipy.signal import fftconvolve

from Implementations.helpers.Helper import toNumbers, ListToPolynomial
from Implementations.FasterSubsetSum.RandomizedBase import NearLinearBase

def partitionSetIntoKGenerator(Z, k):
    k = math.ceil(k)
    partition = np.zeros((k, len(Z)), dtype=np.dtype('u1'))  # Otherwise we use too much memory.
    listUsed = set()
    for i in np.nonzero(Z)[0][1:]:  # Ignore 0 component with 1:
        goesTo = np.random.randint(0, k)
        partition[goesTo][i] = 1
        partition[goesTo][0] = 1
        listUsed.add(goesTo)
    for x in listUsed:
        yield partition[x][:max(np.nonzero(partition[x])[0]) + 1]


def partitionSetIntoKRegularNumbers(Z, k):
    k = math.ceil(k)
    partition = defaultdict(list)
    listUsed = set()
    for i in Z:  # Ignore 0 component with 1:
        goesTo = np.random.randint(0, k)
        partition[goesTo].append(i)
        listUsed.add(goesTo)
    return [partition[x] for x in listUsed]


def sumSet(A, B, threshold):
    eps = 0.0001  # account for floating error
    AsumsetB = fftconvolve(A, B)
    return np.array(np.select([AsumsetB[:int(threshold + 1)] > eps], [1]), dtype=np.dtype('u1'))


def roundToPowerOf2(m):
    return pow(2, math.ceil(math.log2(m)))


class ColorCodingWorker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, threads):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.threads = threads

    def run(self):
        proc_name = self.name
        tasksRun = 0
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                # print('%s: Exiting' % proc_name)
                # print(combineTasksDone)
                self.task_queue.task_done()
                print(tasksRun)
                break
            # print('%s: %s' % (proc_name, next_task))
            if isinstance(next_task, ColorCodingTask):
                next_task(self.task_queue)
                self.task_queue.task_done()
            else:
                start = time.time()
                result = next_task()
                end = time.time()
                tasksRun += 1

                self.result_queue.put(result)
                self.task_queue.task_done()
        return


class ColorCodingLayerWorker(multiprocessing.Process):

    def __init__(self, task_queue, color_queue, result_que, shr_name, dim):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.color_queue = color_queue
        self.results_que = result_que
        self.shr_name = shr_name
        self.dim = dim

    def run(self):
        proc_name = self.name
        existing_shm = shared_memory.SharedMemory(name=self.shr_name)
        np_array = np.ndarray(self.dim, dtype=np.int64, buffer=existing_shm.buf)
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                # print('%s: Exiting' % proc_name)
                existing_shm.close()
                existing_shm.unlink()
                self.task_queue.task_done()
                break
            # mp_array, np_array = self.shared_memory
            # Load the numpy array from memory, copy to avoid inconsisetency
            vals = np_array[next_task.start:next_task.end]
            # print('%s: %s' % (proc_name, next_task))
            next_task(vals, self.color_queue)
            # print('%s: solved %s in %d' % (proc_name, next_task, end - start))
            self.task_queue.task_done()
        return


class CombineTask(object):

    def __init__(self, Z, t, layer, m, j):
        self.Z = Z
        self.t = t
        self.layer = layer
        self.m = m
        self.j = j

    def __call__(self):
        start = time.time()
        if len(self.Z) == 0:
            return Result(self.layer, self.j, self.m, [0])
        ans = ListToPolynomial(self.Z[0])
        for i in range(1, len(self.Z)):
            if len(self.Z[i]) == 0:
                continue
            ans = sumSet(ans, ListToPolynomial(self.Z[i]), self.t)
        end = time.time()
        if self.layer == 5:
            print('Solved %s in %f' % (self, end - start))
        return Result(self.layer, self.j, self.m, toNumbers(ans))

    def __str__(self):
        return 'CombineTask %d' % self.layer


class ColorCodingTask(object):

    def __init__(self, repetitions, Z, t, k, delta, threads, layer, j=None, m=None):
        self.repetitions = repetitions
        self.Z = Z
        self.t = t
        self.k = k
        self.delta = delta
        self.threads = threads
        self.layer = layer
        self.j = j
        self.m = m

    def __call__(self, combine_que):
        repetitions = self.repetitions
        for j in range(0, math.ceil(repetitions)):
            partition = partitionSetIntoKRegularNumbers(self.Z, self.k * self.k)  # max(int(k*k//2), 2))
            if len(partition) < 20:  # Then do the work ourselves.
                combine_que.put(CombineTask(partition, self.t, self.layer, self.m, self.j))
            else:  # Distribute the workload
                partitionInto = 2
                threadPerWork = math.ceil(len(partition) / partitionInto)
                for threadPartition in range(0, partitionInto):
                    combine_que.put(CombineTask(partition[threadPartition * threadPerWork: min(
                        (threadPartition + 1) * threadPerWork, len(partition))], self.t, self.layer, self.m, self.j))

    def __str__(self):
        return 'ColorCoding %d' % self.layer


class Result(object):

    def __init__(self, layer, j, m, result):
        self.layer = layer
        self.j = j
        self.m = m
        self.result = result


class ColorCodingLayerTask(object):
    def __init__(self, start, end, i, t, l, delta, threads):
        self.start = start
        self.end = end
        self.i = i
        self.t = t
        self.l = l
        self.delta = delta
        self.threads = threads

    def __call__(self, Z, color_coding_queue):
        divisor = math.log2(self.l / self.delta)
        if self.l < divisor:
            # color_coding_queue.put
            # TODO: Add data to identify this solution
            color_coding_queue.put(ColorCodingTask(1, Z, self.t, self.l, self.delta, self.threads, self.i))
            return
            # return color_coding(1, Z, self.t, self.l, self.delta)
        m = roundToPowerOf2(self.l / divisor)
        partition = partitionSetIntoKRegularNumbers(Z, m)
        m = roundToPowerOf2(len(partition))
        while len(partition) < m:
            partition.append([0])
        gamma = 6 * divisor

        if gamma > self.l:
            gamma = self.l
        t = self.t
        if 2*gamma*t/self.l <= t:
            t = 2 * gamma * t / self.l
        # Put color coding jobs available on the queue
        for j in range(m):
            # TODO: Add data to identify this solution
            color_coding_queue.put(
                ColorCodingTask(1, partition[j], t, round(gamma), self.delta / self.l, self.threads, self.i, j, m)
            )
        return

    def __str__(self):
        return 'ColorCodingLayer %d' % self.i


def create_shared_block(data):
    a = copy.deepcopy(data)  # Start with an existing NumPy array

    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(a.shape, dtype=np.int64, buffer=shm.buf)
    np_array[:] = a[:]  # Copy the original data into shared memory
    return shm, np_array

class RandomizedMultiThreadedVer2(NearLinearBase):

    def __init__(self, debug, repetitions, threads):
        super().__init__(debug, repetitions)
        self.threads = threads
        self.label = '%d threads' % threads


    def prioritize(self, Z, l, delta):
        divisor = math.log2(l / delta)
        if l < divisor:
            return 0
        if len(Z) <= 10:
            return 0
        return len(Z) * math.log2(len(Z)) * divisor

    def partitionIntoLayers(self, Z, n, t):
        Zi = [Z[(t / pow(2, i) <= Z) & (Z < t / pow(2, i - 1))] for i in
              range(1, math.ceil(math.log2(n)))]
        Zi.append(Z[(0 <= Z) & (Z < t / pow(2, math.ceil(math.log2(n)) - 1))])
        if self.debug:
            self.layerInformation = list()
            for i in range(len(Zi)):
                self.layerInformation.append((len(Zi[i]), t / pow(2, i)))
            self.layerInformation.append((len(Zi[len(Zi) - 1]), 0))
        for i in range(len(Zi)):
            if len(Zi[i]) == 0:
                Zi[i] = np.array([0])
        return Zi

    def fasterSubsetSum(self, Z, t, delta):
        n = len(Z)
        self.n = n
        Z = np.array(Z)
        Zi = self.partitionIntoLayers(Z, n, t)

        # partition_with_index = [(index, value) for index, value in enumerate(Zi)]
        # partition_with_index.sort(key=lambda x: self.prioritize(x[1], math.pow(2, x[0] + 1) - 1,
        #                                                         delta / (math.ceil(math.log2(n)))), reverse=True)
        # partition_with_index = list(map(itemgetter(0), partition_with_index))
        # partition_with_index.remove(0)

        # Zi = np.array(list(map(ListToPolynomial, Zi)))
        S = ListToPolynomial(Zi[0])
        S[0] = 1
        if len(Zi) == 1:
            S = self.ColorCodingLayer(S, t, len(Z), delta / (math.ceil(math.log2(n))))
            return toNumbers(S)
        # Each process will get 'chunksize' nums and a queue to put his out
        # dict into
        color_coding_results = multiprocessing.Queue()
        layer_queue = multiprocessing.JoinableQueue()
        color_queue = multiprocessing.JoinableQueue()

        # Align all partitions into a single layer (to reduce overhead of copying)
        # Make all layers shared across memory
        layerToInterval = []
        nextIndex = 0
        allVals = []
        for value in Zi:
            layerToInterval.append((nextIndex, nextIndex + len(value)))
            nextIndex = nextIndex + len(value)
            # Compose all partitions into one big list
            allVals = allVals + list(value)

        allVals = np.array(allVals, dtype=np.int64)

        shr, np_array = create_shared_block(allVals)
        color_workers = [ColorCodingWorker(color_queue, color_coding_results, self.threads)
                         for process in range(self.threads)]
        layer_worker = ColorCodingLayerWorker(layer_queue, color_queue, color_coding_results, shr.name, allVals.shape)
        for w in color_workers:
            w.start()
        layer_worker.start()

        numJobs = 0
        asd = time.time()
        for i in range(1, len(Zi) // 2):  # We take the strongest layers, and then solve the easy layers.
            numJobs += 1
            interval = layerToInterval[i]
            start = interval[0]
            end = interval[1]
            layer_queue.put(
                ColorCodingLayerTask(start, end, i + 1, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                     self.threads))
        for i in range(len(Zi) // 2, len(Zi)):
            z = ListToPolynomial(Zi[i])
            if len(z) > 1:
                Si = self.ColorCodingLayer(z, t, pow(2, i + 1) - 1, delta / (math.ceil(math.log2(n))),
                                           high=pow(2, i) if i != len(Zi) - 1 else (2 ** i, "Last is zero"))
                S = self.sumSet(Si, S, t)
        # Wait for all layer codings and color codings to complete
        layer_queue.join()
        color_queue.join()
        layer_queue.put(None)
        layer_queue.join()
        for process in range(self.threads):
            color_queue.put(None)
        color_queue.join()
        asdfg = time.time()
        print('Time to compute all solutions:', asdfg - asd)
        results = list()

        start = time.time()
        while True:
            try:
                results.append(color_coding_results.get(timeout=2))
            except queue.Empty:
                break
        print('result length:', len(results))
        combineAndAppendToS = defaultdict()
        binaryTreeSumWay = defaultdict(lambda: defaultdict(list))
        for result in results:
            # Either, it belongs to a sumset from color coding? So should be combined with existing sumsets.
            if result.m is None:
                if result.layer not in combineAndAppendToS:
                    combineAndAppendToS[result.layer] = ListToPolynomial(result.result)
                else:
                    combineAndAppendToS[result.layer] = self.sumSet(ListToPolynomial(result.result),
                                                                    combineAndAppendToS[result.layer], t)
            else:
                if result.j not in binaryTreeSumWay[result.layer][result.j]:
                    binaryTreeSumWay[result.layer][result.j] = ListToPolynomial(result.result)
                else:
                    binaryTreeSumWay[result.layer][result.j] = self.sumSet(binaryTreeSumWay[result.layer][result.j],
                                                                           ListToPolynomial(result.result), t)

        for binaryTreeComputation in binaryTreeSumWay.values():
            m = len(binaryTreeComputation)
            for h in range(1, int(math.log2(m))):
                threshold = t
                for j in range(1, int(m / pow(2, h)) + 1):
                    binaryTreeComputation[j - 1] = self.sumSet(binaryTreeComputation[2 * j - 1 - 1],
                                                               binaryTreeComputation[2 * j - 1], threshold)
            self.sumSet(S, binaryTreeComputation[0], t)

        for color_coding_list in combineAndAppendToS.values():
            S = self.sumSet(S, color_coding_list, t)
        end = time.time()
        print('Time to combine all solutions:', end - start)
        del layer_queue
        del color_queue
        for worker in color_workers:
            del worker
        del color_workers
        del layer_worker
        del color_coding_results
        # while numJobs:
        #     S = sumSet(S, results.get(), t)
        #     numJobs -= 1
        # for p in procs:
        #     S = sumSet(S, out_q.get(), t)
        return toNumbers(S)
