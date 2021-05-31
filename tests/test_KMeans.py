from unittest import TestCase

from Implementations.helpers.ComputeKMeansGrouping import ComputeKMeansGrouping


class Test(TestCase):

    def test_kMeans(self):
        xs = [1, 2, 3, 10, 11, 12, 20, 23]
        kmeans = ComputeKMeansGrouping(xs)
        kmeans.computeClusters(3)
        solution = kmeans.clusters(3)
        self.assertListEqual(solution[0], [1, 2, 3])
        self.assertListEqual(solution[1], [10, 11, 12])
        self.assertListEqual(solution[2], [20, 23])

    def test_kMeans_NClusters(self):
        xs = [1000, 100, 203, 2031, 539, 439, 120392, 2939]
        kmeans = ComputeKMeansGrouping(xs)
        kmeans.computeClusters(8)
        solution = kmeans.clusters(len(xs))
        expected = [[100], [203], [439], [539], [1000], [2031], [2939], [120392]]
        self.assertListEqual(solution, expected)