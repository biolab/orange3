import unittest

from itertools import chain

import numpy
from scipy.spatial import distance

import Orange.data
from Orange.clustering import hierarchical


def flatten(seq):
    return chain(*seq)


class TestHierarchical(unittest.TestCase):
    def setUp(self):
        m = [[],
             [ 3],
             [ 2, 4],
             [17, 5, 4],
             [ 2, 8, 3, 8],
             [ 7, 5, 10, 11, 2],
             [ 8, 4, 1, 5, 11, 13],
             [ 4, 7, 12, 8, 10, 1, 5],
             [13, 9, 14, 15, 7, 8, 4, 6],
             [12, 10, 11, 15, 2, 5, 7, 3, 1]]
        self.items = ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred",
                      "Greg", "Hue", "Ivy", "Jon"]

        dist = numpy.array(list(flatten(m)), dtype=float)
        matrix = hierarchical.squareform(dist, mode="lower")
        self.m = m
        self.matrix = Orange.misc.DistMatrix(matrix)
        self.matrix.items = self.items

        self.cluster = hierarchical.dist_matrix_clustering(self.matrix)

    def test_mapping(self):
        leaves = list(hierarchical.leaves(self.cluster))
        indices = [n.value.index for n in leaves]

        self.assertEqual(len(indices), len(self.matrix.items))
        self.assertEqual(set(indices), set(range(len(self.matrix.items))))

        self.assertEqual(indices,
                         [3, 1, 2, 6, 0, 4, 8, 9, 5, 7])

    def test_order(self):
        post = list(hierarchical.postorder(self.cluster))
        seen = set()

        for n in post:
            self.assertTrue(all(ch in seen for ch in n.branches))
            seen.add(n)

        pre = list(hierarchical.preorder(self.cluster))
        seen = set()
        for n in pre:
            self.assertTrue(all(ch not in seen for ch in n.branches))
            seen.add(n)

    def test_prunning(self):
        pruned = hierarchical.prune(self.cluster, level=2)
        depths = hierarchical.cluster_depths(pruned)
        self.assertTrue(all(d <= 2 for d in depths.values()))

        pruned = hierarchical.prune(self.cluster, height=10)
        self.assertTrue(c.height >= 10 for c in hierarchical.preorder(pruned))

        pruned = hierarchical.prune(self.cluster,
                                    condition=lambda cl: len(cl) <= 3)
        self.assertTrue(len(c) > 3 for c in hierarchical.preorder(pruned))

    def test_form(self):
        m = [[0, 2, 3, 4],
             [2, 0, 6, 7],
             [3, 6, 0, 8],
             [4, 7, 8, 0]]

        m = numpy.array(m)
        dist = hierarchical.condensedform(m, mode="lower")
        numpy.testing.assert_equal(dist, numpy.array([2, 3, 6, 4, 7, 8]))
        numpy.testing.assert_equal(
            hierarchical.squareform(dist, mode="lower"), m)
        dist = hierarchical.condensedform(m, mode="upper")
        numpy.testing.assert_equal(dist, numpy.array([2, 3, 4, 6, 7, 8]))
        numpy.testing.assert_equal(
            hierarchical.squareform(dist, mode="upper"), m)

    def test_pre_post_order(self):
        tree = hierarchical.Tree
        root = tree("A", (tree("B"), tree("C")))
        self.assertEqual([n.value for n in hierarchical.postorder(root)],
                         ["B", "C", "A"])
        self.assertEqual([n.value for n in hierarchical.preorder(root)],
                         ["A", "B", "C"])
