# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from itertools import chain, tee

import numpy

from Orange.clustering import hierarchical
import Orange.misc


def flatten(seq):
    return chain(*seq)


class TestHierarchical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = [[],
             [3],
             [2, 4],
             [17, 5, 4],
             [2, 8, 3, 8],
             [7, 5, 10, 11, 2],
             [8, 4, 1, 5, 11, 13],
             [4, 7, 12, 8, 10, 1, 5],
             [13, 9, 14, 15, 7, 8, 4, 6],
             [12, 10, 11, 15, 2, 5, 7, 3, 1]]
        cls.items = ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred",
                     "Greg", "Hue", "Ivy", "Jon"]

        dist = numpy.array(list(flatten(m)), dtype=float)
        matrix = hierarchical.squareform(dist, mode="lower")
        cls.m = m
        cls.matrix = Orange.misc.DistMatrix(matrix)
        cls.matrix.items = cls.items

        cls.cluster = hierarchical.dist_matrix_clustering(cls.matrix)

    def test_mapping(self):
        leaves = list(hierarchical.leaves(self.cluster))
        indices = [n.value.index for n in leaves]

        self.assertEqual(len(indices), len(self.matrix.items))
        self.assertEqual(set(indices), set(range(len(self.matrix.items))))

        #self.assertEqual(indices,
        #                 [3, 1, 2, 6, 0, 4, 8, 9, 5, 7])

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

        top = hierarchical.top_clusters(self.cluster, 3)
        self.assertEqual(len(top), 3)

        top = hierarchical.top_clusters(self.cluster, len(self.matrix))
        self.assertEqual(len(top), len(self.matrix))
        self.assertTrue(all(n.is_leaf for n in top))

        top1 = hierarchical.top_clusters(self.cluster, len(self.matrix) + 1)
        self.assertEqual(top1, top)

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

    def test_optimal_ordering(self):
        def indices(root):
            return [leaf.value.index for leaf in hierarchical.leaves(root)]

        ordered = hierarchical.optimal_leaf_ordering(
            self.cluster, self.matrix)

        self.assertEqual(ordered.value.range, self.cluster.value.range)
        self.assertSetEqual(set(indices(self.cluster)),
                            set(indices(ordered)))

        def pairs(iterable):
            i1, i2 = tee(iterable)
            next(i1)
            yield from zip(i1, i2)

        def score(root):
            return sum([self.matrix[i, j] for i, j in pairs(indices(root))])
        score_unordered = score(self.cluster)
        score_ordered = score(ordered)
        self.assertGreater(score_unordered, score_ordered)
        self.assertEqual(score_ordered, 21.0)

    def test_table_clustering(self):
        table = Orange.data.Table(numpy.eye(3))
        tree = hierarchical.data_clustering(table, linkage="single")
        numpy.testing.assert_almost_equal(tree.value.height, numpy.sqrt(2))

        tree = hierarchical.feature_clustering(table)
        numpy.testing.assert_almost_equal(tree.value.height, 0.75)


class TestTree(unittest.TestCase):
    def test_tree(self):
        Tree = hierarchical.Tree

        left = Tree(0, ())
        self.assertTrue(left.is_leaf)
        right = Tree(1, ())
        self.assertEqual(left, Tree(0, ()))
        self.assertNotEqual(left, right)
        self.assertLess(left, right)

        root = Tree(2, (left, right))
        self.assertFalse(root.is_leaf)
        self.assertIs(root.left, left)
        self.assertIs(root.right, right)

        val, br = root

        self.assertEqual(val, 2)
        self.assertEqual(br, (left, right))
        self.assertEqual(repr(left), "Tree(value=0, branches=())")
