# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest import TestCase
import os
import pickle

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from Orange.data import (Table, Domain, ContinuousVariable,
                         DiscreteVariable, StringVariable, Instance)
from Orange.distance import (Euclidean, SpearmanR, SpearmanRAbsolute,
                             PearsonR, PearsonRAbsolute, Manhattan, Cosine,
                             Jaccard, _preprocess, MahalanobisDistance)
from Orange.misc import DistMatrix
from Orange.tests import named_file, test_filename
from Orange.util import OrangeDeprecationWarning


def tables_equal(tab1, tab2):
    # TODO: introduce Table.__eq__() ???
    return (tab1 == tab2 or  # catches None
            np.all([i == j for i, j in zip(tab1, tab2)]))


class TestDistMatrix(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.dist = Euclidean(cls.iris)

    def test_submatrix(self):
        sub = self.dist.submatrix([2, 3, 4])
        np.testing.assert_equal(sub, self.dist[2:5, 2:5])
        self.assertTrue(tables_equal(sub.row_items, self.dist.row_items[2:5]))

    def test_pickling(self):
        unpickled_dist = pickle.loads(pickle.dumps(self.dist))
        np.testing.assert_equal(unpickled_dist, self.dist)
        self.assertTrue(tables_equal(unpickled_dist.row_items, self.dist.row_items))
        self.assertTrue(tables_equal(unpickled_dist.col_items, self.dist.col_items))
        self.assertEqual(unpickled_dist.axis, self.dist.axis)

    def test_deprecated(self):
        a9 = np.arange(9).reshape(3, 3)
        m = DistMatrix(a9)
        with self.assertWarns(OrangeDeprecationWarning):
            self.assertEqual(m.dim, 3)
        with self.assertWarns(OrangeDeprecationWarning):
            np.testing.assert_almost_equal(m.X, a9)

    def test_from_file(self):
        with named_file(
            """3 axis=0 asymmetric col_labels row_labels
                ann	bert	chad
                danny	0.12	3.45	6.78
                eve	9.01	2.34	5.67
                frank	8.90	1.23	4.56""") as name:
            m = DistMatrix.from_file(name)
            np.testing.assert_almost_equal(m, np.array([[0.12, 3.45, 6.78],
                                                        [9.01, 2.34, 5.67],
                                                        [8.90, 1.23, 4.56]]))
            self.assertIsInstance(m.row_items, Table)
            self.assertIsInstance(m.col_items, Table)
            self.assertEqual([e.metas[0] for e in m.col_items],
                             ["ann", "bert", "chad"])
            self.assertEqual([e.metas[0] for e in m.row_items],
                             ["danny", "eve", "frank"])
            self.assertEqual(m.axis, 0)

        with named_file(
            """3 axis=1 row_labels
                danny	0.12	3.45	6.78
                eve 	9.01	2.34	5.67
                frank	8.90""") as name:
            m = DistMatrix.from_file(name)
            np.testing.assert_almost_equal(m, np.array([[0.12, 9.01, 8.90],
                                                        [9.01, 2.34, 0],
                                                        [8.90, 0, 0]]))
            self.assertIsInstance(m.row_items, Table)
            self.assertIsNone(m.col_items)
            self.assertEqual([e.metas[0] for e in m.row_items],
                             ["danny", "eve", "frank"])
            self.assertEqual(m.axis, 1)

        with named_file(
            """3 axis=1 symmetric
                0.12	3.45	6.78
                9.01	2.34	5.67
                8.90""") as name:
            m = DistMatrix.from_file(name)
        np.testing.assert_almost_equal(m, np.array([[0.12, 9.01, 8.90],
                                                    [9.01, 2.34, 0],
                                                    [8.90, 0, 0]]))

        with named_file(
            """3 row_labels
                starič	0.12	3.45	6.78
                aleš	9.01	2.34	5.67
                anže	8.90""", encoding="utf-8""") as name:
            m = DistMatrix.from_file(name)
            np.testing.assert_almost_equal(m, np.array([[0.12, 9.01, 8.90],
                                                        [9.01, 2.34, 0],
                                                        [8.90, 0, 0]]))
            self.assertIsInstance(m.row_items, Table)
            self.assertIsNone(m.col_items)
            self.assertEqual([e.metas[0] for e in m.row_items],
                             ["starič", "aleš", "anže"])
            self.assertEqual(m.axis, 1)

        def assertErrorMsg(content, msg):
            with named_file(content) as name:
                with self.assertRaises(ValueError) as cm:
                    DistMatrix.from_file(name)
                self.assertEqual(str(cm.exception), msg)

        assertErrorMsg("",
                       "empty file")
        assertErrorMsg("axis=1\n1\t3\n4",
                       "distance file must begin with dimension")
        assertErrorMsg("3 col_labels\na\tb\n1\n\2\n3",
                       "mismatching number of column labels")
        assertErrorMsg("3 col_labels\na\tb\tc\td\n1\n\2\n3",
                       "mismatching number of column labels")
        assertErrorMsg("2\n  1\t2\t3\n  5",
                       "too many columns in matrix row 1")
        assertErrorMsg("2 row_labels\na\t1\t2\t3\nb\t5",
                       "too many columns in matrix row 'a'")
        assertErrorMsg("2 noflag\n  1\t2\t3\n  5",
                       "invalid flag 'noflag'")
        assertErrorMsg("2 noflag=5\n  1\t2\t3\n  5",
                       "invalid flag 'noflag=5'")
        assertErrorMsg("2\n1\n2\n3",
                       "too many rows")
        assertErrorMsg("2\n1\nasd",
                       "invalid element at row 2, column 1")
        assertErrorMsg("2 row_labels\na\t1\nb\tasd",
                       "invalid element at row 'b', column 1")
        assertErrorMsg("2 col_labels row_labels\nd\te\na\t1\nb\tasd",
                       "invalid element at row 'b', column 'd'")
        assertErrorMsg("2 col_labels\nd\te\n1\nasd",
                       "invalid element at row 2, column 'd'")

    def test_save(self):
        with named_file(
            """3 axis=1 row_labels
                danny	0.12	3.45	6.78
                eve 	9.01	2.34	5.67
                frank	8.90""") as name:
            m = DistMatrix.from_file(name)
            m.save(name)
            m = DistMatrix.from_file(name)
            np.testing.assert_almost_equal(m, np.array([[0.12, 9.01, 8.90],
                                                        [9.01, 2.34, 0],
                                                        [8.90, 0, 0]]))
            self.assertIsInstance(m.row_items, Table)
            self.assertIsNone(m.col_items)
            self.assertEqual([e.metas[0] for e in m.row_items],
                             ["danny", "eve", "frank"])
            self.assertEqual(m.axis, 1)

        with named_file(
            """3 axis=0 asymmetric col_labels row_labels
                         ann	bert	chad
                danny	0.12	3.45	6.78
                  eve	9.01	2.34	5.67
                frank	8.90	1.23	4.56""") as name:
            m = DistMatrix.from_file(name)
            m.save(name)
            m = DistMatrix.from_file(name)
            np.testing.assert_almost_equal(m, np.array([[0.12, 3.45, 6.78],
                                                        [9.01, 2.34, 5.67],
                                                        [8.90, 1.23, 4.56]]))
            self.assertIsInstance(m.row_items, Table)
            self.assertIsInstance(m.col_items, Table)
            self.assertEqual([e.metas[0] for e in m.col_items],
                             ["ann", "bert", "chad"])
            self.assertEqual([e.metas[0] for e in m.row_items],
                             ["danny", "eve", "frank"])
            self.assertEqual(m.axis, 0)


class TestEuclidean(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        cls.dist = Euclidean

    def test_euclidean_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.53851648071346281]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=1), np.array([[0.53851648071346281]]))

    def test_euclidean_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[0., 0.53851648],
                                                 [0.53851648, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[0., 2.48394847, 5.09313263, 6.78969808],
                                                 [2.48394847, 0., 2.64007576, 4.327817],
                                                 [5.09313263, 2.64007576, 0., 1.69705627],
                                                 [6.78969808, 4.327817, 1.69705627, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np. array([[0.50990195, 0.3, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[0.64807407],
                                                 [0.33166248]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[0., 0.53851648, 0.50990195],
                                                 [0.53851648, 0., 0.3]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[0., 2.48394847, 5.09313263, 6.78969808],
                                                 [2.48394847, 0., 2.64007576, 4.327817],
                                                 [5.09313263, 2.64007576, 0., 1.69705627],
                                                 [6.78969808, 4.327817, 1.69705627, 0.]]))

    def test_euclidean_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[0., 3.74165739, 6.164414],
                                                 [3.74165739, 0., 4.47213595],
                                                 [6.164414, 4.47213595, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[0., 4.12310563, 3.31662479],
                                                 [4.12310563, 0., 6.164414],
                                                 [3.31662479, 6.164414, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[0., 3.74165739],
                                                 [3.74165739, 0.]]))

    def test_euclidean_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=1),
                                       np.array([[0.53851648071346281]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[0., 0.53851648],
                                                 [0.53851648, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np. array([[0.50990195, 0.3, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[0.64807407],
                                                 [0.33166248]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[0., 0.53851648, 0.50990195],
                                                 [0.53851648, 0., 0.3]]))


class TestManhattan(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        cls.dist = Manhattan

    def test_manhattan_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.7]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=1), np.array([[0.7]]))

    def test_manhattan_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[0., 0.7],
                                                 [0.7, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[0., 3.5, 7.2, 9.6],
                                                 [3.5, 0., 3.7, 6.1],
                                                 [7.2, 3.7, 0., 2.4],
                                                 [9.6, 6.1, 2.4, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np.array([[0.8, 0.5, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[1.],
                                                 [0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[0., 0.7, 0.8],
                                                 [0.7, 0., 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[0., 3.5, 7.2, 9.6],
                                                 [3.5, 0., 3.7, 6.1],
                                                 [7.2, 3.7, 0., 2.4],
                                                 [9.6, 6.1, 2.4, 0.]]))

    def test_manhattan_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[0., 6., 10.],
                                                 [6., 0., 6.],
                                                 [10., 6., 0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[0., 5., 5.],
                                                 [5., 0., 10.],
                                                 [5., 10., 0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[0., 6.],
                                                 [6., 0.]]))

    def test_manhattan_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=1), np.array([[0.7]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[0., 0.7],
                                                 [0.7, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np.array([[0.8, 0.5, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[1.],
                                                 [0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[0., 0.7, 0.8],
                                                 [0.7, 0., 0.5]]))


class TestCosine(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        cls.dist = Cosine

    def test_cosine_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.00142084]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=1), np.array([[0.00142084]]))

    def test_cosine_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[0., 1.42083650e-03],
                                                 [1.42083650e-03, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[0.0, 1.61124231e-03, 1.99940020e-04, 1.99940020e-04],
                                                 [1.61124231e-03, 0.0, 2.94551450e-03, 2.94551450e-03],
                                                 [1.99940020e-04, 2.94551450e-03, 0.0, 0.0],
                                                 [1.99940020e-04, 2.94551450e-03, 0.0, 0.0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np.array([[1.26527175e-05, 1.20854727e-03, 0.0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[0.00089939],
                                                 [0.00120607]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[0.0, 1.42083650e-03, 1.26527175e-05],
                                                 [1.42083650e-03, 0.0, 1.20854727e-03]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[0.0, 1.61124231e-03, 1.99940020e-04, 1.99940020e-04],
                                                 [1.61124231e-03, 0.0, 2.94551450e-03, 2.94551450e-03],
                                                 [1.99940020e-04, 2.94551450e-03, 0.0, 0.0],
                                                 [1.99940020e-04, 2.94551450e-03, 0.0, 0.0]]))

    def test_cosine_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[0.0, 1.00000000e+00, 7.20627882e-01],
                                                 [1.00000000e+00, 0.0, 2.19131191e-01],
                                                 [7.20627882e-01, 2.19131191e-01, 0.0]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[0.0, 7.57464375e-01, 1.68109669e-01],
                                                 [7.57464375e-01, 0.0, 1.00000000e+00],
                                                 [1.68109669e-01, 1.00000000e+00, 0.0]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[0.0, 1.00000000e+00],
                                                 [1.00000000e+00, 0.0]]))

    def test_cosine_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=1), np.array([[0.00142084]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[0., 1.42083650e-03],
                                                 [1.42083650e-03, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np.array([[1.26527175e-05, 1.20854727e-03, 0.0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[0.00089939],
                                                 [0.00120607]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[0.0, 1.42083650e-03, 1.26527175e-05],
                                                 [1.42083650e-03, 0.0, 1.20854727e-03]]))


class TestJaccard(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.titanic = Table('titanic')[173:177]
        cls.dist = Jaccard

    def test_jaccard_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.titanic[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[2]), np.array([[0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[2], axis=1), np.array([[0.5]]))

    def test_jaccard_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.titanic),
                                       np.array([[0., 0., 0.5, 0.5],
                                                 [0., 0., 0.5, 0.5],
                                                 [0.5, 0.5, 0., 0.],
                                                 [0.5, 0.5, 0., 0.]]))
        np.testing.assert_almost_equal(self.dist(self.titanic, axis=0),
                                       np.array([[0., 1., 0.5],
                                                 [1., 0., 1.],
                                                 [0.5, 1., 0.]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[2], self.titanic[:3]),
                                       np.array([[0.5, 0.5, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2], self.titanic[3]),
                                       np.array([[0.5],
                                                 [0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2], self.titanic[:3]),
                                       np.array([[0., 0., 0.5],
                                                 [0., 0., 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic, self.titanic, axis=0),
                                       np.array([[0., 1., 0.5],
                                                 [1., np.nan, 1.],
                                                 [0.5, 1., 0.]]))

    def test_jaccard_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.titanic[0].x, self.titanic[2].x, axis=1), np.array([[0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic.X),
                                       np.array([[0., 0., 0.5, 0.5],
                                                 [0., 0., 0.5, 0.5],
                                                 [0.5, 0.5, 0., 0.],
                                                 [0.5, 0.5, 0., 0.]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[2].x, self.titanic[:3].X),
                                       np.array([[0.5, 0.5, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2].X, self.titanic[3].x),
                                       np.array([[0.5],
                                                 [0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2].X, self.titanic[:3].X),
                                       np.array([[0., 0., 0.5],
                                                 [0., 0., 0.5]]))


class TestSpearmanR(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.breast = Table("breast-cancer-wisconsin-cont")
        cls.dist = SpearmanR

    def test_spearmanr_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.5083333333333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=1),
                                       np.array([[0.5083333333333333]]))

    def test_spearmanr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[0., 0.5083333333333333],
                                                 [0.5083333333333333, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4]),
                                       np.array([[0., 0.50833333, 0.075, 0.61666667],
                                                 [0.50833333, 0., 0.38333333, 0.53333333],
                                                 [0.075, 0.38333333, 0., 0.63333333],
                                                 [0.61666667, 0.53333333, 0.63333333, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.75, 0.25, 0.75, 0.25, 0.25, 0.25, 0., 0.25, 1.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0.75, 0.25, 0.75, 0.75, 0.75, 1., 0.75, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[0., 0.50833333, 0.075, 0.61666667],
                                                 [0.50833333, 0., 0.38333333, 0.53333333],
                                                 [0.075, 0.38333333, 0., 0.63333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np. array([[0.075, 0.3833333, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[2]),
                                       np. array([[0.075],
                                                  [0.3833333],
                                                  [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:3], axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.75, 0.25, 0.75, 0.25, 0.25, 0.25, 0., 0.25, 1.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0.75, 0.25, 0.75, 0.75, 0.75, 1., 0.75, 0.]]))

    def test_spearmanr_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=1),
                                       np.array([[0.5083333333333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[0., 0.5083333333333333],
                                                 [0.5083333333333333, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np. array([[0.075, 0.3833333, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[2].x),
                                       np. array([[0.075],
                                                  [0.3833333],
                                                  [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[:3].X, axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.75, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.75, 0.25, 0.75, 0.25, 0.25, 0.25, 0., 0.25, 1.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.75],
                                                 [0.25, 0.75, 0.25, 0.75, 0.75, 0.75, 1., 0.75, 0.]]))


class TestSpearmanRAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.breast = Table("breast-cancer-wisconsin-cont")
        cls.dist = SpearmanRAbsolute

    def test_spearmanrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]),
                                       np.array([[0.49166666666666664]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=1),
                                       np.array([[0.49166666666666664]]))

    def test_spearmanrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[0., 0.49166667],
                                                 [0.49166667, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[0., 0.49166667, 0.075, 0.38333333],
                                                 [0.49166667, 0., 0.38333333, 0.46666667],
                                                 [0.075, 0.38333333, 0., 0.36666667]]))
        np.testing.assert_almost_equal(self.dist(self.breast[3], self.breast[:4]),
                                       np.array([[0.3833333, 0.4666667, 0.3666667, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4], self.breast[3]),
                                       np.array([[0.3833333],
                                                 [0.4666667],
                                                 [0.3666667],
                                                 [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:3], axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.]]))

    def test_spearmanrabsolute_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=1),
                                       np.array([[0.49166666666666664]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[0., 0.49166667],
                                                 [0.49166667, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[3].x, self.breast[:4].X),
                                       np.array([[0.3833333, 0.4666667, 0.3666667, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4].X, self.breast[3].x),
                                       np.array([[0.3833333],
                                                 [0.4666667],
                                                 [0.3666667],
                                                 [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[:3].X, axis=0),
                                       np.array([[0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0., 0.25, 0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.],
                                                 [0.25, 0., 0.25, 0., 0., 0., 0.25, 0., 0.25],
                                                 [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0., 0.25, 0.]]))


class TestPearsonR(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.breast = Table("breast-cancer-wisconsin-cont")
        cls.dist = PearsonR

    def test_pearsonr_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=1),
                                       np.array([[0.48462293898088876]]))

    def test_pearsonr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[0., 0.48462294],
                                                 [0.48462294, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.57976119],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.57976119, 0.45930368, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[0., 0.48462294, 0.10133593, 0.5016744],
                                                 [0.48462294, 0., 0.32783865, 0.57317387],
                                                 [0.10133593, 0.32783865, 0., 0.63789635]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np.array([[0.10133593, 0.32783865, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[2]),
                                       np.array([[0.10133593],
                                                 [0.32783865],
                                                 [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], self.breast[:20], axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.57976119],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.57976119, 0.45930368, 0.]]))

    def test_pearsonr_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=1),
                                       np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[0., 0.48462294],
                                                 [0.48462294, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np.array([[0.10133593, 0.32783865, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[2].x),
                                       np.array([[0.10133593],
                                                 [0.32783865],
                                                 [0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20].X, self.breast[:20].X, axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.57976119],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.57976119, 0.45930368, 0.]]))


class TestPearsonRAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.breast = Table("breast-cancer-wisconsin-cont")
        cls.dist = PearsonRAbsolute

    def test_pearsonrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=1), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=1),
                                       np.array([[0.48462293898088876]]))

    def test_pearsonrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[0., 0.48462294],
                                                 [0.48462294, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.42023881],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.42023881, 0.45930368, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[0., 0.48462294, 0.10133593, 0.4983256],
                                                 [0.48462294, 0., 0.32783865, 0.42682613],
                                                 [0.10133593, 0.32783865, 0., 0.36210365]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np.array([[0.10133593, 0.32783865, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2], self.breast[3]),
                                       np.array([[0.4983256],
                                                 [0.42682613]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], self.breast[:20], axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.42023881],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.42023881, 0.45930368, 0.]]))

    def test_pearsonrabsolute_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=1),
                                       np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[0., 0.48462294],
                                                 [0.48462294, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np.array([[0.10133593, 0.32783865, 0.]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X, self.breast[3].x),
                                       np.array([[0.4983256],
                                                 [0.42682613]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20].X, self.breast[:20].X, axis=0),
                                       np.array([[0., 0.10239274, 0.12786763, 0.13435117, 0.15580385, 0.27429811, 0.21006195, 0.24072005, 0.42847752],
                                                 [0.10239274, 0., 0.01695375, 0.10313851, 0.1138925, 0.16978203, 0.1155948, 0.08043531, 0.43326547],
                                                 [0.12786763, 0.01695375, 0., 0.16049178, 0.13692762, 0.21784201, 0.11607395, 0.06493949, 0.46590168],
                                                 [0.13435117, 0.10313851, 0.16049178, 0., 0.07181648, 0.15585667, 0.13891172, 0.21622332, 0.37404826],
                                                 [0.15580385, 0.1138925, 0.13692762, 0.07181648, 0., 0.16301705, 0.17324382, 0.21452448, 0.42283252],
                                                 [0.27429811, 0.16978203, 0.21784201, 0.15585667, 0.16301705, 0., 0.25512861, 0.29560909, 0.42766076],
                                                 [0.21006195, 0.1155948, 0.11607395, 0.13891172, 0.17324382, 0.25512861, 0., 0.14419442, 0.42023881],
                                                 [0.24072005, 0.08043531, 0.06493949, 0.21622332, 0.21452448, 0.29560909, 0.14419442, 0., 0.45930368],
                                                 [0.42847752, 0.43326547, 0.46590168, 0.37404826, 0.42283252, 0.42766076, 0.42023881, 0.45930368, 0.]]))


class TestMahalanobis(TestCase):
    def setUp(self):
        self.n, self.m = 10, 5
        self.x = np.random.rand(self.n, self.m)
        self.x1 = np.random.rand(self.m)
        self.x2 = np.random.rand(self.m)

    def test_correctness(self):
        mah = MahalanobisDistance(self.x)
        d = scipy.spatial.distance.pdist(self.x, 'mahalanobis')
        d = scipy.spatial.distance.squareform(d)
        for i in range(self.n):
            for j in range(self.n):
                self.assertAlmostEqual(d[i][j], mah(self.x[i], self.x[j]), delta=1e-5)

    def test_attributes(self):
        metric = MahalanobisDistance(self.x)
        self.assertEqual(metric(self.x[0], self.x[1]).shape, (1, 1))
        self.assertEqual(metric(self.x).shape, (self.n, self.n))
        self.assertEqual(metric(self.x[0:3], self.x[5:7]).shape, (3, 2))
        self.assertEqual(metric(self.x1, self.x2).shape, (1, 1))
        metric(self.x, impute=True)
        metric(self.x[:-1, :])
        self.assertRaises(ValueError, metric, self.x[:, :-1])
        self.assertRaises(ValueError, metric, self.x1[:-1], self.x2)
        self.assertRaises(ValueError, metric, self.x1, self.x2[:-1])
        self.assertRaises(ValueError, metric, self.x.T)

    def test_iris(self):
        tab = Table('iris')
        metric = MahalanobisDistance(tab)
        self.assertEqual(metric(tab).shape, (150, 150))
        self.assertEqual(metric(tab[0], tab[1]).shape, (1, 1))

    def test_axis(self):
        mah = MahalanobisDistance(self.x, axis=1)
        self.assertEqual(mah(self.x, self.x).shape, (self.n, self.n))
        x = self.x.T
        mah = MahalanobisDistance(x, axis=0)
        self.assertRaises(AssertionError, mah, x, axis=1)
        self.assertEqual(mah(x, x).shape, (self.n, self.n))

    def test_dimensions(self):
        x = Table('iris')[:20].X
        xt = Table('iris')[:20].X.T
        mah = MahalanobisDistance(x)
        mah(x[0], x[1])
        mah = MahalanobisDistance(xt)
        mah(xt[0], xt[1])

    def test_global_is_borked(self):
        """
        Test that the global state retaining non-safe Mahalanobis instance
        raises RuntimeErrors on all invocations
        """
        from Orange.distance import Mahalanobis
        with self.assertRaises(RuntimeError):
            Mahalanobis.fit(self.x)
        with self.assertRaises(RuntimeError):
            Mahalanobis(self.x)


class TestDistances(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test5 = Table(test_filename('test5.tab'))

    def test_preprocess(self):
        domain = Domain([ContinuousVariable("c"),
                         DiscreteVariable("d", values=['a', 'b'])],
                        [DiscreteVariable("cls", values=['e', 'f'])],
                        [StringVariable("m")])
        table = Table(domain, [[1, 'a', 'e', 'm1'],
                               [2, 'b', 'f', 'm2']])
        new_table = _preprocess(table)
        np.testing.assert_equal(new_table.X, table.X[:, 0].reshape(2, 1))
        np.testing.assert_equal(new_table.Y, table.Y)
        np.testing.assert_equal(new_table.metas, table.metas)
        self.assertEqual([a.name for a in new_table.domain.attributes],
                         [a.name for a in table.domain.attributes
                          if a.is_continuous])
        self.assertEqual(new_table.domain.class_vars, table.domain.class_vars)
        self.assertEqual(new_table.domain.metas, table.domain.metas)

    def test_preprocess_multiclass(self):
        table = self.test5
        new_table = _preprocess(table)
        np.testing.assert_equal(new_table.Y, table.Y)
        self.assertEqual([a.name for a in new_table.domain.attributes],
                         [a.name for a in table.domain.attributes
                          if a.is_continuous])
        self.assertEqual(new_table.domain.class_vars, table.domain.class_vars)

    def test_preprocess_impute(self):
        new_table = _preprocess(self.test5)
        self.assertFalse(np.isnan(new_table.X).any())

    def test_distance_to_instance(self):
        iris = Table('iris')
        inst = Instance(iris.domain, np.concatenate((iris[1].x, iris[1].y)))
        self.assertEqual(Euclidean(iris[1], inst), 0)
