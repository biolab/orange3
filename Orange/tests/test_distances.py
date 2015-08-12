from unittest import TestCase
import pickle

import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import (Table, Domain, ContinuousVariable,
                         DiscreteVariable, StringVariable)
from Orange.distance import (Euclidean, SpearmanR, SpearmanRAbsolute,
                             PearsonR, PearsonRAbsolute, Manhattan, Cosine,
                             Jaccard, _preprocess)


def tables_equal(tab1, tab2):
    # TODO: introduce Table.__eq__() ???
    return (tab1 == tab2 or  # catches None
            np.all([i == j for i, j in zip(tab1, tab2)]))


class TestDistMatrix(TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.dist = Euclidean(self.iris)

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


class TestEuclidean(TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        self.dist = Euclidean

    def test_euclidean_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.53851648071346281]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=0), np.array([[0.53851648071346281]]))

    def test_euclidean_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[ 0.        ,  0.53851648],
                                                 [ 0.53851648,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[ 0.        ,  2.48394847,  5.09313263,  6.78969808],
                                                 [ 2.48394847,  0.        ,  2.64007576,  4.327817  ],
                                                 [ 5.09313263,  2.64007576,  0.        ,  1.69705627],
                                                 [ 6.78969808,  4.327817  ,  1.69705627,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np. array([[ 0.50990195,  0.3       ,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[ 0.64807407],
                                                 [ 0.33166248]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[ 0.        ,  0.53851648,  0.50990195],
                                                 [ 0.53851648,  0.        ,  0.3       ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[ 0.        ,  2.48394847,  5.09313263,  6.78969808],
                                                 [ 2.48394847,  0.        ,  2.64007576,  4.327817  ],
                                                 [ 5.09313263,  2.64007576,  0.        ,  1.69705627],
                                                 [ 6.78969808,  4.327817  ,  1.69705627,  0.        ]]))

    def test_euclidean_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[ 0.        ,  3.74165739,  6.164414  ],
                                                 [ 3.74165739,  0.        ,  4.47213595],
                                                 [ 6.164414  ,  4.47213595,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[ 0.        ,  4.12310563,  3.31662479],
                                                 [ 4.12310563,  0.        ,  6.164414  ],
                                                 [ 3.31662479,  6.164414  ,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[ 0.        ,  3.74165739],
                                                 [ 3.74165739,  0.        ]]))

    def test_euclidean_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=0), np.array([[0.53851648071346281]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[ 0.        ,  0.53851648],
                                                 [ 0.53851648,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np. array([[ 0.50990195,  0.3       ,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[ 0.64807407],
                                                 [ 0.33166248]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[ 0.        ,  0.53851648,  0.50990195],
                                                 [ 0.53851648,  0.        ,  0.3       ]]))


class TestManhattan(TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        self.dist = Manhattan

    def test_manhattan_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.7]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=0), np.array([[0.7]]))

    def test_manhattan_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[ 0. ,  0.7],
                                                 [ 0.7,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[ 0. ,  3.5,  7.2,  9.6],
                                                 [ 3.5,  0. ,  3.7,  6.1],
                                                 [ 7.2,  3.7,  0. ,  2.4],
                                                 [ 9.6,  6.1,  2.4,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np.array([[ 0.8,  0.5,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[ 1. ],
                                                 [ 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[ 0. ,  0.7,  0.8],
                                                 [ 0.7,  0. ,  0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[ 0. ,  3.5,  7.2,  9.6],
                                                 [ 3.5,  0. ,  3.7,  6.1],
                                                 [ 7.2,  3.7,  0. ,  2.4],
                                                 [ 9.6,  6.1,  2.4,  0. ]]))

    def test_manhattan_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[  0.,   6.,  10.],
                                                 [  6.,   0.,   6.],
                                                 [ 10.,   6.,   0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[  0.,   5.,   5.],
                                                 [  5.,   0.,  10.],
                                                 [  5.,  10.,   0.]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[  0.,   6.],
                                                 [  6.,   0.]]))

    def test_manhattan_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=0), np.array([[0.7]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[ 0. ,  0.7],
                                                 [ 0.7,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np.array([[ 0.8,  0.5,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[ 1. ],
                                                 [ 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[ 0. ,  0.7,  0.8],
                                                 [ 0.7,  0. ,  0.5]]))


class TestCosine(TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.sparse = Table(csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]]))
        self.dist = Cosine

    def test_cosine_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1]), np.array([[0.00142084]]))
        np.testing.assert_almost_equal(self.dist(self.iris[0], self.iris[1], axis=0), np.array([[0.00142084]]))

    def test_cosine_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.iris[:2]),
                                       np.array([[  0.            ,   1.42083650e-03],
                                                 [  1.42083650e-03,   0.             ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], axis=0),
                                       np.array([[  0.0           ,   1.61124231e-03,   1.99940020e-04, 1.99940020e-04],
                                                 [  1.61124231e-03,   0.0           ,   2.94551450e-03, 2.94551450e-03],
                                                 [  1.99940020e-04,   2.94551450e-03,   0.0           , 0.0           ],
                                                 [  1.99940020e-04,   2.94551450e-03,   0.0           , 0.0           ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2], self.iris[:3]),
                                       np.array([[  1.26527175e-05,   1.20854727e-03,   0.0         ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[3]),
                                       np.array([[ 0.00089939],
                                                 [ 0.00120607]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:3]),
                                       np.array([[  0.0           ,   1.42083650e-03,   1.26527175e-05],
                                                 [  1.42083650e-03,   0.0           ,   1.20854727e-03]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2], self.iris[:2], axis=0),
                                       np.array([[  0.0           ,   1.61124231e-03,   1.99940020e-04, 1.99940020e-04],
                                                 [  1.61124231e-03,   0.0           ,   2.94551450e-03, 2.94551450e-03],
                                                 [  1.99940020e-04,   2.94551450e-03,   0.0           , 0.0           ],
                                                 [  1.99940020e-04,   2.94551450e-03,   0.0           , 0.0           ]]))

    def test_cosine_distance_sparse(self):
        np.testing.assert_almost_equal(self.dist(self.sparse),
                                       np.array([[  0.0           ,   1.00000000e+00,   7.20627882e-01],
                                                 [  1.00000000e+00,   0.0           ,   2.19131191e-01],
                                                 [  7.20627882e-01,   2.19131191e-01,   0.0           ]]))
        np.testing.assert_almost_equal(self.dist(self.sparse, axis=0),
                                       np.array([[  0.0           ,   7.57464375e-01,   1.68109669e-01],
                                                 [  7.57464375e-01,   0.0           ,   1.00000000e+00],
                                                 [  1.68109669e-01,   1.00000000e+00,   0.0           ]]))
        np.testing.assert_almost_equal(self.dist(self.sparse[:2]),
                                       np.array([[  0.0           ,   1.00000000e+00],
                                                 [  1.00000000e+00,   0.0           ]]))

    def test_cosine_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.iris[0].x, self.iris[1].x, axis=0), np.array([[0.00142084]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X),
                                       np.array([[  0.            ,   1.42083650e-03],
                                                 [  1.42083650e-03,   0.             ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[2].x, self.iris[:3].X),
                                       np.array([[  1.26527175e-05,   1.20854727e-03,   0.0         ]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[3].x),
                                       np.array([[ 0.00089939],
                                                 [ 0.00120607]]))
        np.testing.assert_almost_equal(self.dist(self.iris[:2].X, self.iris[:3].X),
                                       np.array([[  0.0           ,   1.42083650e-03,   1.26527175e-05],
                                                 [  1.42083650e-03,   0.0           ,   1.20854727e-03]]))


class TestJaccard(TestCase):
    def setUp(self):
        self.titanic = Table('titanic')[173:177]
        self.dist = Jaccard

    def test_jaccard_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.titanic[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[2]),  np.array([[0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[0], self.titanic[2], axis=0),  np.array([[0.5]]))

    def test_jaccard_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.titanic),
                                       np.array([[ 0. ,  0. ,  0.5,  0.5],
                                                 [ 0. ,  0. ,  0.5,  0.5],
                                                 [ 0.5,  0.5,  0. ,  0. ],
                                                 [ 0.5,  0.5,  0. ,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.titanic, axis=0),
                                       np.array([[ 0. ,  1. ,  0.5],
                                                 [ 1. ,  0. ,  1. ],
                                                 [ 0.5,  1. ,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[2], self.titanic[:3]),
                                       np.array([[ 0.5,  0.5,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2], self.titanic[3]),
                                       np.array([[ 0.5],
                                                 [ 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2], self.titanic[:3]),
                                       np.array([[ 0. ,  0. ,  0.5],
                                                 [ 0. ,  0. ,  0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic, self.titanic, axis=0),
                                       np.array([[ 0. ,  1. ,  0.5],
                                                 [ 1. ,  np.nan ,  1. ],
                                                 [ 0.5,  1. ,  0. ]]))

    def test_jaccard_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.titanic[0].x, self.titanic[2].x, axis=0),  np.array([[0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic.X),
                                       np.array([[ 0. ,  0. ,  0.5,  0.5],
                                                 [ 0. ,  0. ,  0.5,  0.5],
                                                 [ 0.5,  0.5,  0. ,  0. ],
                                                 [ 0.5,  0.5,  0. ,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[2].x, self.titanic[:3].X),
                                       np.array([[ 0.5,  0.5,  0. ]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2].X, self.titanic[3].x),
                                       np.array([[ 0.5],
                                                 [ 0.5]]))
        np.testing.assert_almost_equal(self.dist(self.titanic[:2].X, self.titanic[:3].X),
                                       np.array([[ 0. ,  0. ,  0.5],
                                                 [ 0. ,  0. ,  0.5]]))


class TestSpearmanR(TestCase):
    def setUp(self):
        self.breast = Table("breast-cancer-wisconsin-cont")
        self.dist = SpearmanR

    def test_spearmanr_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.5083333333333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=0), np.array([[0.5083333333333333]]))

    def test_spearmanr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[ 0.                ,  0.5083333333333333],
                                                 [ 0.5083333333333333,  0.                ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4]),
                                       np.array([[ 0.        ,  0.50833333,  0.075     ,  0.61666667],
                                                 [ 0.50833333,  0.        ,  0.38333333,  0.53333333],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.63333333],
                                                 [ 0.61666667,  0.53333333,  0.63333333,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.75,  0.25,  0.75,  0.25,  0.25,  0.25,  0.  ,  0.25,  1.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.75,  0.25,  0.75,  0.75,  0.75,  1.  ,  0.75,  0.  ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[ 0.        ,  0.50833333,  0.075     ,  0.61666667],
                                                 [ 0.50833333,  0.        ,  0.38333333,  0.53333333],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.63333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np. array([[ 0.56282809, 0.65526475, 0.3288367 ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[2]),
                                       np. array([[ 0.56282809 ],
                                                  [ 0.65526475 ],
                                                  [ 0.3288367  ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:3], axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.75,  0.25,  0.75,  0.25,  0.25,  0.25,  0.  ,  0.25,  1.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.75,  0.25,  0.75,  0.75,  0.75,  1.  ,  0.75,  0.  ]]))

    def test_spearmanr_distacne_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=0), np.array([[0.5083333333333333]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[ 0.                ,  0.5083333333333333],
                                                 [ 0.5083333333333333,  0.                ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np. array([[ 0.56282809, 0.65526475, 0.3288367 ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[2].x),
                                       np. array([[ 0.56282809 ],
                                                  [ 0.65526475 ],
                                                  [ 0.3288367  ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[:3].X, axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.75,  0.25,  0.75,  0.25,  0.25,  0.25,  0.  ,  0.25,  1.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.75,  0.25,  0.75,  0.75,  0.75,  1.  ,  0.75,  0.  ]]))


class TestSpearmanRAbsolute(TestCase):
    def setUp(self):
        self.breast = Table("breast-cancer-wisconsin-cont")
        self.dist = SpearmanRAbsolute

    def test_spearmanrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.49166666666666664]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=0), np.array([[0.49166666666666664]]))

    def test_spearmanrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[ 0.        ,  0.49166667],
                                                 [ 0.49166667,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[ 0.        ,  0.49166667,  0.075     ,  0.38333333],
                                                 [ 0.49166667,  0.        ,  0.38333333,  0.46666667],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.36666667]]))
        np.testing.assert_almost_equal(self.dist(self.breast[3], self.breast[:4]),
                                       np.array([[ 0.40995497,  0.3288367 ,  0.29564403,  0.07836298]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4], self.breast[3]),
                                       np.array([[ 0.40995497 ],
                                                 [ 0.3288367  ],
                                                 [ 0.29564403 ],
                                                 [ 0.07836298 ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:3], axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ]]))

    def test_spearmanrabsolute_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=0), np.array([[0.49166666666666664]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[ 0.        ,  0.49166667],
                                                 [ 0.49166667,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[3].x, self.breast[:4].X),
                                       np.array([[ 0.40995497,  0.3288367 ,  0.29564403,  0.07836298]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:4].X, self.breast[3].x),
                                       np.array([[ 0.40995497 ],
                                                 [ 0.3288367  ],
                                                 [ 0.29564403 ],
                                                 [ 0.07836298 ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[:3].X, axis=0),
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ]]))


class TestPearsonR(TestCase):
    def setUp(self):
        self.breast = Table("breast-cancer-wisconsin-cont")
        self.dist = PearsonR

    def test_pearsonr_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=0), np.array([[0.48462293898088876]]))

    def test_pearsonr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.57976119],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.57976119,  0.45930368,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[ 0.        ,  0.48462294,  0.10133593,  0.5016744 ],
                                                 [ 0.48462294,  0.        ,  0.32783865,  0.57317387],
                                                 [ 0.10133593,  0.32783865,  0.        ,  0.63789635]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[2]),
                                       np.array([[ 0.10133593 ],
                                                 [0.32783865  ],
                                                 [0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], self.breast[:20], axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.57976119],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.57976119,  0.45930368,  0.        ]]))

    def test_pearsonr_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=0), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3].X, self.breast[2].x),
                                       np.array([[ 0.10133593 ],
                                                 [0.32783865  ],
                                                 [0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20].X, self.breast[:20].X, axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.57976119],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.57976119,  0.45930368,  0.        ]]))


class TestPearsonRAbsolute(TestCase):
    def setUp(self):
        self.breast = Table("breast-cancer-wisconsin-cont")
        self.dist = PearsonRAbsolute

    def test_pearsonrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0]), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[0], axis=0), np.array([[0]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1]), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[0], self.breast[1], axis=0), np.array([[0.48462293898088876]]))

    def test_pearsonrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.dist(self.breast[:2]),
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.42023881],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.42023881,  0.45930368,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:3], self.breast[:4]),
                                       np.array([[ 0.        ,  0.48462294,  0.10133593,  0.4983256 ],
                                                 [ 0.48462294,  0.        ,  0.32783865,  0.42682613],
                                                 [ 0.10133593,  0.32783865,  0.        , 0.36210365]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2], self.breast[:3]),
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2], self.breast[3]),
                                       np.array([[ 0.4983256 ],
                                                 [ 0.42682613]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20], self.breast[:20], axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.42023881],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.42023881,  0.45930368,  0.        ]]))

    def test_pearsonrabsolute_distance_numpy(self):
        np.testing.assert_almost_equal(self.dist(self.breast[0].x, self.breast[1].x, axis=0), np.array([[0.48462293898088876]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X),
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[2].x, self.breast[:3].X),
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:2].X, self.breast[3].x),
                                       np.array([[ 0.4983256 ],
                                                 [ 0.42682613]]))
        np.testing.assert_almost_equal(self.dist(self.breast[:20].X, self.breast[:20].X, axis=0),
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.42023881],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.42023881,  0.45930368,  0.        ]]))


class TestDistances(TestCase):
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
        table = Table('test5.tab')
        new_table = _preprocess(table)
        np.testing.assert_equal(new_table.Y, table.Y)
        self.assertEqual([a.name for a in new_table.domain.attributes],
                         [a.name for a in table.domain.attributes
                          if a.is_continuous])
        self.assertEqual(new_table.domain.class_vars, table.domain.class_vars)

    def test_preprocess_impute(self):
        table = Table('test5.tab')
        new_table = _preprocess(table)
        self.assertFalse(np.isnan(new_table.X).any())
