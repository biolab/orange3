from unittest import TestCase
from Orange.data import Table
from Orange.distance import Euclidean, Mahalanobis, SpearmanR, SpearmanRAbsolute, PearsonR, PearsonRAbsolute
from scipy.sparse import csr_matrix
import numpy as np


class TestEuclidean(TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.random_sparse = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        self.ed = Euclidean()

    def test_euclidean_distance_one_example(self):
        np.testing.assert_almost_equal(self.ed(self.iris[0]), 0)
        np.testing.assert_almost_equal(self.ed(self.iris[0], self.iris[0]), 0)
        np.testing.assert_almost_equal(self.ed(self.iris[0], axis=0), 0)
        np.testing.assert_almost_equal(self.ed(self.iris[0], self.iris[0], axis=0), 0)
        np.testing.assert_almost_equal(self.ed(self.iris[0], self.iris[1]), 0.53851648071346281)
        np.testing.assert_almost_equal(self.ed(self.iris[0], self.iris[1], axis=0), 0.53851648071346281)

    def test_euclidean_distance_many_examples(self):
        np.testing.assert_almost_equal(self.ed(self.iris[:2]).X,
                                       np.array([[ 0.        ,  0.53851648],
                                                 [ 0.53851648,  0.        ]]))
        np.testing.assert_almost_equal(self.ed(self.iris[:2], axis=0).X,
                                       np.array([[ 0.        ,  2.48394847,  5.09313263,  6.78969808],
                                                 [ 2.48394847,  0.        ,  2.64007576,  4.327817  ],
                                                 [ 5.09313263,  2.64007576,  0.        ,  1.69705627],
                                                 [ 6.78969808,  4.327817  ,  1.69705627,  0.        ]]))
        np.testing.assert_almost_equal(self.ed(self.iris[2], self.iris[:3]).X,
                                       np. array([[ 0.50990195,  0.3       ,  0.        ]]))
        np.testing.assert_almost_equal(self.ed(self.iris[:2], self.iris[3]).X,
                                       np.array([[ 0.64807407],
                                                 [ 0.33166248]]))
        np.testing.assert_almost_equal(self.ed(self.iris[:2], self.iris[:3]).X,
                                       np.array([[ 0.        ,  0.53851648,  0.50990195],
                                                 [ 0.53851648,  0.        ,  0.3       ]]))
        np.testing.assert_almost_equal(self.ed(self.iris[:2], self.iris[:2], axis=0).X,
                                       np.array([[ 0.        ,  2.48394847,  5.09313263,  6.78969808],
                                                 [ 2.48394847,  0.        ,  2.64007576,  4.327817  ],
                                                 [ 5.09313263,  2.64007576,  0.        ,  1.69705627],
                                                 [ 6.78969808,  4.327817  ,  1.69705627,  0.        ]]))

    def test_euclidean_distance_sparse(self):
        np.testing.assert_almost_equal(self.ed(self.random_sparse).X,
                                       np.array([[ 0.        ,  3.74165739,  6.164414  ],
                                                 [ 3.74165739,  0.        ,  4.47213595],
                                                 [ 6.164414  ,  4.47213595,  0.        ]]))


class TestMahalanobis(TestCase):
    def setUp(self):
        self.random_array = Table(np.array([[ 0.56534695,  0.89940663,  0.00703735,  0.97163439],
                                            [ 0.31496845,  0.40197171,  0.59052145,  0.99071001],
                                            [ 0.2104845 ,  0.85329328,  0.66341486,  0.25772055]]))
        self.hilbert = Table(np.array([[ 1.        ,  0.5       ,  0.33333333],
                                       [ 0.5       ,  0.33333333,  0.25      ],
                                       [ 0.33333333,  0.25      ,  0.2       ]]))
        self.hilbert_cov_inv = np.linalg.inv(np.cov(self.hilbert))
        self.ma = Mahalanobis()

    def test_mahalanobis_distance_many_examples(self):
        np.testing.assert_almost_equal(self.ma(self.hilbert).X,
                                       np.array([[ 0.        ,  2.82842712,  1.78885431],
                                                 [ 2.82842712,  0.        ,  2.11344888],
                                                 [ 1.78885431,  2.11344888,  0.        ]]))
        np.testing.assert_almost_equal(self.ma(self.random_array, axis=0).X,
                                       np.array([[ 0.        ,  2.44948974,  2.44948974,  2.44948974],
                                                 [ 2.44948974,  0.        ,  2.44948974,  2.44948974],
                                                 [ 2.44948974,  2.44948974,  0.        ,  2.44948974],
                                                 [ 2.44948974,  2.44948974,  2.44948974,  0.        ]]))
        """
        np.testing.assert_almost_equal(self.ma(self.hilbert, self.hilbert).X,
                                       np.array([[ 0.        ,  2.29128807,  1.54919324],
                                                 [ 2.29128807,  0.        ,  2.2416512 ],
                                                 [ 1.54919324,  2.2416512 ,  0.        ]]))
        """
        np.testing.assert_almost_equal(self.ma(self.random_array, self.random_array, axis=0).X,
                                       np.array([[ 0.        ,  2.64575131,  2.64575131,  2.64575131],
                                                 [ 2.64575131,  0.        ,  2.64575131,  2.64575131],
                                                 [ 2.64575131,  2.64575131,  0.        ,  2.64575131],
                                                 [ 2.64575131,  2.64575131,  2.64575131,  0.        ]]))

    def test_mahalanobis_distance_inverse_given(self):
        np.testing.assert_almost_equal(self.ma(self.hilbert, self.hilbert, VI=self.hilbert_cov_inv).X,
                                       np.array([[ 0.        ,  2.58198888,      np.nan],
                                                 [ 2.58198888,  0.        ,  2.26568596],
                                                 [     np.nan,  2.26568596,  0.        ]]))


class TestSpearmanR(TestCase):
    def setUp(self):
        self.breast = Table("breast-cancer-wisconsin-cont")
        self.sp = SpearmanR()

    def test_spearmanr_distance_one_example(self):
        np.testing.assert_almost_equal(self.sp(self.breast[0]), 0)
        np.testing.assert_almost_equal(self.sp(self.breast[0], self.breast[0]), 0)
        np.testing.assert_almost_equal(self.sp(self.breast[0], self.breast[1]), 0.5083333333333333)

    def test_spearmanr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.sp(self.breast[:2]).X,
                                       np.array([[ 0.                ,  0.5083333333333333],
                                                 [ 0.5083333333333333,  0.                ]]))
        np.testing.assert_almost_equal(self.sp(self.breast[:4]).X,
                                       np.array([[ 0.        ,  0.50833333,  0.075     ,  0.61666667],
                                                 [ 0.50833333,  0.        ,  0.38333333,  0.53333333],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.63333333],
                                                 [ 0.61666667,  0.53333333,  0.63333333,  0.        ]]))
        np.testing.assert_almost_equal(self.sp(self.breast[:3], axis=0).X,
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.75,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.75,  0.25,  0.75,  0.25,  0.25,  0.25,  0.  ,  0.25,  1.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.75],
                                                 [ 0.25,  0.75,  0.25,  0.75,  0.75,  0.75,  1.  ,  0.75,  0.  ]]))
        np.testing.assert_almost_equal(self.sp(self.breast[:3], self.breast[:4]).X,
                                       np.array([[ 0.        ,  0.50833333,  0.075     ,  0.61666667],
                                                 [ 0.50833333,  0.        ,  0.38333333,  0.53333333],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.63333333]]))
        np.testing.assert_almost_equal(self.sp(self.breast[2], self.breast[:3]).X,
                                       np. array([[ 0.56282809, 0.65526475, 0.3288367 ]]))
        np.testing.assert_almost_equal(self.sp(self.breast[:3], self.breast[:3], axis=0).X,
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
        self.spa = SpearmanRAbsolute()

    def test_spearmanrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.spa(self.breast[0]), 0)
        np.testing.assert_almost_equal(self.spa(self.breast[0], self.breast[0]), 0)
        np.testing.assert_almost_equal(self.spa(self.breast[0], self.breast[1]), 0.49166666666666664)

    def test_spearmanrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.spa(self.breast[:2]).X,
                                       np.array([[ 0.        ,  0.49166667],
                                                 [ 0.49166667,  0.        ]]))
        np.testing.assert_almost_equal(self.spa(self.breast[:3], axis=0).X,
                                       np.array([[ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.  ,  0.25,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ],
                                                 [ 0.25,  0.  ,  0.25,  0.  ,  0.  ,  0.  ,  0.25,  0.  ,  0.25],
                                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.  ,  0.25,  0.  ]]))
        np.testing.assert_almost_equal(self.spa(self.breast[:3], self.breast[:4]).X,
                                       np.array([[ 0.        ,  0.49166667,  0.075     ,  0.38333333],
                                                 [ 0.49166667,  0.        ,  0.38333333,  0.46666667],
                                                 [ 0.075     ,  0.38333333,  0.        ,  0.36666667]]))
        np.testing.assert_almost_equal(self.spa(self.breast[3], self.breast[:4]).X,
                                       np.array([[ 0.40995497,  0.3288367 ,  0.29564403,  0.07836298]]))
        np.testing.assert_almost_equal(self.spa(self.breast[:3], self.breast[:3], axis=0).X,
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
        self.p = PearsonR()

    def test_pearsonr_distance_one_example(self):
        np.testing.assert_almost_equal(self.p(self.breast[0]), 0)
        np.testing.assert_almost_equal(self.p(self.breast[0], self.breast[0]), 0)
        np.testing.assert_almost_equal(self.p(self.breast[0], self.breast[1]), 0.48462293898088876)
        np.testing.assert_almost_equal(self.p(self.breast[0], self.breast[1], axis=0), 0.48462293898088876)

    def test_pearsonr_distance_many_examples(self):
        np.testing.assert_almost_equal(self.p(self.breast[:2]).X,
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.p(self.breast[:20], axis=0).X,
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.57976119],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.57976119,  0.45930368,  0.        ]]))
        np.testing.assert_almost_equal(self.p(self.breast[:3], self.breast[:4]).X,
                                       np.array([[ 0.        ,  0.48462294,  0.10133593,  0.5016744 ],
                                                 [ 0.48462294,  0.        ,  0.32783865,  0.57317387],
                                                 [ 0.10133593,  0.32783865,  0.        ,  0.63789635]]))
        np.testing.assert_almost_equal(self.p(self.breast[2], self.breast[:3]).X,
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.p(self.breast[:20], self.breast[:20], axis=0).X,
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
        self.pa = PearsonRAbsolute()

    def test_pearsonrabsolute_distance_one_example(self):
        np.testing.assert_almost_equal(self.pa(self.breast[0]), 0)
        np.testing.assert_almost_equal(self.pa(self.breast[0], self.breast[0]), 0)
        np.testing.assert_almost_equal(self.pa(self.breast[0], self.breast[1]), 0.48462293898088876)
        np.testing.assert_almost_equal(self.pa(self.breast[0], self.breast[1], axis=0), 0.48462293898088876)

    def test_pearsonrabsolute_distance_many_examples(self):
        np.testing.assert_almost_equal(self.pa(self.breast[:2]).X,
                                       np.array([[ 0.        ,  0.48462294],
                                                 [ 0.48462294,  0.        ]]))
        np.testing.assert_almost_equal(self.pa(self.breast[:20], axis=0).X,
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.42023881],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.42023881,  0.45930368,  0.        ]]))
        np.testing.assert_almost_equal(self.pa(self.breast[:3], self.breast[:4]).X,
                                       np.array([[ 0.        ,  0.48462294,  0.10133593,  0.4983256 ],
                                                 [ 0.48462294,  0.        ,  0.32783865,  0.42682613],
                                                 [ 0.10133593,  0.32783865,  0.        , 0.36210365]]))
        np.testing.assert_almost_equal(self.pa(self.breast[2], self.breast[:3]).X,
                                       np.array([[ 0.10133593,  0.32783865,  0.        ]]))
        np.testing.assert_almost_equal(self.pa(self.breast[:2], self.breast[3]).X,
                                       np.array([[ 0.4983256 ],
                                                 [ 0.42682613]]))
        np.testing.assert_almost_equal(self.pa(self.breast[:20], self.breast[:20], axis=0).X,
                                       np.array([[ 0.        ,  0.10239274,  0.12786763,  0.13435117,  0.15580385, 0.27429811,  0.21006195,  0.24072005,  0.42847752],
                                                 [ 0.10239274,  0.        ,  0.01695375,  0.10313851,  0.1138925 , 0.16978203,  0.1155948 ,  0.08043531,  0.43326547],
                                                 [ 0.12786763,  0.01695375,  0.        ,  0.16049178,  0.13692762, 0.21784201,  0.11607395,  0.06493949,  0.46590168],
                                                 [ 0.13435117,  0.10313851,  0.16049178,  0.        ,  0.07181648, 0.15585667,  0.13891172,  0.21622332,  0.37404826],
                                                 [ 0.15580385,  0.1138925 ,  0.13692762,  0.07181648,  0.        , 0.16301705,  0.17324382,  0.21452448,  0.42283252],
                                                 [ 0.27429811,  0.16978203,  0.21784201,  0.15585667,  0.16301705, 0.        ,  0.25512861,  0.29560909,  0.42766076],
                                                 [ 0.21006195,  0.1155948 ,  0.11607395,  0.13891172,  0.17324382, 0.25512861,  0.        ,  0.14419442,  0.42023881],
                                                 [ 0.24072005,  0.08043531,  0.06493949,  0.21622332,  0.21452448, 0.29560909,  0.14419442,  0.        ,  0.45930368],
                                                 [ 0.42847752,  0.43326547,  0.46590168,  0.37404826,  0.42283252, 0.42766076,  0.42023881,  0.45930368,  0.        ]]))