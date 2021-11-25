import unittest
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable,\
    Domain, Table
from Orange import distance


class CommonTests:
    """Tests applicable to all distance measures"""

    def test_no_data(self):
        """distances return zero-dimensional matrices when no data"""
        n = len(self.data)
        np.testing.assert_almost_equal(
            self.Distance(Table.from_domain(self.domain)),
            np.zeros((0, 0)))
        np.testing.assert_almost_equal(
            self.Distance(self.data, Table.from_domain(self.domain)),
            np.zeros((n, 0)))
        np.testing.assert_almost_equal(
            self.Distance(Table.from_domain(self.domain), self.data),
            np.zeros((0, n)))

    def test_sparse(self):
        """Test sparse support in distances."""
        domain = Domain([ContinuousVariable(c) for c in "abc"])
        dense_data = Table.from_list(
            domain, [[1, 0, 2], [-1, 5, 0], [0, 1, 1], [7, 0, 0]])
        sparse_data = Table(domain, csr_matrix(dense_data.X))

        if not self.Distance.supports_sparse:
            self.assertRaises(TypeError, self.Distance, sparse_data)
        else:
            # check the result is the same for sparse and dense
            dist_dense = self.Distance(dense_data)
            dist_sparse = self.Distance(sparse_data)
            np.testing.assert_allclose(dist_sparse, dist_dense)


class CommonFittedTests(CommonTests):
    """Tests applicable to all distances with fitting"""
    def test_mismatching_attributes(self):
        """distances can't be computed if fit from other attributes"""
        def new_table(name):
            return Table.from_list(Domain([ContinuousVariable(name)]), [[1]])

        table1 = new_table("a")
        model = self.Distance().fit(table1)
        self.assertRaises(ValueError, model, new_table("b"))
        self.assertRaises(ValueError, model, table1, new_table("c"))
        self.assertRaises(ValueError, model, new_table("d"), table1)


class CommonNormalizedTests(CommonFittedTests):
    """Tests applicable to distances the have normalization"""

    def test_zero_variance(self):
        """zero-variance columns have no effect on row distance"""
        assert_almost_equal = np.testing.assert_almost_equal
        normalized = self.Distance(axis=1, normalize=True)
        nonnormalized = self.Distance(axis=1, normalize=False)

        def is_same(d1, d2, fit1=None, fit2=None):
            if fit1 is None:
                fit1 = d1
            if fit2 is None:
                fit2 = d2
            assert_almost_equal(normalized.fit(fit1)(d1),
                                normalized.fit(fit2)(d2))
            assert_almost_equal(nonnormalized.fit(fit1)(d1),
                                nonnormalized.fit(fit2)(d2))

        data = self.cont_data
        n = len(data)
        X = data.X
        domain = Domain(data.domain.attributes + (ContinuousVariable("d"),))
        data_const = Table(domain, np.hstack((X, np.ones((n, 1)))))
        data_nan = Table(domain, np.hstack((X, np.full((n, 1), np.nan))))
        data_nan_1 = Table(domain, np.hstack((X, np.full((n, 1), np.nan))))
        with data_nan_1.unlocked():
            data_nan_1.X[0, -1] = 1
        is_same(data, data_const)
        is_same(data, data_nan)
        is_same(data, data_nan_1)

        # Test whether it's possible to fit with singular data and use the
        # parameters on data where the same column is defined
        is_same(data_nan, data_const, data_const)
        is_same(data_const, data_const, data_nan)

        self.assertRaises(ValueError, self.Distance,
                          data_const, axis=0, normalize=True)
        self.assertRaises(ValueError, self.Distance,
                          data_nan, axis=0, normalize=True)
        self.assertRaises(ValueError, self.Distance,
                          data_nan_1, axis=0, normalize=True)

    def test_mixed_cols(self):
        """distance over columns raises exception for discrete columns"""
        self.assertRaises(ValueError, self.Distance, self.mixed_data, axis=0)
        self.assertRaises(ValueError,
                          self.Distance(axis=0).fit, self.mixed_data)


class FittedDistanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attributes = (
            ContinuousVariable("c1"),
            ContinuousVariable("c2"),
            ContinuousVariable("c3"),
            DiscreteVariable("d1", values=("a", "b")),
            DiscreteVariable("d2", values=("a", "b", "c", "d")),
            DiscreteVariable("d3", values=("a", "b", "c")))
        cls.domain = Domain(cls.attributes)
        cls.cont_domain = Domain(cls.attributes[:3])
        cls.disc_domain = Domain(cls.attributes[3:])

    def setUp(self):
        self.cont_data = Table.from_list(
            self.cont_domain,
            [[1, 3, 2],
             [-1, 5, 0],
             [1, 1, 1],
             [7, 2, 3]])

        self.cont_data2 = Table.from_list(
            self.cont_domain,
            [[2, 1, 3],
             [1, 2, 2]]
        )

        self.disc_data = Table.from_list(
            self.disc_domain,
            [[0, 0, 0],
             [0, 1, 1],
             [1, 3, 1]]
        )

        self.disc_data4 = Table.from_list(
            self.disc_domain,
            [[0, 0, 0],
             [0, 1, 1],
             [0, 1, 1],
             [1, 3, 1]]
        )

        self.mixed_data = self.data = Table.from_numpy(
            self.domain, np.hstack((self.cont_data.X[:3], self.disc_data.X)))



# Correct results in these tests were computed manually or with Excel;
# these are not regression tests
class EuclideanDistanceTest(FittedDistanceTest, CommonNormalizedTests):
    Distance = distance.Euclidean

    def test_euclidean_disc(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.disc_data

        model = distance.Euclidean().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1/3, 2/3, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 5/9, 1 - 3/9, 1 - 5/9])

        dist = model(data)
        assert_almost_equal(dist,
                            np.sqrt(np.array([[0, 2, 3],
                                              [2, 0, 2],
                                              [3, 2, 0]])))

        with data.unlocked():
            data.X[1, 0] = np.nan
        model = distance.Euclidean().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1/2, 1/2, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]
                            ])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 2/4, 1 - 3/9, 1 - 5/9])

        with data.unlocked():
            dist = model(data)
        assert_almost_equal(dist,
                            np.sqrt(np.array([[0, 2.5, 3],
                                              [2.5, 0, 1.5],
                                              [3, 1.5, 0]])))

        with data.unlocked():
            data.X[0, 0] = np.nan
        model = distance.Euclidean().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1, 0, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 1, 1 - 3/9, 1 - 5/9])

        dist = model(data)
        assert_almost_equal(dist,
                            np.sqrt(np.array([[0, 2, 2],
                                              [2, 0, 1],
                                              [2, 1, 0]])))

        data = self.disc_data4
        with data.unlocked():
            data.X[:2, 0] = np.nan
        model = distance.Euclidean().fit(data)

        assert_almost_equal(model.dist_missing_disc,
                            [[1/2, 1/2, 1, 1],
                             [3/4, 2/4, 1, 3/4],
                             [3/4, 1/4, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 2/4, 1 - 6/16, 1 - 10/16])

        dist = model(data)
        assert_almost_equal(dist,
                            np.sqrt(np.array([[0, 2.5, 2.5, 2.5],
                                              [2.5, 0, 0.5, 1.5],
                                              [2.5, 0.5, 0, 2],
                                              [2.5, 1.5, 2, 0]])))

    def test_euclidean_cont(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Euclidean(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            np.sqrt(np.array([[0, 12, 5, 38],
                              [12, 0, 21, 82],
                              [5, 21, 0, 41],
                              [38, 82, 41, 0]])))

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Euclidean(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 4.472135955, 2.236067977, 6.164414003],
             [4.472135955, 0, 5.385164807, 6.480740698],
             [2.236067977, 5.385164807, 0, 6.403124237],
             [6.164414003, 6.480740698, 6.403124237, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        dist = distance.Euclidean(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 5.099019514, 4.795831523, 4.472135955],
             [5.099019514, 0, 5.916079783, 6],
             [4.795831523, 5.916079783, 0, 6.403124237],
             [4.472135955, 6, 6.403124237, 0]])

    def test_euclidean_cont_normalized(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        model = distance.Euclidean(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.means, [2, 2.75, 1.5])
        assert_almost_equal(model.vars, [9, 2.1875, 1.25])
        assert_almost_equal(model.dist_missing2_cont, [1, 1, 1])

        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 1.654239383, 1.146423008, 1.621286967],
             [1.654239383, 0, 2.068662631, 3.035242727],
             [1.146423008, 2.068662631, 0, 1.956673562],
             [1.621286967, 3.035242727, 1.956673562, 0]])

        dist = distance.Euclidean(data, axis=1, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 1.654239383, 1.146423008, 1.621286967],
             [1.654239383, 0, 2.068662631, 3.035242727],
             [1.146423008, 2.068662631, 0, 1.956673562],
             [1.621286967, 3.035242727, 1.956673562, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        model = distance.Euclidean(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.means, [3, 2.75, 1.5])
        assert_almost_equal(model.vars, [8, 2.1875, 1.25])
        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 1.806733438, 1.146423008, 1.696635326],
             [1.806733438, 0, 2.192519751, 2.675283697],
             [1.146423008, 2.192519751, 0, 2.019547333],
             [1.696635326, 2.675283697, 2.019547333, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        model = distance.Euclidean(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.means, [4, 2.75, 1.5])
        assert_almost_equal(model.vars, [9, 2.1875, 1.25])
        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 1.874642823, 1.521277659, 1.276154939],
             [1.874642823, 0, 2.248809209, 2.580143961],
             [1.521277659, 2.248809209, 0, 1.956673562],
             [1.276154939, 2.580143961, 1.956673562, 0]])

    def test_euclidean_cols(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Euclidean(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 8.062257748, 4.242640687],
             [8.062257748, 0, 5.196152423],
             [4.242640687, 5.196152423, 0]])

        with data.unlocked():
            data.X[1, 1] = np.nan
        dist = distance.Euclidean(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 6.218252702, 4.242640687],
             [6.218252702, 0, 2.581988897],
             [4.242640687, 2.581988897, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Euclidean(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 6.218252702, 5.830951895],
             [6.218252702, 0, 2.581988897],
             [5.830951895, 2.581988897, 0]])

    def test_euclidean_cols_normalized(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Euclidean(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 2.455273959, 0.649839392],
             [2.455273959, 0, 2.473176308],
             [0.649839392, 2.473176308, 0]])

        with data.unlocked():
            data.X[1, 1] = np.nan
        dist = distance.Euclidean(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 2, 0.649839392],
             [2, 0, 1.704275472],
             [0.649839392, 1.704275472, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Euclidean(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 2, 1.450046001],
             [2, 0, 1.704275472],
             [1.450046001, 1.704275472, 0]])

    def test_euclidean_mixed(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.mixed_data

        model = distance.Euclidean(axis=1, normalize=True).fit(data)

        assert_almost_equal(model.means, [1/3, 3, 1])
        assert_almost_equal(model.vars, [8/9, 8/3, 2/3])
        assert_almost_equal(model.dist_missing_disc,
                            [[1/3, 2/3, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_cont, [1, 1, 1])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 5/9, 1 - 3/9, 1 - 5/9])
        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 2.828427125, 2.121320344],
             [2.828427125, 0, 2.828427125],
             [2.121320344, 2.828427125, 0]])

    def test_two_tables(self):
        assert_almost_equal = np.testing.assert_almost_equal

        dist = distance.Euclidean(self.cont_data, self.cont_data2,
                                  normalize=True)
        assert_almost_equal(
            dist,
            [[1.17040218, 0.47809144],
             [2.78516478, 1.96961039],
             [1.28668394, 0.79282497],
             [1.27179413, 1.54919334]])

        model = distance.Euclidean(normalize=True).fit(self.cont_data)
        dist = model(self.cont_data, self.cont_data2)
        assert_almost_equal(
            dist,
            [[1.17040218, 0.47809144],
             [2.78516478, 1.96961039],
             [1.28668394, 0.79282497],
             [1.27179413, 1.54919334]])

        dist = model(self.cont_data2)
        assert_almost_equal(dist, [[0, 0.827119692], [0.827119692, 0]])


class ManhattanDistanceTest(FittedDistanceTest, CommonNormalizedTests):
    Distance = distance.Euclidean

    # The data used for testing Euclidean distances unfortunately yields
    # mads = 1, so we change it a bit
    def setUp(self):
        super().setUp()
        self.cont_data = Table.from_list(
            self.cont_domain,
            [[1, 3, 2],
             [-1, 6, 0],
             [2, 7, 1],
             [7, 1, 3]])

    def test_manhattan_no_data(self):
        np.testing.assert_almost_equal(
            distance.Manhattan(Table.from_domain(self.domain)),
            np.zeros((0, 0)))
        np.testing.assert_almost_equal(
            distance.Manhattan(self.mixed_data, Table.from_domain(self.domain)),
            np.zeros((3, 0)))
        np.testing.assert_almost_equal(
            distance.Manhattan(Table.from_domain(self.domain), self.mixed_data),
            np.zeros((0, 3)))
        self.assertRaises(
            ValueError,
            distance.Manhattan, Table.from_domain(self.cont_domain),
            axis=0, normalize=True)

    def test_manhattan_disc(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.disc_data

        model = distance.Manhattan().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1/3, 2/3, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 5/9, 1 - 3/9, 1 - 5/9])
        dist = model(data)
        assert_almost_equal(dist,
                            [[0, 2, 3],
                             [2, 0, 2],
                             [3, 2, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        model = distance.Manhattan().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1/2, 1/2, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 2/4, 1 - 3/9, 1 - 5/9])

        dist = model(data)
        assert_almost_equal(dist,
                            [[0, 2.5, 3],
                             [2.5, 0, 1.5],
                             [3, 1.5, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        model = distance.Manhattan().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1, 0, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 1, 1 - 3/9, 1 - 5/9])

        dist = model(data)
        assert_almost_equal(dist,
                            [[0, 2, 2],
                             [2, 0, 1],
                             [2, 1, 0]])

        data = self.disc_data4
        with data.unlocked():
            data.X[:2, 0] = np.nan
        model = distance.Manhattan().fit(data)
        assert_almost_equal(model.dist_missing_disc,
                            [[1/2, 1/2, 1, 1],
                             [3/4, 2/4, 1, 3/4],
                             [3/4, 1/4, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 2/4, 1 - 6/16, 1 - 10/16])

        dist = model(data)
        assert_almost_equal(dist,
                            [[0, 2.5, 2.5, 2.5],
                             [2.5, 0, 0.5, 1.5],
                             [2.5, 0.5, 0, 2],
                             [2.5, 1.5, 2, 0]])

    def test_manhattan_cont(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Manhattan(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 7, 6, 9],
             [7, 0, 5, 16],
             [6, 5, 0, 13],
             [9, 16, 13, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Manhattan(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 7, 6, 9],
             [7, 0, 3, 14],
             [6, 3, 0, 13],
             [9, 14, 13, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        dist = distance.Manhattan(data, axis=1, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 10, 10, 8],
             [10, 0, 7, 13],
             [10, 7, 0, 13],
             [8, 13, 13, 0]])

    def test_manhattan_cont_normalized(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        model = distance.Manhattan(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.medians, [1.5, 4.5, 1.5])
        assert_almost_equal(model.mads, [1.5, 2, 1])
        assert_almost_equal(model.dist_missing2_cont, np.ones(3))

        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 2.416666667, 1.833333333, 3],
             [2.416666667, 0, 1.75, 5.416666667],
             [1.833333333, 1.75, 0, 4.166666667],
             [3, 5.416666667, 4.166666667, 0]])

        dist = distance.Manhattan(data, axis=1, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 2.416666667, 1.833333333, 3],
             [2.416666667, 0, 1.75, 5.416666667],
             [1.833333333, 1.75, 0, 4.166666667],
             [3, 5.416666667, 4.166666667, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        model = distance.Manhattan(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.medians, [2, 4.5, 1.5])
        assert_almost_equal(model.mads, [1, 2, 1])

        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 2.75, 2, 4],
             [2.75, 0, 1.25, 5.75],
             [2, 1.25, 0, 5],
             [4, 5.75, 5, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        model = distance.Manhattan(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.medians, [4.5, 4.5, 1.5])
        assert_almost_equal(model.mads, [2.5, 2, 1])

        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 2.75, 2.5, 2],
             [2.75, 0, 1.75, 3.75],
             [2.5, 1.75, 0, 3.5],
             [2, 3.75, 3.5, 0]])

    def test_manhattan_cols(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Manhattan(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 20, 7],
             [20, 0, 15],
             [7, 15, 0]])

        with data.unlocked():
            data.X[1, 1] = np.nan
        dist = distance.Manhattan(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 19, 7],
             [19, 0, 14],
             [7, 14, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Manhattan(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 17, 9],
             [17, 0, 14],
             [9, 14, 0]])


    def test_manhattan_cols_normalized(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Manhattan(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 4.5833333, 2],
             [4.5833333, 0, 4.25],
             [2, 4.25, 0]])

        with data.unlocked():
            data.X[1, 1] = np.nan
        dist = distance.Manhattan(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 4.6666667, 2],
             [4.6666667, 0, 4],
             [2, 4, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Manhattan(data, axis=0, normalize=True)
        assert_almost_equal(
            dist,
            [[0, 5.5, 4],
             [5.5, 0, 4],
             [4, 4, 0]])

    def test_manhattan_mixed(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.mixed_data

        with data.unlocked():
            data.X[2, 0] = 2  # prevent mads[0] = 0
        model = distance.Manhattan(axis=1, normalize=True).fit(data)
        assert_almost_equal(model.medians, [1, 3, 1])
        assert_almost_equal(model.mads, [1, 2, 1])
        assert_almost_equal(model.dist_missing_disc,
                            [[1/3, 2/3, 1, 1],
                             [2/3, 2/3, 1, 2/3],
                             [2/3, 1/3, 1, 1]])
        assert_almost_equal(model.dist_missing2_disc,
                            [1 - 5/9, 1 - 3/9, 1 - 5/9])

        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 4.5, 4.5],
             [4.5, 0, 5],
             [4.5, 5, 0]])

    def test_two_tables(self):
        assert_almost_equal = np.testing.assert_almost_equal

        dist = distance.Manhattan(self.cont_data, self.cont_data2,
                                  normalize=True)
        assert_almost_equal(
            dist,
            [[1.3333333, 0.25],
             [3.75, 2.6666667],
             [2.5, 2.0833333],
             [1.6666667, 2.75]])

        model = distance.Manhattan(normalize=True).fit(self.cont_data)
        dist = model(self.cont_data, self.cont_data2)
        assert_almost_equal(
            dist,
            [[1.3333333, 0.25],
             [3.75, 2.6666667],
             [2.5, 2.0833333],
             [1.6666667, 2.75]])

        dist = model(self.cont_data2)
        assert_almost_equal(dist, [[0, 1.083333333], [1.083333333, 0]])

    def test_manhattan_mixed_cols(self):
        self.assertRaises(ValueError,
                          distance.Manhattan, self.mixed_data, axis=0)
        self.assertRaises(ValueError,
                          distance.Manhattan(axis=0).fit, self.mixed_data)


class CosineDistanceTest(FittedDistanceTest, CommonFittedTests):
    Distance = distance.Cosine

    def test_cosine_disc(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.disc_data
        with data.unlocked():
            data.X = np.array([[1, 0, 0],
                               [0, 1, 1],
                               [1, 3, 0]], dtype=float)

        model = distance.Cosine().fit(data)
        assert_almost_equal(model.means, [2 / 3, 2 / 3, 1 / 3])

        dist = model(data)
        assert_almost_equal(dist, 1 - np.array([[1, 0, 1 / sqrt(2)],
                                                [0, 1, 0.5],
                                                [1 / sqrt(2), 0.5, 1]]))

        with data.unlocked():
            data.X[1, 1] = np.nan
        model = distance.Cosine().fit(data)
        assert_almost_equal(model.means, [2 / 3, 1 / 2, 1 / 3])
        dist = model(data)
        assert_almost_equal(
            dist,
            1 - np.array([[1, 0, 1 / sqrt(2)],
                          [0, 1, 0.5 / sqrt(1.25) / sqrt(2)],
                          [1 / sqrt(2), 0.5 / sqrt(1.25) / sqrt(2), 1]]))

        with data.unlocked():
            data.X = np.array([[1, 0, 0],
                               [0, np.nan, 1],
                               [1, np.nan, 1],
                               [1, 3, 1]])
        model = distance.Cosine().fit(data)
        dist = model(data)
        assert_almost_equal(model.means, [0.75, 0.5, 0.75])
        assert_almost_equal(dist, [[0, 1, 0.333333333, 0.422649731],
                                   [1, 0, 0.254644008, 0.225403331],
                                   [0.333333333, 0.254644008, 0, 0.037749551],
                                   [0.422649731, 0.225403331, 0.037749551, 0]])

    def test_cosine_cont(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Cosine(data, axis=1)
        assert_almost_equal(
            dist,
            [[0, 0.266200614, 0.0741799, 0.355097978],
             [0.266200614, 0, 0.547089186, 0.925279678],
             [0.0741799, 0.547089186, 0, 0.12011731],
             [0.355097978, 0.925279678, 0.12011731, 0]]
        )

        with data.unlocked():
            data.X[1, 0] = np.nan
        dist = distance.Cosine(data, axis=1)
        assert_almost_equal(
            dist,
            [[0, 0.174971353, 0.0741799, 0.355097978],
             [0.174971353, 0, 0.207881966, 0.324809395],
             [0.0741799, 0.207881966, 0, 0.12011731],
             [0.355097978, 0.324809395, 0.12011731, 0]])

        with data.unlocked():
            data.X[0, 0] = np.nan
        dist = distance.Cosine(data, axis=1)
        assert_almost_equal(
            dist,
            [[0, 0.100977075, 0.035098719, 0.056666739],
             [0.100977075, 0, 0.188497329, 0.246304671],
             [0.035098719, 0.188497329, 0, 0.12011731],
             [0.056666739, 0.246304671, 0.12011731, 0]])

    def test_cosine_mixed(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.mixed_data
        with data.unlocked():
            data.X = np.array([[1, 3, 2, 1, 0, 0],
                               [-1, 5, 0, 0, 1, 1],
                               [1, 1, 1, 1, 3, 0]], dtype=float)

        model = distance.Cosine(axis=1).fit(data)
        assert_almost_equal(model.means, [1/3, 3, 1, 2/3, 2/3, 1/3])
        dist = model(data)
        assert_almost_equal(
            dist,
            [[0, 0.316869949, 0.191709623],
             [0.316869949, 0, 0.577422873],
             [0.191709623, 0.577422873, 0]])

    def test_two_tables(self):
        assert_almost_equal = np.testing.assert_almost_equal
        with self.cont_data.unlocked(), self.cont_data2.unlocked():
            self.cont_data.X[1, 0] = np.nan
            self.cont_data2.X[1, 0] = np.nan

        dist = distance.Cosine(self.cont_data, self.cont_data2)
        assert_almost_equal(
            dist,
            [[0.2142857, 0.1573352],
             [0.4958158, 0.2097042],
             [0.0741799, 0.0198039],
             [0.1514447, 0.0451363]])

        model = distance.Cosine().fit(self.cont_data)
        dist = model(self.cont_data, self.cont_data2)
        assert_almost_equal(
            dist,
            [[0.2142857, 0.1573352],
             [0.4958158, 0.2097042],
             [0.0741799, 0.0198039],
             [0.1514447, 0.0451363]])

        dist = model(self.cont_data2)
        assert_almost_equal(dist, [[0, 0.092514787], [0.092514787, 0]])

    def test_cosine_cols(self):
        assert_almost_equal = np.testing.assert_almost_equal
        data = self.cont_data

        dist = distance.Cosine(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 0.711324865, 0.11050082],
             [0.711324865, 0, 0.44365136],
             [0.11050082, 0.44365136, 0]])

        with data.unlocked():
            data.X[1, 1] = np.nan
        dist = distance.Cosine(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 0.47702364, 0.11050082],
             [0.47702364, 0, 0.181076975],
             [0.11050082, 0.181076975, 0]])

        with data.unlocked():
            data.X[1, 0] = np.nan
            data.X[1, 2] = 2
        dist = distance.Cosine(data, axis=0, normalize=False)
        assert_almost_equal(
            dist,
            [[0, 0.269703257, 0.087129071],
             [0.269703257, 0, 0.055555556],
             [0.087129071, 0.055555556, 0]])


class JaccardDistanceTest(unittest.TestCase, CommonFittedTests):
    Distance = distance.Jaccard

    def setUp(self):
        self.domain = Domain([DiscreteVariable(c) for c in "abc"])
        self.data = Table.from_list(
            self.domain,
            [[0, 1, 1],
             [1, 1, 1],
             [1, 0, 1],
             [1, 0, 0]])

    def test_jaccard_rows(self):
        assert_almost_equal = np.testing.assert_almost_equal

        model = distance.Jaccard().fit(self.data)
        assert_almost_equal(model.ps, [0.75, 0.5, 0.75])
        assert_almost_equal(
            model(self.data),
            1 - np.array([[1, 2/3, 1/3, 0],
                          [2/3, 1, 2/3, 1/3],
                          [1/3, 2/3, 1, 1/2],
                          [0, 1/3, 1/2, 1]]))

        X = self.data.X
        with self.data.unlocked():
            X[1, 0] = X[2, 0] = X[3, 1] = np.nan
        model = distance.Jaccard().fit(self.data)
        assert_almost_equal(model.ps, np.array([0.5, 2/3, 0.75]))

        assert_almost_equal(
            model(self.data),
            1 - np.array([[      1,        2 / 2.5,        1 / 2.5,        2/3 / 3],
                          [2 / 2.5,              1,    1.25 / 2.75,  (1/2+2/3) / 3],
                          [1 / 2.5,    1.25 / 2.75,              1,  0.5 / (2+2/3)],
                          [2/3 / 3,  (1/2+2/3) / 3,  0.5 / (2+2/3),            1]]))

    def test_jaccard_cols(self):
        assert_almost_equal = np.testing.assert_almost_equal
        model = distance.Jaccard(axis=0).fit(self.data)
        assert_almost_equal(model.ps, [0.75, 0.5, 0.75])
        assert_almost_equal(
            model(self.data),
            1 - np.array([[1, 1/4, 1/2],
                          [1/4, 1, 2/3],
                          [1/2, 2/3, 1]]))

        with self.data.unlocked():
            self.data.X = np.array([[0, 1, 1],
                                    [np.nan, np.nan, 1],
                                    [np.nan, 0, 1],
                                    [1, 1, 0]])
        model = distance.Jaccard(axis=0).fit(self.data)
        assert_almost_equal(model.ps, [0.5, 2/3, 0.75])
        assert_almost_equal(
            model(self.data),
            1 - np.array([[1, 0.4, 0.25],
                          [0.4, 1, 5/12],
                          [0.25, 5/12, 1]]))

    def test_zero_instances(self):
        "Test all zero instances"
        domain = Domain([ContinuousVariable(c) for c in "abc"])
        dense_data = Table.from_list(
            domain, [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        sparse_data = Table(domain, csr_matrix(dense_data.X))
        dist_dense = self.Distance(dense_data)
        dist_sparse = self.Distance(sparse_data)

        # false positive, pylint: disable=unsubscriptable-object
        self.assertEqual(dist_dense[0][1], 0)
        self.assertEqual(dist_sparse[0][1], 0)
        self.assertEqual(dist_dense[0][2], 1)
        self.assertEqual(dist_sparse[0][2], 1)


class HammingDistanceTest(FittedDistanceTest):
    Distance = distance.Hamming

    def test_hamming_col(self):
        np.testing.assert_almost_equal(
            self.Distance(self.disc_data, axis=0),
            [[0, 2 / 3, 1 / 3],
             [2 / 3, 0, 1 / 3],
             [1 / 3, 1 / 3, 0]])

    def test_hamming_row(self):
        np.testing.assert_almost_equal(
            self.Distance(self.disc_data),
            [[0, 2 / 3, 1],
             [2 / 3, 0, 2 / 3],
             [1, 2 / 3, 0]]
        )

    def test_hamming_row_secondary_data(self):
        np.testing.assert_almost_equal(
            self.Distance(self.disc_data, self.disc_data[:2]),
            [[0, 2 / 3],
             [2 / 3, 0],
             [1, 2 / 3]])


class TestHelperFunctions(unittest.TestCase):
    # pylint: disable=protected-access, no-self-use
    def test_interruptable_dot(self):
        dot = distance.distance._interruptible_dot
        k, m, n = 20, 30, 40
        a = np.random.randint(10, size=(m, n))
        b = np.random.randint(10, size=(n, k))

        c = dot(a, b, step=10)
        np.testing.assert_array_equal(c, np.dot(a, b))

    def test_interruptable_dot_list(self):
        dot = distance.distance._interruptible_dot
        a = [[1, 2, 3], [1, 2, 3]]
        b = [3, 2, 1]
        c = dot(a, b, step=10)
        np.testing.assert_array_equal(c, np.dot(a, b))

    def test_interruptable_dot_scalar(self):
        dot = distance.distance._interruptible_dot
        a = 2
        b = 3
        c = dot(a, b, step=10)
        np.testing.assert_array_equal(c, np.dot(a, b))

    def test_interruptable_sqrt(self):
        sqrt_ = distance.distance._interruptible_sqrt
        n = 100
        a = np.random.randint(10, size=(n, n))

        new_a = sqrt_(a, step=10)
        np.testing.assert_array_equal(new_a, np.sqrt(a))

    def test_interruptable_sqrt_list(self):
        sqrt_ = distance.distance._interruptible_sqrt
        l = [1, 2, 3]
        new_l = sqrt_(l, step=10)
        np.testing.assert_array_equal(new_l, np.sqrt(l))

    def test_interruptable_sqrt_scalar(self):
        sqrt_ = distance.distance._interruptible_sqrt
        i = 9
        new_i = sqrt_(i, step=10)
        np.testing.assert_array_equal(new_i, np.sqrt(i))


class TestDataUtilities(unittest.TestCase):
    def test_remove_discrete(self):
        d1, d2, d3 = (DiscreteVariable(c, values=tuple("123")) for c in "abc")
        c1, c2 = (ContinuousVariable(c) for c in "xy")
        t = StringVariable("t")
        domain = Domain([d1, c1], d2, [c2, d3, t])
        data = Table.from_domain(domain, 5)

        reduced = distance.remove_discrete_features(data)
        self.assertEqual(reduced.domain.attributes, (c1, ))
        self.assertEqual(reduced.domain.class_var, d2)
        self.assertEqual(reduced.domain.metas, (c2, d3, t))

        reduced = distance.remove_discrete_features(data, to_metas=True)
        self.assertEqual(reduced.domain.attributes, (c1, ))
        self.assertEqual(reduced.domain.class_var, d2)
        self.assertEqual(reduced.domain.metas, (c2, d3, t, d1))

    def test_remove_non_binary(self):
        b1, b2, b3 = (DiscreteVariable(c, values=tuple("12")) for c in "abc")
        d1, d2, d3 = (DiscreteVariable(c, values=tuple("123")) for c in "def")
        c1, c2 = (ContinuousVariable(c) for c in "xy")
        t = StringVariable("t")
        domain = Domain([d1, b1, b2, c1], d2, [c2, d3, t, b3])
        data = Table.from_domain(domain, 5)

        reduced = distance.remove_nonbinary_features(data)
        self.assertEqual(reduced.domain.attributes, (b1, b2))
        self.assertEqual(reduced.domain.class_var, d2)
        self.assertEqual(reduced.domain.metas, (c2, d3, t, b3))

        reduced = distance.remove_nonbinary_features(data, to_metas=True)
        self.assertEqual(reduced.domain.attributes, (b1, b2))
        self.assertEqual(reduced.domain.class_var, d2)
        self.assertEqual(reduced.domain.metas, (c2, d3, t, b3, d1, c1))


if __name__ == "__main__":
    unittest.main()
