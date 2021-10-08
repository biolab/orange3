# Test methods with long descriptive names can omit docstrings
# Test internal methods
# pylint: disable=missing-docstring, protected-access
import copy
import pickle
import unittest
from unittest.mock import Mock
import warnings

import numpy as np
import scipy.sparse as sp

from Orange.statistics import distribution
from Orange import data
from Orange.tests import test_filename


def assert_dist_equal(dist, expected):
    np.testing.assert_array_equal(np.asarray(dist), expected)


def assert_dist_almost_equal(dist, expected):
    np.testing.assert_almost_equal(np.asarray(dist), expected)


class TestDiscreteDistribution(unittest.TestCase):
    def setUp(self):
        self.freqs = [4.0, 20.0, 13.0, 8.0, 10.0, 41.0, 5.0]
        s = sum(self.freqs)
        self.rfreqs = [x/s for x in self.freqs]

        self.data = data.Table.from_numpy(
            data.Domain(
                attributes=[
                    data.DiscreteVariable('rgb', values=('r', 'g', 'b', 'a')),
                    data.DiscreteVariable('num', values=('1', '2', '3')),
                ]
            ),
            X=np.array([
                [0, 2, 0, 1, 1, 0, np.nan, 1],
                [0, 2, 0, np.nan, 1, 2, np.nan, 1],
            ]).T
        )
        self.rgb, self.num = distribution.get_distributions(self.data)

    def test_from_table(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        assert_dist_equal(disc, self.freqs)

        disc2 = distribution.Discrete(d, d.domain.class_var)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIs(disc2.variable, d.domain.class_var)
        self.assertEqual(disc, disc2)

        disc3 = distribution.Discrete(d, len(d.domain.attributes))
        self.assertIsInstance(disc3, np.ndarray)
        self.assertIs(disc3.variable, d.domain.class_var)
        self.assertEqual(disc, disc3)

        disc5 = distribution.class_distribution(d)
        self.assertIsInstance(disc5, np.ndarray)
        self.assertIs(disc5.variable, d.domain.class_var)
        self.assertEqual(disc, disc5)

    def test_construction(self):
        d = data.Table("zoo")

        disc = distribution.Discrete(d, "type")
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        self.assertIs(disc.variable, d.domain.class_var)

        disc7 = distribution.Discrete(self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Discrete(None, d.domain.class_var)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc1.variable, d.domain.class_var)
        self.assertEqual(disc.unknowns, 0)
        assert_dist_equal(disc1, [0]*len(d.domain.class_var.values))

    def test_fallback(self):
        d = data.Table("zoo")
        default = distribution.Discrete(d, "type")

        d._compute_distributions = Mock(side_effect=NotImplementedError)
        fallback = distribution.Discrete(d, "type")

        np.testing.assert_almost_equal(
            np.asarray(fallback), np.asarray(default))
        np.testing.assert_almost_equal(fallback.unknowns, default.unknowns)

    def test_fallback_with_weights_and_nan(self):
        d = data.Table("zoo")
        with d.unlocked():
            d.set_weights(np.random.uniform(0., 1., size=len(d)))
            d.Y[::10] = np.nan

        default = distribution.Discrete(d, "type")
        d._compute_distributions = Mock(side_effect=NotImplementedError)
        fallback = distribution.Discrete(d, "type")

        np.testing.assert_almost_equal(
            np.asarray(fallback), np.asarray(default))
        np.testing.assert_almost_equal(fallback.unknowns, default.unknowns)

    def test_pickle(self):
        d = data.Table("zoo")
        d1 = distribution.Discrete(d, 0)
        dc = pickle.loads(pickle.dumps(d1))
        # This always worked because `other` wasn't required to have `unknowns`
        self.assertEqual(d1, dc)
        # This failed before implementing `__reduce__`
        self.assertEqual(dc, d1)
        self.assertEqual(hash(d1), hash(dc))
        # Test that `dc` has the required attributes
        self.assertEqual(dc.variable, d1.variable)
        self.assertEqual(dc.unknowns, d1.unknowns)

    def test_deepcopy(self):
        d = data.Table("zoo")
        d1 = distribution.Discrete(d, 0)
        dc = copy.deepcopy(d1)
        # This always worked because `other` wasn't required to have `unknowns`
        self.assertEqual(d1, dc)
        # This failed before implementing `__deepcopy__`
        self.assertEqual(dc, d1)
        self.assertEqual(hash(d1), hash(dc))
        # Test that `dc` has the required attributes
        self.assertEqual(dc.variable, d1.variable)
        self.assertEqual(dc.unknowns, d1.unknowns)

    def test_equality(self):
        d = data.Table("zoo")
        d1 = distribution.Discrete(d, 0)
        d2 = distribution.Discrete(d, 0)
        d3 = distribution.Discrete(d, 1)

        self.assertEqual(d1, d1)
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)

    def test_indexing(self):
        d = data.Table("zoo")
        indamphibian = d.domain.class_var.to_val("amphibian")

        disc = distribution.class_distribution(d)

        self.assertEqual(len(disc), len(d.domain.class_var.values))

        self.assertEqual(disc["mammal"], 41)
        self.assertEqual(disc[indamphibian], 4)

        disc["mammal"] = 100
        self.assertEqual(disc[d.domain.class_var.to_val("mammal")], 100)

        disc[indamphibian] = 33
        self.assertEqual(disc["amphibian"], 33)

        disc = distribution.class_distribution(d)
        self.assertEqual(list(disc), self.freqs)

    def test_hash(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")

        disc2 = distribution.Discrete(d, d.domain.class_var)
        self.assertEqual(hash(disc), hash(disc2))

        disc2[0] += 1
        self.assertNotEqual(hash(disc), hash(disc2))

        disc2[0] -= 1
        self.assertEqual(hash(disc), hash(disc2))

        disc2.unknowns += 1
        self.assertNotEqual(hash(disc), hash(disc2))

    def test_add(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")

        disc += [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(disc, [5.0, 22.0, 16.0, 12.0, 15.0, 47.0, 12.0])

        disc2 = distribution.Discrete(d, d.domain.class_var)

        disc3 = disc - disc2
        self.assertEqual(disc3, list(range(1, 8)))

        disc3 *= 2
        self.assertEqual(disc3, [2*x for x in range(1, 8)])

    def test_normalize(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)

        disc1 = distribution.Discrete(None, d.domain.class_var)
        disc1.normalize()
        v = len(d.domain.class_var.values)
        assert_dist_almost_equal(disc1, [1/v]*v)

    def test_modus(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        self.assertEqual(str(disc.modus()), "mammal")

    def test_sample(self):
        ans = self.num.sample((500, 2), replace=True)
        np.testing.assert_equal(np.unique(ans), [0, 1, 2])

        # Check that samping a single value works too
        self.assertIn(self.num.sample(), [0, 1, 2])

    def test_min_max(self):
        # Min and max don't make sense in the context of nominal variables
        self.assertEqual(self.rgb.min(), None)
        self.assertEqual(self.rgb.max(), None)
        # Min and max should work for ordinal variables
        self.assertEqual(self.num.min(), None)
        self.assertEqual(self.num.max(), None)

    def test_array_with_unknowns(self):
        d = data.Table("zoo")
        with d.unlocked():
            d.Y[0] = np.nan
        disc = distribution.Discrete(d, "type")
        self.assertIsInstance(disc, np.ndarray)
        self.assertEqual(disc.unknowns, 1)
        true_freq = [4., 20., 13., 8., 10., 40., 5.]
        assert_dist_equal(disc, true_freq)
        np.testing.assert_array_equal(disc.array_with_unknowns,
                                      np.append(true_freq, 1))


class TestContinuousDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = data.Table("iris")

        cls.data = data.Table.from_numpy(
            data.Domain(
                attributes=[
                    data.ContinuousVariable('n1'),
                    data.ContinuousVariable('n2'),
                ]
            ),
            X=np.array([range(10), [1, 1, 1, 5, 5, 8, 9, np.nan, 9, 9]]).T
        )
        cls.n1, cls.n2 = distribution.get_distributions(cls.data)

    def setUp(self):
        self.freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                               (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                               (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                               (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                               (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                               (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                               (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                               (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                               (6.6, 1), (6.7, 2), (6.9, 1)]).T

    def test_from_table(self):
        d = self.iris
        petal_length = d.columns.petal_length

        for attr in ["petal length", d.domain[2], 2]:
            disc = distribution.Continuous(d, attr)
            self.assertIsInstance(disc, np.ndarray)
            self.assertIs(disc.variable, petal_length)
            self.assertEqual(disc.unknowns, 0)
            assert_dist_almost_equal(disc, self.freqs)

    def test_construction(self):
        d = self.iris
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")

        disc7 = distribution.Continuous(self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc7 = distribution.Continuous(self.freqs, petal_length)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Continuous(10, petal_length)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        assert_dist_equal(disc1, np.zeros((2, 10)))

        dd = [list(range(5)), [1, 1, 2, 5, 1]]
        disc2 = distribution.Continuous(dd)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIsNone(disc2.variable)
        self.assertEqual(disc2.unknowns, 0)
        assert_dist_equal(disc2, dd)

    def test_pickle(self):
        d1 = distribution.Continuous(self.iris, 0)
        dc = pickle.loads(pickle.dumps(d1))
        # This always worked because `other` wasn't required to have `unknowns`
        self.assertEqual(d1, dc)
        # This failed before implementing `__reduce__`
        self.assertEqual(dc, d1)
        self.assertEqual(hash(d1), hash(dc))
        # Test that `dc` has the required attributes
        self.assertEqual(dc.variable, d1.variable)
        self.assertEqual(dc.unknowns, d1.unknowns)

    def test_deepcopy(self):
        d1 = distribution.Continuous(self.iris, 0)
        dc = copy.deepcopy(d1)
        # This always worked because `other` wasn't required to have `unknowns`
        self.assertEqual(d1, dc)
        # This failed before implementing `__deepcopy__`
        self.assertEqual(dc, d1)
        self.assertEqual(hash(d1), hash(dc))
        # Test that `dc` has the required attributes
        self.assertEqual(dc.variable, d1.variable)
        self.assertEqual(dc.unknowns, d1.unknowns)

    def test_hash(self):
        d = self.iris
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")
        disc2 = distribution.Continuous(d, petal_length)
        self.assertEqual(hash(disc), hash(disc2))

        disc2[0, 0] += 1
        self.assertNotEqual(hash(disc), hash(disc2))

        disc2[0, 0] -= 1
        self.assertEqual(hash(disc), hash(disc2))

        disc2.unknowns += 1
        self.assertNotEqual(hash(disc), hash(disc2))

    def test_normalize(self):
        d = self.iris
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")

        assert_dist_equal(disc, self.freqs)
        disc.normalize()
        self.freqs[1, :] /= 150
        assert_dist_equal(disc, self.freqs)

        disc1 = distribution.Continuous(10, petal_length)
        disc1.normalize()
        f = np.zeros((2, 10))
        f[1, :] = 0.1
        assert_dist_equal(disc1, f)

    def test_modus(self):
        disc = distribution.Continuous([list(range(5)), [1, 1, 2, 5, 1]])
        self.assertEqual(disc.modus(), 3)

    def test_random(self):
        d = self.iris

        disc = distribution.Continuous(d, "petal length")
        ans = set()
        for i in range(1000):
            v = disc.sample()
            self.assertIn(v, self.freqs)
            ans.add(v)
        self.assertGreater(len(ans), 10)

    def test_min_max(self):
        self.assertEqual(self.n1.min(), 0)
        self.assertFalse(isinstance(self.n1.min(), distribution.Continuous))
        self.assertEqual(self.n1.max(), 9)
        self.assertFalse(isinstance(self.n1.max(), distribution.Continuous))


class TestClassDistribution(unittest.TestCase):
    def test_class_distribution(self):
        d = data.Table("zoo")
        disc = distribution.class_distribution(d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        assert_dist_equal(disc, [4.0, 20.0, 13.0, 8.0, 10.0, 41.0, 5.0])

    def test_multiple_target_variables(self):
        d = data.Table.from_numpy(
            data.Domain(
                attributes=[data.ContinuousVariable('n1')],
                class_vars=[
                    data.DiscreteVariable('c1', values=('r', 'g', 'b', 'a')),
                    data.DiscreteVariable('c2', values=('r', 'g', 'b', 'a')),
                    data.DiscreteVariable('c3', values=('r', 'g', 'b', 'a')),
                ]
            ),
            X=np.array([range(5)]).T,
            Y=np.array([
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ]).T
        )
        dists = distribution.class_distribution(d)
        self.assertEqual(len(dists), 3)
        self.assertTrue(all(isinstance(dist, distribution.Discrete) for dist in dists))


class TestGetDistribution(unittest.TestCase):
    def test_get_distribution(self):
        d = data.Table("iris")
        cls = d.domain.class_var
        disc = distribution.get_distribution(d, cls)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, cls)
        self.assertEqual(disc.unknowns, 0)
        assert_dist_equal(disc, [50, 50, 50])

        petal_length = d.columns.petal_length
        freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                          (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                          (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                          (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                          (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                          (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                          (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                          (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                          (6.6, 1), (6.7, 2), (6.9, 1)]).T
        disc = distribution.get_distribution(d, petal_length)
        assert_dist_equal(disc, freqs)


class TestDomainDistribution(unittest.TestCase):
    def test_get_distributions(self):
        d = data.Table("iris")
        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 5)
        for i in range(4):
            self.assertIsInstance(ddist[i], distribution.Continuous)
        self.assertIsInstance(ddist[-1], distribution.Discrete)

        freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                          (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                          (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                          (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                          (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                          (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                          (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                          (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                          (6.6, 1), (6.7, 2), (6.9, 1)]).T
        assert_dist_equal(ddist[2], freqs)
        assert_dist_equal(ddist[-1], [50, 50, 50])

    def test_sparse_get_distributions(self):
        def assert_dist_and_unknowns(computed, goal_dist):
            nonlocal d
            goal_dist = np.array(goal_dist)
            sum_dist = np.sum(goal_dist[1, :] if goal_dist.ndim == 2 else goal_dist)
            n_all = np.sum(d.W) if d.has_weights() else len(d)

            assert_dist_almost_equal(computed, goal_dist)
            self.assertEqual(computed.unknowns, n_all - sum_dist)

        domain = data.Domain(
            [data.DiscreteVariable("d%i" % i, values=tuple("abc")) for i in range(10)] +
            [data.ContinuousVariable("c%i" % i) for i in range(10)])

        # pylint: disable=bad-whitespace
        X = sp.csr_matrix(
            # 0  1  2  3       4       5       6  7  8  9 10 11 12   13 14 15 16      17 18 19
            # --------------------------------------------------------------------------------
            [[0, 2, 0, 2,      1,      1,      2, 0, 0, 1, 0, 0, 0,   1, 1, 0, 2, np.nan, 2, 0],
             [0, 0, 1, 1, np.nan, np.nan,      1, 0, 2, 0, 0, 0, 0,   0, 2, 0, 1, np.nan, 0, 0],
             [0, 0, 0, 1,      0,      2, np.nan, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0,      0, 0, 0],
             [0, 0, 0, 0,      0,      0,      0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0,      0, 0, 0],
             [0, 0, 2, 0,      0,      0,      1, 0, 0, 0, 0, 0, 0, 1.1, 0, 0, 0,      0, 0, 0]]
        )
        warnings.filterwarnings("ignore", ".*", sp.SparseEfficiencyWarning)
        X[0, 0] = 0

        d = data.Table.from_numpy(domain, X)
        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 20)
        zeros = [5, 0, 0]
        assert_dist_and_unknowns(ddist[0], zeros)
        assert_dist_and_unknowns(ddist[1], [4, 0, 1])
        assert_dist_and_unknowns(ddist[2], [3, 1, 1])
        assert_dist_and_unknowns(ddist[3], [2, 2, 1])
        assert_dist_and_unknowns(ddist[4], [3, 1, 0])
        assert_dist_and_unknowns(ddist[5], [2, 1, 1])
        assert_dist_and_unknowns(ddist[6], [1, 2, 1])
        assert_dist_and_unknowns(ddist[7], zeros)
        assert_dist_and_unknowns(ddist[8], [4, 0, 1])
        assert_dist_and_unknowns(ddist[9], [4, 1, 0])

        zeros = [[0], [5]]
        assert_dist_and_unknowns(ddist[10], zeros)
        assert_dist_and_unknowns(ddist[11], zeros)
        assert_dist_and_unknowns(ddist[12], zeros)
        assert_dist_and_unknowns(ddist[13], [[0, 1, 1.1], [3, 1, 1]])
        assert_dist_and_unknowns(ddist[14], [[0, 1, 2], [3, 1, 1]])
        assert_dist_and_unknowns(ddist[15], zeros)
        assert_dist_and_unknowns(ddist[16], [[0, 1, 2], [3, 1, 1]])
        assert_dist_and_unknowns(ddist[17], [[0], [3]])
        assert_dist_and_unknowns(ddist[18], [[0, 2], [4, 1]])
        assert_dist_and_unknowns(ddist[19], zeros)

        with d.unlocked():
            d.set_weights(np.array([1, 2, 3, 4, 5]))
        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 20)
        assert_dist_and_unknowns(ddist[0], [15, 0, 0])
        assert_dist_and_unknowns(ddist[1], [14, 0, 1])
        assert_dist_and_unknowns(ddist[2], [8, 2, 5])
        assert_dist_and_unknowns(ddist[3], [9, 5, 1])
        assert_dist_and_unknowns(ddist[4], [12, 1, 0])
        assert_dist_and_unknowns(ddist[5], [9, 1, 3])
        assert_dist_and_unknowns(ddist[6], [4, 7, 1])
        assert_dist_and_unknowns(ddist[7], [15, 0, 0])
        assert_dist_and_unknowns(ddist[8], [13, 0, 2])
        assert_dist_and_unknowns(ddist[9], [14, 1, 0])

        zeros = [[0], [15]]
        assert_dist_and_unknowns(ddist[10], zeros)
        assert_dist_and_unknowns(ddist[11], zeros)
        assert_dist_and_unknowns(ddist[12], zeros)
        assert_dist_and_unknowns(ddist[13], [[0, 1, 1.1], [9, 1, 5]])
        assert_dist_and_unknowns(ddist[14], [[0, 1, 2], [12, 1, 2]])
        assert_dist_and_unknowns(ddist[15], zeros)
        assert_dist_and_unknowns(ddist[16], [[0, 1, 2], [12, 2, 1]])
        assert_dist_and_unknowns(ddist[17], [[0], [12]])
        assert_dist_and_unknowns(ddist[18], [[0, 2], [14, 1]])
        assert_dist_and_unknowns(ddist[19], zeros)

    def test_compute_distributions_metas(self):
        d = data.Table(test_filename("datasets/test9.tab"))
        variable = d.domain[-2]
        dist, _ = d._compute_distributions([variable])[0]
        assert_dist_equal(dist, [3, 3, 2])
        # repeat with nan values
        assert d.metas.dtype.kind == "O"
        assert d.metas[0, 1] == 0

        with d.unlocked():
            d.metas[0, 1] = np.nan
        dist, nanc = d._compute_distributions([variable])[0]
        assert_dist_equal(dist, [2, 3, 2])
        self.assertEqual(nanc, 1)


class TestContinuous(unittest.TestCase):
    def test_mean(self):
        # pylint: disable=bad-whitespace
        x = np.array([[0, 5, 10],
                      [9, 0,  1]])
        dist = distribution.Continuous(x)

        self.assertEqual(dist.mean(), np.mean(([0] * 9) + [10]))

    def test_variance(self):
        # pylint: disable=bad-whitespace
        x = np.array([[0, 5, 10],
                      [9, 0,  1]])
        dist = distribution.Continuous(x)

        self.assertEqual(dist.variance(), np.var(([0] * 9) + [10]))

    def test_standard_deviation(self):
        # pylint: disable=bad-whitespace
        x = np.array([[0, 5, 10],
                      [9, 0,  1]])
        dist = distribution.Continuous(x)

        self.assertEqual(dist.standard_deviation(), np.std(([0] * 9) + [10]))


if __name__ == "__main__":
    unittest.main()
