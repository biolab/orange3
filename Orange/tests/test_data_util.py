import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from Orange.data.util import scale, one_hot, SharedComputeValue, SubarrayComputeValue
import Orange

class TestDataUtil(unittest.TestCase):
    def test_scale(self):
        np.testing.assert_equal(scale([0, 1, 2], -1, 1), [-1, 0, 1])
        np.testing.assert_equal(scale([3, 3, 3]), [1, 1, 1])
        np.testing.assert_equal(scale([.1, .5, np.nan]), [0, 1, np.nan])
        np.testing.assert_equal(scale(np.array([])), np.array([]))

    def test_one_hot(self):
        np.testing.assert_equal(
            one_hot([0, 1, 2, 1], int), [[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],
                                         [0, 1, 0]])
        np.testing.assert_equal(one_hot([], int), np.zeros((0, 0), dtype=int))


class DummyPlus(SharedComputeValue):

    def compute(self, data, shared_data):
        return data.X[:, 0] + shared_data


class DummyTable(Orange.data.Table):
    pass


class TestSharedComputeValue(unittest.TestCase):

    def test_compat_compute_value(self):
        data = Orange.data.Table("iris")
        obj = DummyPlus(lambda data: 1.)
        res = obj(data)
        obj = lambda data: data.X[:, 0] + 1.
        res2 = obj(data)
        np.testing.assert_equal(res, res2)

    def test_with_row_indices(self):
        obj = DummyPlus(lambda data: 1.)
        data = Orange.data.Table("iris")
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("cv", compute_value=obj)])
        data1 = Orange.data.Table.from_table(domain, data)[:10]
        data2 = Orange.data.Table.from_table(domain, data, range(10))
        np.testing.assert_equal(data1.X, data2.X)

    def test_single_call(self):
        obj = DummyPlus(Mock(return_value=1))
        self.assertEqual(obj.compute_shared.call_count, 0)
        data = Orange.data.Table("iris")[45:55]  # two classes
        domain = Orange.data.Domain([at.copy(compute_value=obj)
                                     for at in data.domain.attributes],
                                    data.domain.class_vars)

        Orange.data.Table.from_table(domain, data)
        self.assertEqual(obj.compute_shared.call_count, 1)
        ndata = Orange.data.Table.from_table(domain, data)
        self.assertEqual(obj.compute_shared.call_count, 2)

        #the learner performs imputation
        c = Orange.classification.LogisticRegressionLearner()(ndata)
        self.assertEqual(obj.compute_shared.call_count, 2)
        c(data) #the new data should be converted with one call
        self.assertEqual(obj.compute_shared.call_count, 3)

        #test with descendants of table
        DummyTable.from_table(c.domain, data)
        self.assertEqual(obj.compute_shared.call_count, 4)

    def test_compute_shared_eq_warning(self):
        with warnings.catch_warnings(record=True) as warns:
            DummyPlus(compute_shared=lambda *_: 42)

            class Valid:
                def __eq__(self, other):
                    pass

                def __hash__(self):
                    pass

            DummyPlus(compute_shared=Valid())
            self.assertEqual(warns, [])

            class Invalid:
                pass

            DummyPlus(compute_shared=Invalid())
            self.assertNotEqual(warns, [])

        with warnings.catch_warnings(record=True) as warns:

            class MissingHash:
                def __eq__(self, other):
                    pass

            DummyPlus(compute_shared=MissingHash())
            self.assertNotEqual(warns, [])

        with warnings.catch_warnings(record=True) as warns:

            class MissingEq:
                def __hash__(self):
                    pass

            DummyPlus(compute_shared=MissingEq())
            self.assertNotEqual(warns, [])

        with warnings.catch_warnings(record=True) as warns:

            class Subclass(Valid):
                pass

            DummyPlus(compute_shared=Subclass())
            self.assertNotEqual(warns, [])

    def test_eq_hash(self):
        x = Orange.data.ContinuousVariable("x")
        y = Orange.data.ContinuousVariable("y")
        x2 = Orange.data.ContinuousVariable("x")
        assert x == x2
        assert hash(x) == hash(x2)
        assert x != y
        assert hash(x) != hash(y)

        c1 = SharedComputeValue(abs, x)
        c2 = SharedComputeValue(abs, x2)

        d = SharedComputeValue(abs, y)
        e = SharedComputeValue(len, x)

        self.assertNotEqual(c1, None)

        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

        self.assertNotEqual(c1, d)
        self.assertNotEqual(hash(c1), hash(d))

        self.assertNotEqual(c1, e)
        self.assertNotEqual(hash(c1), hash(e))


class DummyPlusSubarray(SubarrayComputeValue):
    pass


class TestSubarrayComputeValue(unittest.TestCase):

    def test_values(self):
        fn = lambda data, cols: data.X[:, cols] + 1
        cv1 = DummyPlusSubarray(fn, 1)
        cv2 = DummyPlusSubarray(fn, 3)
        iris = Orange.data.Table("iris")
        domain = Orange.data.Domain([
            Orange.data.ContinuousVariable("cv1", compute_value=cv1),
            Orange.data.ContinuousVariable("cv2", compute_value=cv2)
        ])
        data = iris.transform(domain)
        np.testing.assert_equal(iris.X[:, [1,3]] + 1, data.X)

    def test_with_row_indices(self):
        fn = lambda data, cols: data.X[:, cols] + 1
        cv = DummyPlusSubarray(fn, 1)
        iris = Orange.data.Table("iris")
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("cv", compute_value=cv)])
        data1 = Orange.data.Table.from_table(domain, iris)[10:20]
        data2 = Orange.data.Table.from_table(domain, iris, range(10, 20))
        np.testing.assert_equal(data1.X, data2.X)

    def test_single_call(self):
        fn = lambda data, cols: data.X[:, cols] + 1
        mockfn = Mock(side_effect=fn)
        cvs = [DummyPlusSubarray(mockfn, i) for i in range(4)]
        self.assertEqual(mockfn.call_count, 0)
        data = Orange.data.Table("iris")[45:55]  # two classes
        domain = Orange.data.Domain([at.copy(compute_value=cv)
                                     for at, cv in zip(data.domain.attributes, cvs)],
                                     data.domain.class_vars)

        assert cvs[0].compute_shared is mockfn

        Orange.data.Table.from_table(domain, data)
        self.assertEqual(mockfn.call_count, 1)
        ndata = Orange.data.Table.from_table(domain, data)
        self.assertEqual(mockfn.call_count, 2)

        np.testing.assert_equal(ndata.X, data.X + 1)

        # learner performs imputation
        c = Orange.classification.LogisticRegressionLearner()(ndata)
        self.assertEqual(mockfn.call_count, 2)
        c(data)  # new data should be converted with one call
        self.assertEqual(mockfn.call_count, 3)

        # test with descendants of table
        DummyTable.from_table(c.domain, data)
        self.assertEqual(mockfn.call_count, 4)
