import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data.util import scale, one_hot, SharedComputeValue
import Orange

class TestDataUtil(unittest.TestCase):
    def test_scale(self):
        np.testing.assert_equal(scale([0, 1, 2], -1, 1), [-1, 0, 1])
        np.testing.assert_equal(scale([3, 3, 3]), [1, 1, 1])
        np.testing.assert_equal(scale([.1, .5, np.nan]), [0, 1, np.nan])

    def test_one_hot(self):
        np.testing.assert_equal(
            one_hot([0, 1, 2, 1], int), [[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],
                                         [0, 1, 0]])


class DummyPlus(SharedComputeValue):

    def compute(self, data, shared_data):
        return data.X[:,0] + shared_data


class DummyTable(Orange.data.Table): pass


class TestSharedComputeValue(unittest.TestCase):

    def test_compat_compute_value(self):
        data = Orange.data.Table("iris")
        obj = DummyPlus(lambda data: 1.)
        res = obj(data)
        obj = lambda data: data.X[:,0] + 1.
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
        data = Orange.data.Table("iris")
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
