# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

import Orange
from Orange.data import Domain, Table, DiscreteVariable
from Orange.preprocess import RemoveNaNClasses


class TestPreprocess(unittest.TestCase):
    def test_read_data_calls_reader(self):
        class MockPreprocessor(Orange.preprocess.preprocess.Preprocess):
            __init__ = Mock(return_value=None)
            __call__ = Mock()
            @classmethod
            def reset(cls):
                cls.__init__.reset_mock()
                cls.__call__.reset_mock()

        table = Mock(Orange.data.Table)
        MockPreprocessor(table, 1, 2, a=3)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        MockPreprocessor.__call__.assert_called_with(table)
        MockPreprocessor.reset()

        MockPreprocessor(1, 2, a=3)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor(a=3)
        MockPreprocessor.__init__.assert_called_with(a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor()
        MockPreprocessor.__init__.assert_called_with()
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)


class RemoveConstant(unittest.TestCase):
    def test_remove_columns(self):
        X = np.random.rand(6, 4)
        X[:, (1,3)] = 5
        X[3, 1] = np.nan
        X[1, 1] = np.nan
        data = Orange.data.Table(X)
        d = Orange.preprocess.preprocess.RemoveConstant(data)
        self.assertEqual(len(d.domain.attributes), 2)

        pp_rc = Orange.preprocess.preprocess.RemoveConstant()
        d = pp_rc(data)
        self.assertEqual(len(d.domain.attributes), 2)

    def test_nothing_to_remove(self):
        data = Orange.data.Table("iris")
        d = Orange.preprocess.preprocess.RemoveConstant(data)
        self.assertEqual(len(d.domain.attributes), 4)


class TestRemoveNanClass(unittest.TestCase):
    def test_remove_nan_classes(self):
        table = Table("imports-85")
        self.assertTrue(np.isnan(table.Y).any())
        table = RemoveNaNClasses(table)
        self.assertTrue(not np.isnan(table.Y).any())

    def test_remove_nan_classes_multiclass(self):
        domain = Domain([DiscreteVariable("a", values="01")],
                        [DiscreteVariable("b", values="01"),
                        DiscreteVariable("c", values="01")])
        table = Table(domain, [[0, 1, np.nan],
                               [1, np.nan, 0],
                               [1, 0, 1],
                               [1, np.nan, np.nan]])
        table = RemoveNaNClasses(table)
        self.assertTrue(not np.isnan(table).any())
        self.assertEqual(table.domain, domain)
        self.assertEqual(len(table), 1)
