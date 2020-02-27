# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table
from Orange.preprocess import Remove, discretize
from Orange.tests import test_filename


class TestRemover(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test8 = Table(test_filename('datasets/test8.tab'))

    def test_remove(self):
        data = Table("iris")[:5]
        attr_flags = sum([Remove.SortValues,
                          Remove.RemoveConstant,
                          Remove.RemoveUnusedValues])
        class_flags = sum([Remove.SortValues,
                           Remove.RemoveConstant,
                           Remove.RemoveUnusedValues])
        remover = Remove(attr_flags, class_flags)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["sepal length", "sepal width", "petal length"])
        self.assertEqual([c.name for c in new_data.domain.class_vars], [])
        self.assertDictEqual(attr_res,
                             {'removed': 1, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 1, 'reduced': 0, 'sorted': 0})

    def test_remove_constant_attr(self):
        data = self.test8
        remover = Remove(Remove.RemoveConstant)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        np.testing.assert_equal(new_data.X, np.hstack((data[:, 1],
                                                       data[:, 3])))
        np.testing.assert_equal(new_data.Y, data.Y)
        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["c0", "d0"])
        self.assertEqual([c.name for c in new_data.domain.class_vars],
                         ["cl1", "cl0", "cl3", "cl4"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [('4', '6')])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [('1', '2', '3'), ('2', )])
        self.assertDictEqual(attr_res,
                             {'removed': 2, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})

    def test_remove_constant_class(self):
        data = self.test8
        remover = Remove(class_flags=Remove.RemoveConstant)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        np.testing.assert_equal(new_data.X, data.X)
        np.testing.assert_equal(new_data.Y, np.hstack((data[:, 4],
                                                       data[:, 5])))
        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["c1", "c0", "d1", "d0"])
        self.assertEqual([c.name for c in new_data.domain.class_vars],
                         ["cl1", "cl0"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [('1', ), ('4', '6')])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [('1', '2', '3')])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 2, 'reduced': 0, 'sorted': 0})

    def test_remove_unused_values_attr(self):
        data = self.test8
        data = data[1:]
        remover = Remove(Remove.RemoveUnusedValues)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        np.testing.assert_equal(new_data.X, data.X)
        np.testing.assert_equal(new_data.Y, data.Y)
        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["c1", "c0", "d1", "d0"])
        self.assertEqual([c.name for c in new_data.domain.class_vars],
                         ["cl1", "cl0", "cl3", "cl4"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [('1', ), ('4', )])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [('1', '2', '3'), ('2', )])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 1, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})

    def test_remove_unused_values_class(self):
        data = self.test8
        data = data[:2]
        remover = Remove(class_flags=Remove.RemoveUnusedValues)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        for i in range(len(data)):
            for j in range(len(data[i])):
                self.assertEqual(new_data[i, j], data[i, j])

        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["c1", "c0", "d1", "d0"])
        self.assertEqual([c.name for c in new_data.domain.class_vars],
                         ["cl1", "cl0", "cl3", "cl4"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [('1', ), ('4', '6')])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [('2', '3'), ('2', )])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 1, 'sorted': 0})

    def test_remove_unused_values_metas(self):
        data = Table(test_filename("datasets/test9.tab"))
        subset = data[:4]
        res = Remove(attr_flags=Remove.RemoveUnusedValues,
                     meta_flags=Remove.RemoveUnusedValues)(subset)

        self.assertEqual(res.domain["b"].values, res.domain["c"].values)
        self.assertEqual(res.domain["d"].values, ("1", "2"))
        self.assertEqual(res.domain["f"].values, ('1', 'hey'))

    def test_remove_unused_values_attr_sparse(self):
        data = self.test8
        data = data[1:].to_sparse()
        remover = Remove(Remove.RemoveUnusedValues)
        new_data = remover(data)
        attr_res = remover.attr_results

        self.assertEqual((new_data.X != data.X).nnz, 0)
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [('1', ), ('4', )])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [('1', '2', '3'), ('2', )])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 1, 'sorted': 0})

    def test_remove_mapping(self):
        data = Table("iris")
        x = np.vstack((data.X[:50], data.X[100:]))
        y = np.hstack((data.Y[:50], data.Y[100:]))
        data = Table.from_numpy(data.domain, x, y)
        remover = Remove(class_flags=Remove.RemoveUnusedValues)
        cleaned = remover(data)
        np.testing.assert_array_equal(cleaned.Y[:50], 0)
        np.testing.assert_array_equal(cleaned.Y[50:], 1)

    def test_remove_mapping_after_compute_value(self):
        housing = Table("housing")
        method = discretize.EqualFreq(n=3)
        discretizer = discretize.DomainDiscretizer(
            discretize_class=True, method=method)
        domain = discretizer(housing)
        data = housing.transform(domain)
        val12 = np.nonzero(data.Y > 0)[0]
        data = data[val12]
        remover = Remove(class_flags=Remove.RemoveUnusedValues)
        cleaned = remover(data)
        np.testing.assert_equal(cleaned.Y, data.Y - 1)
