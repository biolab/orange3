import unittest

import numpy as np
from Orange.data import Table
from Orange.preprocess import Remove


class TestRemover(unittest.TestCase):
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
        data = Table("test8.tab")
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
                          if a.is_discrete], [['4', '6']])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [['1', '2', '3'], ['2']])
        self.assertDictEqual(attr_res,
                             {'removed': 2, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})

    def test_remove_constant_class(self):
        data = Table("test8.tab")
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
                          if a.is_discrete], [['1'], ['4', '6']])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [['1', '2', '3']])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 2, 'reduced': 0, 'sorted': 0})

    def test_remove_unused_values_attr(self):
        data = Table("test8.tab")
        data = data[1:]
        remover = Remove(Remove.RemoveUnusedValues)
        new_data = remover(data)
        attr_res, class_res = remover.attr_results, remover.class_results

        np.testing.assert_equal(new_data.X, data.X)
        np.testing.assert_equal(new_data.Y, data.Y)
        self.assertEqual([a.name for a in new_data.domain.attributes],
                         ["c1", "c0", "d1", "R_d0"])
        self.assertEqual([c.name for c in new_data.domain.class_vars],
                         ["cl1", "cl0", "cl3", "cl4"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [['1'], ['4']])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [['1', '2', '3'], ['2']])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 1, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})

    def test_remove_unused_values_class(self):
        data = Table("test8.tab")
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
                         ["R_cl1", "cl0", "cl3", "cl4"])
        self.assertEqual([a.values for a in new_data.domain.attributes
                          if a.is_discrete], [['1'], ['4', '6']])
        self.assertEqual([c.values for c in new_data.domain.class_vars
                          if c.is_discrete], [['2', '3'], ['2']])
        self.assertDictEqual(attr_res,
                             {'removed': 0, 'reduced': 0, 'sorted': 0})
        self.assertDictEqual(class_res,
                             {'removed': 0, 'reduced': 1, 'sorted': 0})
