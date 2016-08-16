# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table, Variable
from Orange.preprocess.continuize import DomainContinuizer
from Orange.preprocess import Continuize
from Orange.preprocess import transformation
from Orange.tests import test_filename


class TestDomainContinuizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Variable._clear_all_caches()
        cls.data = Table(test_filename("test4"))

    def test_default(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value, transformation.Indicator)

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2    d3       cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 1, 0, 1, 0, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 0, 1, 0, 1, 0, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 0, 1, 0, 0, 1, "c"], dtype=object))

    def test_continuous_transform_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, transform_class=True)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom))
            self.assertIsNot(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value, transformation.Indicator)

            dat2 = Table(dom, self.data)
            # __weights__ c1   c2  d2    d3       cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 1, 0, 1, 0, 0, 1, 0, 0], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 0, 1, 0, 0, 1, 0, 0, 1], dtype=object))

    def test_multi_indicators(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp,
                                    multinomial_treatment=Continuize.Indicators)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                              "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2    d3       cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 1, 0, 1, 0, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 0, 1, 0, 1, 0, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 0, 1, 0, 0, 1, "c"], dtype=object))

    def test_multi_lowest_base(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.FirstAsBase)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2 d3     cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 0, 0, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 1, 1, 0, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 1, 0, 1, "c"], dtype=object))

    def test_multi_lowest_base_base(self):
        self.data.domain[4].base_value = 1
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.FirstAsBase)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=a", "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2 d3    cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 0, 1, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 1, 0, 0, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 1, 0, 1, "c"], dtype=object))

    def test_multi_ignore(self):
        dom = DomainContinuizer(self.data.domain,
                                multinomial_treatment=Continuize.Remove)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(self.data.domain,
                                multinomial_treatment=Continuize.Remove,
                                transform_class=True)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_ignore_multi(self):
        dom = DomainContinuizer(
            self.data.domain,
            multinomial_treatment=Continuize.RemoveMultinomial)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom],
                         ["c1", "c2", "d2=b", "cl1"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(
            self.data.domain,
            multinomial_treatment=Continuize.RemoveMultinomial,
            transform_class=True)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2", "d2=b"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_error(self):
        self.assertRaises(ValueError, DomainContinuizer,
                          self.data.domain,
                          multinomial_treatment=Continuize.ReportError)

    def test_as_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, multinomial_treatment=Continuize.AsOrdinal)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2 d3  cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 0, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 1, 1, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 1, 2, "c"], dtype=object))

    def test_as_ordinal_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.AsOrdinal,
                transform_class=True)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertTrue(dom.has_continuous_class)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            # c1 c2  d2 d3  cl1
            np.testing.assert_array_equal(dat2.iloc[0], np.array([1, 1, -2, 0, 0, 0], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1], np.array([1, 0, 0, 1, 1, 1], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2], np.array([1, 2, 2, 1, 2, 2], dtype=object))

    def test_as_normalized_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.AsNormalizedOrdinal)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            # __weights__ c1 c2  d2 d3  cl1
            np.testing.assert_array_equal(dat2.iloc[0].values, np.array([1, 1, -2, 0, 0, "a"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[1].values, np.array([1, 0, 0, 1, 0.5, "b"], dtype=object))
            np.testing.assert_array_equal(dat2.iloc[2].values, np.array([1, 2, 2, 1, 1, "c"], dtype=object))
