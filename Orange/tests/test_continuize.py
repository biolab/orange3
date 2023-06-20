# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table, Variable
from Orange.data.dask import DaskTable
from Orange.preprocess.continuize import DomainContinuizer
from Orange.preprocess import Continuize
from Orange.preprocess import transformation
from Orange.tests import test_filename
from Orange.tests.test_dasktable import temp_dasktable


class TestDomainContinuizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = Table(test_filename("datasets/test4"))

    def compare_tables(self, data, solution):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if type(solution[i][j]) == float:
                    self.assertAlmostEqual(data[i, j], solution[i][j], places=20)
                else:
                    self.assertEqual(data[i, j], solution[i][j])

    def test_default(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer()
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value, transformation.Indicator)

            dat2 = self.data.transform(dom)
            # c1 c2  d2    d3       cl1
            solution = [[1, -2, 1, 0, 1, 0, 0, "a"],
                        [0, 0, 0, 1, 0, 1, 0, "b"],
                        [2, 2, 0, 1, 0, 0, 1, "c"]]
            self.compare_tables(dat2, solution)

    def test_continuous_transform_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(transform_class=True)
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.variables))
            self.assertIsNot(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value, transformation.Indicator)

            dat2 = self.data.transform(dom)
            # c1   c2  d2    d3       cl1
            solution = [[1, -2, 1, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                        [2, 2, 0, 1, 0, 0, 1, 0, 0, 1]]
            self.compare_tables(dat2, solution)

    def test_multi_indicators(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(multinomial_treatment=Continuize.Indicators)
            dom = dom(inp)
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

            dat2 = self.data.transform(dom)
            # c1 c2  d2    d3       cl1
            solution = [[1, -2, 1, 0, 1, 0, 0, "a"],
                        [0, 0, 0, 1, 0, 1, 0, "b"],
                        [2, 2, 0, 1, 0, 0, 1, "c"]]
            self.compare_tables(dat2, solution)

    def test_multi_lowest_base(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(multinomial_treatment=Continuize.FirstAsBase)
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = self.data.transform(dom)
            # c1 c2  d2 d3     cl1
            solution = [[1, -2, 0, 0, 0, "a"],
                        [0, 0, 1, 1, 0, "b"],
                        [2, 2, 1, 0, 1, "c"]]
            self.compare_tables(dat2, solution)

    def test_multi_ignore(self):
        dom = DomainContinuizer(multinomial_treatment=Continuize.Remove)
        dom = dom(self.data.domain)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(multinomial_treatment=Continuize.Remove,
                                transform_class=True)
        dom = dom(self.data.domain)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_ignore_multi(self):
        dom = DomainContinuizer(
            multinomial_treatment=Continuize.RemoveMultinomial)
        dom = dom(self.data.domain)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.variables],
                         ["c1", "c2", "d2=b", "cl1"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(
            multinomial_treatment=Continuize.RemoveMultinomial,
            transform_class=True)
        dom = dom(self.data.domain)
        self.assertTrue(all(attr.is_continuous
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2", "d2=b"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_error(self):
        self.assertRaises(ValueError,
                          DomainContinuizer(
                              multinomial_treatment=Continuize.ReportError),
                          self.data.domain)

    def test_as_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(multinomial_treatment=Continuize.AsOrdinal)
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.variables],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = self.data.transform(dom)
            # c1 c2  d2 d3  cl1
            solution = [[1, -2, 0, 0, "a"],
                        [0, 0, 1, 1, "b"],
                        [2, 2, 1, 2, "c"]]
            self.compare_tables(dat2, solution)

    def test_as_ordinal_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(multinomial_treatment=Continuize.AsOrdinal,
                                    transform_class=True)
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertTrue(dom.has_continuous_class)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.variables],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = self.data.transform(dom)
            # c1 c2  d2 d3  cl1
            solution = [[1, -2, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [2, 2, 1, 2, 2]]
            self.compare_tables(dat2, solution)

    def test_as_normalized_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(multinomial_treatment=Continuize.AsNormalizedOrdinal)
            dom = dom(inp)
            self.assertTrue(all(attr.is_continuous
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.variables],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = self.data.transform(dom)
            # c1 c2  d2 d3  cl1
            solution = [[1, -2, 0, 0, "a"],
                        [0, 0, 1, 0.5, "b"],
                        [2, 2, 1, 1, "c"]]
            self.compare_tables(dat2, solution)


class TestDomainContinuizerDask(TestDomainContinuizer):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = temp_dasktable(cls.data)

    def compare_tables(self, data, solution):
        self.assertIsInstance(data, DaskTable)
        super().compare_tables(data.compute(), solution)  # .compute avoids warning
