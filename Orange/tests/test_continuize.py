import unittest

from Orange.data import Table, Variable, ContinuousVariable
from Orange.preprocess.continuize import DomainContinuizer
from Orange.preprocess import Continuize
from Orange.preprocess import transformation


class ContinuizerTest(unittest.TestCase):
    def setUp(self):
        Variable._clear_all_caches()
        self.data = Table("test4")

    def test_default(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
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
            #                          c1 c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, "c"])

    def test_continuous(self):
        self.assertRaises(TypeError, DomainContinuizer, self.data.domain,
                          normalize_continuous=Continuize.NormalizeBySpan)

        domZB = DomainContinuizer(
            self.data, normalize_continuous=Continuize.NormalizeBySpan,
            zero_based=True)

        dom = DomainContinuizer(
            self.data, normalize_continuous=Continuize.NormalizeBySpan,
            zero_based=False)

        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in domZB.attributes))
        self.assertIs(domZB.class_var, self.data.domain.class_var)
        self.assertIsNot(domZB[0], self.data.domain[0])
        self.assertIsNot(domZB[1], self.data.domain[1])
        self.assertEqual([attr.name for attr in domZB.attributes],
                         ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                          "d3=c"])
        self.assertIsInstance(domZB[2].compute_value, transformation.Indicator)

        dat2 = Table(domZB, self.data)
        #                          c1   c2  d2    d3       cl1
        self.assertEqual(dat2[0], [0.5,  0, 1, 0, 1, 0, 0, "a"])
        self.assertEqual(dat2[1], [0,  0.5, 0, 1, 0, 1, 0, "b"])
        self.assertEqual(dat2[2], [1,    1, 0, 1, 0, 0, 1, "c"])

        self.assertIs(dom.class_var, self.data.domain.class_var)
        self.assertIsNot(dom[0], self.data.domain[0])
        self.assertIsNot(dom[1], self.data.domain[1])
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                          "d3=c"])
        self.assertIsInstance(dom[2].compute_value,
                              transformation.Indicator1)

        dat3 = Table(dom, self.data)
        #                          c1   c2  d2      d3          cl1
        self.assertEqual(dat3[0], [0,   -1,  1, -1,  1, -1, -1, "a"])
        self.assertEqual(dat3[1], [-1,   0, -1,  1, -1,  1, -1, "b"])
        self.assertEqual(dat3[2], [1,    1, -1,  1, -1, -1,  1, "c"])

    def test_continuous_by_standard_deviation(self):
        self.assertRaises(TypeError, DomainContinuizer, self.data.domain,
                          normalize_continuous=Continuize.NormalizeBySD)

        domZB = DomainContinuizer(
            self.data, normalize_continuous=Continuize.NormalizeBySD,
            zero_based=True)
        dom = DomainContinuizer(
            self.data, normalize_continuous=Continuize.NormalizeBySD,
            zero_based=False)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in domZB.attributes))
        self.assertIs(domZB.class_var, self.data.domain.class_var)
        self.assertIsNot(domZB[0], self.data.domain[0])
        self.assertIsNot(domZB[1], self.data.domain[1])
        self.assertEqual([attr.name for attr in domZB.attributes],
                         ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                          "d3=c"])
        self.assertIsInstance(domZB[2].compute_value,
                              transformation.Indicator)

        solution = [[ 0,    -1.225,  1, 0, 1, 0, 0, 0],
                    [-1.225, 0,      0, 1, 0, 1, 0, 1],
                    [ 1.225, 1.225,  0, 1, 0, 0, 1, 2]]
        # I'm sorry about that, but checking whole rows with assertEqual doesn't work here
        # because of the rounding errors I guess
        dat2 = Table(domZB, self.data)
        for rd,rs in zip(dat2, solution):
            for x,y in zip(rd,rs):
                self.assertAlmostEqual(x,y, places=3)

        self.assertIsInstance(dom[2].compute_value,
                              transformation.Indicator1)

        dat3 = Table(dom, self.data)
        self.assertEqual(list(dat2.X[0,:2]),list(dat3.X[0,:2]))
        self.assertEqual(list(dat2.X[1,:2]),list(dat3.X[1,:2]))
        self.assertEqual(list(dat2.X[2,:2]),list(dat3.X[2,:2]))

    def test_continuous_transform_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, transform_class=True)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom))
            self.assertIsNot(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                              "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1   c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, 1, 0, 0])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, 0, 1, 0])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, 0, 0, 1])

    def test_continuous_transform_class_minus_one(self):
        self.assertRaises(TypeError, DomainContinuizer,
                        self.data.domain, normalize_continuous=True)

        dom = DomainContinuizer(
            self.data,
            normalize_continuous=Continuize.NormalizeBySpan,
            transform_class=True, zero_based=False)

        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom))
        self.assertIsNot(dom.class_var, self.data.domain.class_var)
        self.assertIsNot(dom[0], self.data.domain[0])
        self.assertIsNot(dom[1], self.data.domain[1])
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                          "d3=c"])
        self.assertEqual([attr.name for attr in dom.class_vars],
                         ["cl1=a", "cl1=b", "cl1=c"])
        self.assertIsInstance(dom[2].compute_value,
                              transformation.Indicator1)

        dat2 = Table(dom, self.data)
        #                          c1   c2  d2      d3         cl1
        self.assertEqual(dat2[0], [0,   -1,  1, -1,  1, -1, -1,  1, -1, -1])
        self.assertEqual(dat2[1], [-1,   0, -1,  1, -1,  1, -1, -1,  1, -1])
        self.assertEqual(dat2[2], [1,    1, -1,  1, -1, -1,  1, -1, -1,  1])

    def test_multi_indicators(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp,
                                    multinomial_treatment=Continuize.Indicators)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
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
            #                          c1 c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, "c"])

    def test_multi_lowest_base(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.FirstAsBase)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3     cl1
            self.assertEqual(dat2[0], [1, -2, 0, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 0, 1, "c"])

    def test_multi_lowest_base_base(self):
        self.data.domain[4].base_value=1
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.FirstAsBase)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=a", "d3=c"])
            self.assertIsInstance(dom[2].compute_value,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3    cl1
            self.assertEqual(dat2[0], [1, -2, 0, 1, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 0, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 0, 1, "c"])

    def test_multi_ignore(self):
        dom = DomainContinuizer(self.data.domain,
                                multinomial_treatment=Continuize.Remove)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(self.data.domain,
                                multinomial_treatment=Continuize.Remove,
                                transform_class=True)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_ignore_multi(self):
        dom = DomainContinuizer(
            self.data.domain,
            multinomial_treatment=Continuize.RemoveMultinomial)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom],
                         ["c1", "c2", "d2=b", "cl1"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(
            self.data.domain,
            multinomial_treatment=Continuize.RemoveMultinomial,
            transform_class=True)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
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
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.AsOrdinal)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3  cl1
            self.assertEqual(dat2[0], [1, -2, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 1, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 2, "c"])

    def test_as_ordinal_class(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.AsOrdinal,
                transform_class=True)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIsInstance(dom.class_var, ContinuousVariable)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3  cl1
            self.assertEqual(dat2[0], [1, -2, 0, 0, 0])
            self.assertEqual(dat2[1], [0,  0, 1, 1, 1])
            self.assertEqual(dat2[2], [2,  2, 1, 2, 2])

    def test_as_normalized_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(
                inp, multinomial_treatment=Continuize.AsNormalizedOrdinal)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom],
                             ["c1", "c2", "d2", "d3", "cl1"])

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3  cl1
            self.assertEqual(dat2[0], [1, -2, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 0.5, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 1, "c"])
