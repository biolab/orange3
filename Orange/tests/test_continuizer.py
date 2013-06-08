from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.data.continuizer import DomainContinuizer
from Orange.feature import transformation
import unittest


class Continuizer_Test(unittest.TestCase):
    def setUp(self):
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
            self.assertIsInstance(dom[2].get_value_from,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, "c"])

    def test_continuous(self):
        self.assertRaises(TypeError, DomainContinuizer,
                          self.data.domain, normalize_continuous=True)

        dom = DomainContinuizer(self.data, normalize_continuous=True)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertIs(dom.class_var, self.data.domain.class_var)
        self.assertIsNot(dom[0], self.data.domain[0])
        self.assertIsNot(dom[1], self.data.domain[1])
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                          "d3=c"])
        self.assertIsInstance(dom[2].get_value_from,
                              transformation.Indicator)

        dat2 = Table(dom, self.data)
        #                          c1   c2  d2    d3       cl1
        self.assertEqual(dat2[0], [0.5,  0, 1, 0, 1, 0, 0, "a"])
        self.assertEqual(dat2[1], [0,  0.5, 0, 1, 0, 1, 0, "b"])
        self.assertEqual(dat2[2], [1,    1, 0, 1, 0, 0, 1, "c"])

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
            self.assertIsInstance(dom[2].get_value_from,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1   c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, 1, 0, 0])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, 0, 1, 0])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, 0, 0, 1])


    def test_continuous_transform_class_minus_one(self):
        self.assertRaises(TypeError, DomainContinuizer,
                  self.data.domain, normalize_continuous=True)

        dom = DomainContinuizer(self.data, normalize_continuous=True,
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
        self.assertIsInstance(dom[2].get_value_from,
                              transformation.Indicator_1)

        dat2 = Table(dom, self.data)
        #                          c1   c2  d2      d3         cl1
        self.assertEqual(dat2[0], [0,   -1,  1, -1,  1, -1, -1,  1, -1, -1])
        self.assertEqual(dat2[1], [-1,   0, -1,  1, -1,  1, -1, -1,  1, -1])
        self.assertEqual(dat2[2], [1,    1, -1,  1, -1, -1,  1, -1, -1,  1])

    def test_multi_nvalues(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.NValues)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=a", "d2=b", "d3=a", "d3=b",
                              "d3=c"])
            self.assertIsInstance(dom[2].get_value_from,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2    d3       cl1
            self.assertEqual(dat2[0], [1, -2, 1, 0, 1, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 0, 1, 0, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 0, 1, 0, 0, 1, "c"])

    def test_multi_lowest_base(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.LowestIsBase)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=b", "d3=c"])
            self.assertIsInstance(dom[2].get_value_from,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3     cl1
            self.assertEqual(dat2[0], [1, -2, 0, 0, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 1, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 0, 1, "c"])

    def test_multi_lowest_base_base(self):
        self.data.domain[4].base_value=1
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.LowestIsBase)
            self.assertTrue(all(isinstance(attr, ContinuousVariable)
                                for attr in dom.attributes))
            self.assertIs(dom.class_var, self.data.domain.class_var)
            self.assertIs(dom[0], self.data.domain[0])
            self.assertIs(dom[1], self.data.domain[1])
            self.assertEqual([attr.name for attr in dom.attributes],
                             ["c1", "c2", "d2=b", "d3=a", "d3=c"])
            self.assertIsInstance(dom[2].get_value_from,
                                  transformation.Indicator)

            dat2 = Table(dom, self.data)
            #                          c1 c2  d2 d3    cl1
            self.assertEqual(dat2[0], [1, -2, 0, 1, 0, "a"])
            self.assertEqual(dat2[1], [0,  0, 1, 0, 0, "b"])
            self.assertEqual(dat2[2], [2,  2, 1, 0, 1, "c"])


    def test_multi_ignore(self):
        dom = DomainContinuizer(self.data.domain, multinomial_treatment=
                                DomainContinuizer.Ignore)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(self.data.domain, multinomial_treatment=
                                DomainContinuizer.Ignore,
                                transform_class=True)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom.attributes],
                         ["c1", "c2"])
        self.assertEqual(len(dom.class_vars), 0)
        self.assertIsNone(dom.class_var)

    def test_multi_ignore_multi(self):
        dom = DomainContinuizer(self.data.domain, multinomial_treatment=
                                DomainContinuizer.IgnoreMulti)
        self.assertTrue(all(isinstance(attr, ContinuousVariable)
                            for attr in dom.attributes))
        self.assertEqual([attr.name for attr in dom],
                         ["c1", "c2", "d2=b", "cl1"])

    def test_multi_ignore_class(self):
        dom = DomainContinuizer(self.data.domain, multinomial_treatment=
                                DomainContinuizer.IgnoreMulti,
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
                          multinomial_treatment=DomainContinuizer.ReportError)

    def test_as_ordinal(self):
        for inp in (self.data, self.data.domain):
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.AsOrdinal)
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
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.AsOrdinal,
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
            dom = DomainContinuizer(inp, multinomial_treatment=
                                    DomainContinuizer.AsNormalizedOrdinal)
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

