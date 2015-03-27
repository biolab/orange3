from math import isnan
import warnings
import unittest
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import \
    Instance, Domain, Unknown, Value, \
    DiscreteVariable, ContinuousVariable, StringVariable


class TestInstance(unittest.TestCase):
    attributes = ["Feature %i" % i for i in range(10)]
    class_vars = ["Class %i" % i for i in range(1)]
    metas = [DiscreteVariable("Meta 1", values="XYZ"),
             ContinuousVariable("Meta 2"),
             StringVariable("Meta 3")]

    def mock_domain(self, with_classes=False, with_metas=False):
        attributes = self.attributes
        class_vars = self.class_vars if with_classes else []
        metas = self.metas if with_metas else []
        variables = attributes + class_vars
        return MagicMock(Domain,
                         attributes=attributes,
                         class_vars=class_vars,
                         metas=metas,
                         variables=variables)

    def create_domain(self, attributes=(), classes=(), metas=()):
        attr_vars = [ContinuousVariable(name=a) if isinstance(a, str) else a
                     for a in attributes]
        class_vars = [ContinuousVariable(name=c) if isinstance(c, str) else c
                      for c in classes]
        meta_vars = [DiscreteVariable(name=m, values=map(str, range(5)))
                     if isinstance(m, str) else m
                     for m in metas]
        domain = Domain(attr_vars, class_vars, meta_vars)
        return domain

    def test_init_x_no_data(self):
        domain = self.mock_domain()
        inst = Instance(domain)
        self.assertIsInstance(inst, Instance)
        self.assertIs(inst.domain, domain)
        self.assertEqual(inst._x.shape, (len(self.attributes), ))
        self.assertEqual(inst._y.shape, (0, ))
        self.assertEqual(inst._metas.shape, (0, ))
        self.assertTrue(all(isnan(x) for x in inst._x))

    def test_init_xy_no_data(self):
        domain = self.mock_domain(with_classes=True)
        inst = Instance(domain)
        self.assertIsInstance(inst, Instance)
        self.assertIs(inst.domain, domain)
        self.assertEqual(inst._x.shape, (len(self.attributes), ))
        self.assertEqual(inst._y.shape, (len(self.class_vars), ))
        self.assertEqual(inst._metas.shape, (0, ))
        self.assertTrue(all(isnan(x) for x in inst._x))
        self.assertTrue(all(isnan(x) for x in inst._y))

    def test_init_xym_no_data(self):
        domain = self.mock_domain(with_classes=True, with_metas=True)
        inst = Instance(domain)
        self.assertIsInstance(inst, Instance)
        self.assertIs(inst.domain, domain)
        self.assertEqual(inst._x.shape, (len(self.attributes), ))
        self.assertEqual(inst._y.shape, (len(self.class_vars), ))
        self.assertEqual(inst._metas.shape, (3, ))
        self.assertTrue(all(isnan(x) for x in inst._x))
        self.assertTrue(all(isnan(x) for x in inst._y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert_array_equal(inst._metas, np.array([Unknown, Unknown, None]))

    def test_init_x_arr(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")])
        vals = np.array([42, 0])
        inst = Instance(domain, vals)
        assert_array_equal(inst._x, vals)
        self.assertEqual(inst._y.shape, (0, ))
        self.assertEqual(inst._metas.shape, (0, ))

        domain = self.create_domain()
        inst = Instance(domain, np.empty((0,)))
        self.assertEqual(inst._x.shape, (0, ))
        self.assertEqual(inst._y.shape, (0, ))
        self.assertEqual(inst._metas.shape, (0, ))


    def test_init_x_list(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")])
        lst = [42, 0]
        vals = np.array(lst)
        inst = Instance(domain, vals)
        assert_array_equal(inst._x, vals)
        self.assertEqual(inst._y.shape, (0, ))
        self.assertEqual(inst._metas.shape, (0, ))

        domain = self.create_domain()
        inst = Instance(domain, [])
        self.assertEqual(inst._x.shape, (0, ))
        self.assertEqual(inst._y.shape, (0, ))
        self.assertEqual(inst._metas.shape, (0, ))

    def test_init_xy_arr(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")])
        vals = np.array([42, 0, 1])
        inst = Instance(domain, vals)
        assert_array_equal(inst._x, vals[:2])
        self.assertEqual(inst._y.shape, (1, ))
        self.assertEqual(inst._y[0], 1)
        self.assertEqual(inst._metas.shape, (0, ))

    def test_init_xy_list(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")])
        lst = [42, "M", "C"]
        vals = np.array([42, 0, 2])
        inst = Instance(domain, vals)
        assert_array_equal(inst._x, vals[:2])
        self.assertEqual(inst._y.shape, (1, ))
        self.assertEqual(inst._y[0], 2)
        self.assertEqual(inst._metas.shape, (0, ))

    def test_init_xym_arr(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = np.array([42, "M", "B", "X", 43, "Foo"], dtype=object)
        inst = Instance(domain, vals)
        self.assertIsInstance(inst, Instance)
        self.assertIs(inst.domain, domain)
        self.assertEqual(inst._x.shape, (2, ))
        self.assertEqual(inst._y.shape, (1, ))
        self.assertEqual(inst._metas.shape, (3, ))
        assert_array_equal(inst._x, np.array([42, 0]))
        self.assertEqual(inst._y[0], 1)
        assert_array_equal(inst._metas, np.array([0, 43, "Foo"], dtype=object))

    def test_init_xym_list(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = [42, "M", "B", "X", 43, "Foo"]
        inst = Instance(domain, vals)
        self.assertIsInstance(inst, Instance)
        self.assertIs(inst.domain, domain)
        self.assertEqual(inst._x.shape, (2, ))
        self.assertEqual(inst._y.shape, (1, ))
        self.assertEqual(inst._metas.shape, (3, ))
        assert_array_equal(inst._x, np.array([42, 0]))
        self.assertEqual(inst._y[0], 1)
        assert_array_equal(inst._metas, np.array([0, 43, "Foo"], dtype=object))

    def test_init_inst(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = [42, "M", "B", "X", 43, "Foo"]
        inst = Instance(domain, vals)

        inst2 = Instance(domain, inst)
        assert_array_equal(inst2._x, np.array([42, 0]))
        self.assertEqual(inst2._y[0], 1)
        assert_array_equal(inst2._metas, np.array([0, 43, "Foo"], dtype=object))

        domain2 = self.create_domain(["z", domain[1], self.metas[1]],
                                     domain.class_vars,
                                     [self.metas[0], "w", domain[0]])
        inst2 = Instance(domain2, inst)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert_array_equal(inst2._x, np.array([Unknown, 0, 43]))
            self.assertEqual(inst2._y[0], 1)
            assert_array_equal(inst2._metas, np.array([0, Unknown, 42],
                                                      dtype=object))

    def test_get_item(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = [42, "M", "B", "X", 43, "Foo"]
        inst = Instance(domain, vals)

        val = inst[0]
        self.assertIsInstance(val, Value)
        self.assertEqual(inst[0], 42)
        self.assertEqual(inst["x"], 42)
        self.assertEqual(inst[domain[0]], 42)

        val = inst[1]
        self.assertIsInstance(val, Value)
        self.assertEqual(inst[1], "M")
        self.assertEqual(inst["g"], "M")
        self.assertEqual(inst[domain[1]], "M")

        val = inst[2]
        self.assertIsInstance(val, Value)
        self.assertEqual(inst[2], "B")
        self.assertEqual(inst["y"], "B")
        self.assertEqual(inst[domain.class_var], "B")

        val = inst[-2]
        self.assertIsInstance(val, Value)
        self.assertEqual(inst[-2], 43)
        self.assertEqual(inst["Meta 2"], 43)
        self.assertEqual(inst[self.metas[1]], 43)

        with self.assertRaises(ValueError):
            inst["asdf"] = 42
        with self.assertRaises(ValueError):
            inst[ContinuousVariable("asdf")] = 42

    def test_set_item(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = [42, "M", "B", "X", 43, "Foo"]
        inst = Instance(domain, vals)

        inst[0] = 43
        self.assertEqual(inst[0], 43)
        inst["x"] = 44
        self.assertEqual(inst[0], 44)
        inst[domain[0]] = 45
        self.assertEqual(inst[0], 45)

        inst[1] = "F"
        self.assertEqual(inst[1], "F")
        inst["g"] = "M"
        self.assertEqual(inst[1], "M")
        with self.assertRaises(ValueError):
            inst[1] = "N"
        with self.assertRaises(ValueError):
            inst["asdf"] = 42

        inst[2] = "C"
        self.assertEqual(inst[2], "C")
        inst["y"] = "A"
        self.assertEqual(inst[2], "A")
        inst[domain.class_var] = "B"
        self.assertEqual(inst[2], "B")

        inst[-1] = "Y"
        self.assertEqual(inst[-1], "Y")
        inst["Meta 1"] = "Z"
        self.assertEqual(inst[-1], "Z")
        inst[domain.metas[0]] = "X"
        self.assertEqual(inst[-1], "X")

    def test_str(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")])
        inst = Instance(domain, [42, 0])
        self.assertEqual(str(inst), "[42.000, M]")

        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")])
        inst = Instance(domain, [42, "M", "B"])
        self.assertEqual(str(inst), "[42.000, M | B]")

        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        inst = Instance(domain, [42, "M", "B", "X", 43, "Foo"])
        self.assertEqual(str(inst), "[42.000, M | B] {X, 43.000, Foo}")

        domain = self.create_domain([],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        inst = Instance(domain, ["B", "X", 43, "Foo"])
        self.assertEqual(str(inst), "[ | B] {X, 43.000, Foo}")

        domain = self.create_domain([],
                                    [],
                                    self.metas)
        inst = Instance(domain, ["X", 43, "Foo"])
        self.assertEqual(str(inst), "[] {X, 43.000, Foo}")

        domain = self.create_domain(self.attributes)
        inst = Instance(domain, range(len(self.attributes)))
        self.assertEqual(
            str(inst),
            "[{}]".format(", ".join("{:.3f}".format(x)
                                    for x in range(len(self.attributes)))))

        for attr in domain:
            attr.number_of_decimals = 0
        self.assertEqual(
            str(inst),
            "[{}]".format(", ".join("{}".format(x)
                                    for x in range(len(self.attributes)))))

    def test_repr(self):
        domain = self.create_domain(self.attributes)
        inst = Instance(domain, range(len(self.attributes)))
        self.assertEqual(repr(inst), "[0.000, 1.000, 2.000, 3.000, 4.000, ...]")

        for attr in domain:
            attr.number_of_decimals = 0
        self.assertEqual(repr(inst), "[0, 1, 2, 3, 4, ...]")

    def test_eq(self):
        domain = self.create_domain(["x", DiscreteVariable("g", values="MF")],
                                    [DiscreteVariable("y", values="ABC")],
                                    self.metas)
        vals = [42, "M", "B", "X", 43, "Foo"]
        inst = Instance(domain, vals)
        inst2 = Instance(domain, vals)
        self.assertTrue(inst == inst2)
        self.assertTrue(inst2 == inst)

        inst2[0] = 43
        self.assertFalse(inst == inst2)

        inst2[0] = Unknown
        self.assertFalse(inst == inst2)

        inst2 = Instance(domain, vals)
        inst2[2] = "C"
        self.assertFalse(inst == inst2)

        inst2 = Instance(domain, vals)
        inst2[-1] = "Y"
        self.assertFalse(inst == inst2)

        inst2 = Instance(domain, vals)
        inst2[-2] = "33"
        self.assertFalse(inst == inst2)

        inst2 = Instance(domain, vals)
        inst2[-3] = "Bar"
        self.assertFalse(inst == inst2)

