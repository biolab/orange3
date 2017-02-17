# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from collections import defaultdict

import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, StringVariable
from Orange.tests import test_filename
from Orange.widgets.data.owmergedata import group_table_indices, OWMergeData
from Orange.widgets.tests.base import WidgetTest


class TestOWMergeData(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        domainA = Domain([DiscreteVariable("d1", values=("a", "b"))],
                         DiscreteVariable("c1", values=("aaa", "bbb")),
                         [DiscreteVariable("m1", values=("aa", "bb"))])
        XA = np.array([[0], [1], [0], [np.nan]])
        yA = np.array([0, 1, 0, 1])
        metasA = np.array([[0], [1], [np.nan], [1]]).astype(object)

        domainB = Domain([DiscreteVariable("d2", values=("a", "c", "d"))],
                         DiscreteVariable("c2", values=("bbb", "ddd")),
                         [StringVariable("m2")])
        XB = np.array([[1], [2], [1], [np.nan], [0]])
        yB = np.array([0, 0, np.nan, 1, 1])
        metasB = np.array([["cc"], [""], ["bb"], ["bb"], ["cc"]]).astype(object)
        cls.dataA = Table(domainA, XA, yA, metasA)
        cls.dataB = Table(domainB, XB, yB, metasB)

    def setUp(self):
        self.widget = self.create_widget(OWMergeData)

    def test_group_table_indices(self):
        table = Table(test_filename("test9.tab"))
        dd = defaultdict(list)
        dd["1"] = [0, 1]
        dd["huh"] = [2]
        dd["hoy"] = [3]
        dd["?"] = [4]
        dd["2"] = [5]
        dd["oh yeah"] = [6]
        dd["3"] = [7]
        self.assertEqual(dd, group_table_indices(table, "g"))

    def test_output_merge_by_ids_inner(self):
        """Check output for merging only matching values by IDs"""
        domain = self.dataA.domain
        result = Table(domain, np.array([[1], [0]]), np.array([1, 0]),
                       np.array([[1.0], [np.nan]]).astype(object))
        self.send_signal("Data A", self.dataA[:3, [0, -1]])
        self.send_signal("Data B", self.dataA[1:, [0, "c1"]])
        self.widget.attr_a = "Source position (index)"
        self.widget.attr_b = "Source position (index)"
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_ids_outer(self):
        """Check output for merging all values by IDs"""
        self.send_signal("Data A", self.dataA[:, [0, -1]])
        self.send_signal("Data B", self.dataA[:, [0, "c1"]])
        self.widget.attr_a = "Source position (index)"
        self.widget.attr_b = "Source position (index)"
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), self.dataA)

    def test_output_merge_by_index_inner_AB(self):
        """Check output for merging only matching values by row index"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 1], [1, 2], [0, 1], [np.nan, np.nan]])
        result_Y = np.array([[0, 0], [1, 0], [0, np.nan], [1, 1]])
        result_M = np.array([[0.0, "cc"], [1.0, ""], [np.nan, "bb"],
                             [1.0, "bb"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_index_inner_BA(self):
        """Check output for merging only matching values by row index"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, 0], [2, 1], [1, 0], [np.nan, np.nan]])
        result_Y = np.array([[0, 0], [0, 1], [np.nan, 0], [1, 1]])
        result_M = np.array([["cc", 0.0], ["", 1.0], ["bb", np.nan],
                             ["bb", 1.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_index_outer_AB(self):
        """Check output for merging all values by row index"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 1], [1, 2], [0, 1],
                             [np.nan, np.nan], [np.nan, 0]])
        result_Y = np.array([[0, 0], [1, 0], [0, np.nan], [1, 1], [np.nan, 1]])
        result_M = np.array([[0.0, "cc"], [1.0, ""], [np.nan, "bb"],
                             [1.0, "bb"], [np.nan, "cc"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_index_outer_BA(self):
        """Check output for merging all values by row index"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, 0], [2, 1], [1, 0],
                             [np.nan, np.nan], [0, np.nan]])
        result_Y = np.array([[0, 0], [0, 1], [np.nan, 0], [1, 1], [1, np.nan]])
        result_M = np.array([["cc", 0.0], ["", 1.0], ["bb", np.nan],
                             ["bb", 1.0], ["cc", np.nan]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_attribute_inner_AB(self):
        """Check output for merging only matching values by attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0], [0]])
        result_Y = np.array([[0, 1], [0, 1]])
        result_M = np.array([[0.0, "cc"], [np.nan, "cc"]])
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA[0]
        self.widget.attr_b = domainB[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_attribute_inner_BA(self):
        """Check output for merging only matching values by attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[0]])
        result_Y = np.array([[1, 0]])
        result_M = np.array([["cc", 0.0]])
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB[0]
        self.widget.attr_b = domainA[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_attribute_outer_AB(self):
        """Check output for merging all values by attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0], [1, np.nan], [0, 0], [np.nan, np.nan],
                             [np.nan, 1], [np.nan, 2], [np.nan, 1],
                             [np.nan, np.nan]])
        result_Y = np.array([[0, 1], [1, np.nan], [0, 1], [1, np.nan],
                             [np.nan, 0], [np.nan, 0], [np.nan, np.nan],
                             [np.nan, 1]])
        result_M = np.array([[0.0, "cc"], [1.0, ""], [np.nan, "cc"], [1.0, ""],
                             [np.nan, "cc"], [np.nan, ""], [np.nan, "bb"],
                             [np.nan, "bb"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA[0]
        self.widget.attr_b = domainB[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_attribute_outer_BA(self):
        """Check output for merging all values by attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, np.nan], [2, np.nan], [1, np.nan],
                             [np.nan, np.nan], [0, 0], [np.nan, 1],
                             [np.nan, np.nan]])
        result_Y = np.array([[0, np.nan], [0, np.nan], [np.nan, np.nan],
                             [1, np.nan], [1, 0], [np.nan, 1], [np.nan, 1]])
        result_M = np.array([["cc", np.nan], ["", np.nan], ["bb", np.nan],
                             ["bb", np.nan], ["cc", 0.0], ["", 1.0],
                             ["", 1.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB[0]
        self.widget.attr_b = domainA[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_class_inner_AB(self):
        """Check output for merging only matching values by class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[1, 1], [np.nan, 1]])
        result_Y = np.array([1, 1])
        result_M = np.array([[1.0, "cc"], [1.0, "cc"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA.class_vars[0]
        self.widget.attr_b = domainB.class_vars[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_class_inner_BA(self):
        """Check output for merging only matching values by class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, 1], [2, 1]])
        result_Y = np.array([0, 0])
        result_M = np.array([["cc", 1.0], ["", 1.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB.class_vars[0]
        self.widget.attr_b = domainA.class_vars[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_class_outer_AB(self):
        """Check output for merging all values by class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, np.nan], [1, 1], [0, np.nan], [np.nan, 1],
                             [np.nan, 1], [np.nan, np.nan], [np.nan, 0]])
        result_Y = np.array([[0, np.nan], [1, 0], [0, np.nan], [1, 0],
                             [np.nan, np.nan], [np.nan, 1], [np.nan, 1]])
        result_M = np.array([[0.0, ""], [1.0, "cc"], [np.nan, ""], [1.0, "cc"],
                             [np.nan, "bb"], [np.nan, "bb"],
                             [np.nan, "cc"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA.class_vars[0]
        self.widget.attr_b = domainB.class_vars[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_class_outer_BA(self):
        """Check output for merging all values by class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, 1], [2, 1], [1, np.nan], [np.nan, np.nan],
                             [0, np.nan], [np.nan, 0], [np.nan, 0]])
        result_Y = np.array([[0, 1], [0, 1], [np.nan, np.nan], [1, np.nan],
                             [1, np.nan], [np.nan, 0], [np.nan, 0]])
        result_M = np.array([["cc", 1.0], ["", 1.0], ["bb", np.nan],
                             ["bb", np.nan], ["cc", np.nan], ["", 0.0],
                             ["", np.nan]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB.class_vars[0]
        self.widget.attr_b = domainA.class_vars[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_meta_inner_AB(self):
        """Check output for merging only matching values by meta attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas)
        result_X = np.array([[1, 1], [np.nan, 1]])
        result_Y = np.array([[1, np.nan], [1, np.nan]])
        result_M = np.array([[1.0], [1.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA.metas[0]
        self.widget.attr_b = domainB.metas[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_meta_inner_BA(self):
        """Check output for merging only matching values by meta attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas)
        result_X = np.array([[1, 1], [np.nan, 1]])
        result_Y = np.array([[np.nan, 1], [1, 1]])
        result_M = np.array([["bb"], ["bb"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB.metas[0]
        self.widget.attr_b = domainA.metas[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_meta_outer_AB(self):
        """Check output for merging all values by meta attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain

        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, np.nan], [1, 1], [0, np.nan], [np.nan, 1],
                             [np.nan, 1], [np.nan, 2], [np.nan, 0]])
        result_Y = np.array([[0, np.nan], [1, np.nan], [0, np.nan], [1, np.nan],
                             [np.nan, 0], [np.nan, 0], [np.nan, 1]])
        result_M = np.array([[0.0, ""], [1.0, "bb"], [np.nan, ""], [1.0, "bb"],
                             [np.nan, "cc"], [np.nan, ""],
                             [np.nan, "cc"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataA)
        self.send_signal("Data B", self.dataB)
        self.widget.attr_a = domainA.metas[0]
        self.widget.attr_b = domainB.metas[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def test_output_merge_by_meta_outer_BA(self):
        """Check output for merging all values by meta attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainB.attributes + domainA.attributes,
                          domainB.class_vars + domainA.class_vars,
                          domainB.metas + domainA.metas)
        result_X = np.array([[1, np.nan], [2, np.nan], [1, 1], [np.nan, 1],
                             [0, np.nan], [np.nan, 0], [np.nan, 0]])
        result_Y = np.array([[0, np.nan], [0, np.nan], [np.nan, 1], [1, 1],
                             [1, np.nan], [np.nan, 0], [np.nan, 0]])
        result_M = np.array([["cc", np.nan], ["", np.nan], ["bb", 1.0],
                             ["bb", 1.0], ["cc", np.nan], ["", 0.0],
                             ["", np.nan]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal("Data A", self.dataB)
        self.send_signal("Data B", self.dataA)
        self.widget.attr_a = domainB.metas[0]
        self.widget.attr_b = domainA.metas[0]
        self.widget.controls.inner.setChecked(False)
        self.assertTablesEqual(self.get_output("Merged Data"), result)

    def assertTablesEqual(self, table1, table2):
        self.assertEqual(table1.domain, table2.domain)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas.astype(str),
                                      table2.metas.astype(str))
