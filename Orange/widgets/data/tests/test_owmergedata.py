# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from itertools import chain
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain, DiscreteVariable, StringVariable, \
    ContinuousVariable
from Orange.widgets.data.owmergedata import OWMergeData, INSTANCEID, INDEX
from Orange.widgets.tests.base import WidgetTest
from Orange.tests import test_filename


class TestOWMergeData(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        domainA = Domain([DiscreteVariable("dA1", ("a", "b", "c", "d")),
                          DiscreteVariable("dA2", ("aa", "bb"))],
                         DiscreteVariable("cls", ("aaa", "bbb", "ccc")),
                         [DiscreteVariable("mA1", ("cc", "dd")),
                          StringVariable("mA2")])
        XA = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        yA = np.array([0, 1, 2, np.nan])
        metasA = np.array([[0.0, "m1"], [1.0, "m2"], [np.nan, "m3"],
                           [0.0, "m4"]]).astype(object)

        domainB = Domain([DiscreteVariable("dB1", values=("a", "b", "c")),
                          DiscreteVariable("dB2", values=("aa", "bb"))],
                         DiscreteVariable("cls", values=("bbb", "ccc")),
                         [DiscreteVariable("mB1", ("m4", "m5"))])
        XB = np.array([[0, 0], [1, 1], [2, np.nan]])
        yB = np.array([np.nan, 1, 0])
        metasB = np.array([[np.nan], [1], [0]]).astype(object)
        cls.dataA = Table(domainA, XA, yA, metasA)
        cls.dataA.name = 'dataA'
        cls.dataA.attributes = 'dataA attributes'
        cls.dataB = Table(domainB, XB, yB, metasB)
        cls.dataB.name = 'dataB'
        cls.dataB.attributes = 'dataB attributes'

    def setUp(self):
        self.widget = self.create_widget(OWMergeData)

    def test_input_remove(self):
        """Check widget after inputs have been removed"""
        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.extra_data, None)

    def test_combobox_items_inner(self):
        """Check if combo box content is properly set"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        row = self.widget.attr_boxes.rows[0]
        data_combo, extra_combo = row.left_combo, row.right_combo

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        data_items = extra_items = list(
            chain([INDEX, INSTANCEID], domainA.variables, domainA.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        data_items = list(
            chain([INDEX, INSTANCEID], domainA.variables, domainA.metas))
        extra_items = list(
            chain([INDEX, INSTANCEID], domainB.variables, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.data, self.dataB)
        data_items = extra_items = list(
            chain([INDEX, INSTANCEID], domainB.variables, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

    def test_output_merge_by_ids_inner(self):
        """Check output for merging option 'Find matching rows' by
        Source position (index)"""
        domain = self.dataA.domain
        result = Table(domain, np.array([[1, 1], [2, 0]]), np.array([1, 2]),
                       np.array([[1.0, "m2"], [np.nan, "m3"]]).astype(object))
        self.send_signal(self.widget.Inputs.data, self.dataA[:3, [0, "cls", -1]])
        self.send_signal(self.widget.Inputs.extra_data, self.dataA[1:, [1, "cls", -2]])
        self.widget.attr_boxes.set_state([(INSTANCEID, INSTANCEID)])
        self.widget.controls.merging.buttons[self.widget.InnerJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_ids_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        Source position (index)"""
        domain = self.dataA.domain
        result = Table(domain,
                       np.array([[0, np.nan], [1, 1], [2, 0], [np.nan, 1]]),
                       np.array([0, 1, 2, np.nan]),
                       np.array([[0.0, ""], [1.0, "m2"], [np.nan, "m3"],
                                 [np.nan, "m4"]]).astype(object))
        self.send_signal(self.widget.Inputs.data, self.dataA[:3, [0, "cls", -1]])
        self.send_signal(self.widget.Inputs.extra_data, self.dataA[1:, [1, "cls", -2]])
        self.widget.attr_boxes.set_state([(INSTANCEID, INSTANCEID)])
        self.widget.controls.merging.buttons[self.widget.OuterJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_index_left(self):
        """Check output for merging option 'Append columns from Extra Data' by
        Position (index)"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                             [2, 0, 2, np.nan], [3, 1, np.nan, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0], [np.nan, np.nan]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0], [0.0, "m4", np.nan]
                            ]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_index_inner(self):
        """Check output for merging option 'Find matching rows' by
        Position (index)"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 0, 2, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.controls.merging.buttons[1].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_index_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        Position (index)"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                             [2, 0, 2, np.nan], [3, 1, np.nan, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0], [np.nan, np.nan]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0], [0.0, "m4", np.nan]]
                           ).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.controls.merging.buttons[2].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_attribute_left(self):
        """Check output for merging option 'Append columns from Extra Data' by
        attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes[1:],
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0], [1, 1, 1],
                             [2, 0, np.nan], [3, 1, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0], [np.nan, np.nan]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0], [0.0, "m4", np.nan]]
                           ).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[0], domainB[0])])
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertTablesEqual(output, result)
        self.assertEqual(output.name, self.dataA.name)
        np.testing.assert_array_equal(output.ids, self.dataA.ids)
        self.assertEqual(output.attributes, self.dataA.attributes)

    def test_output_merge_by_attribute_inner(self):
        """Check output for merging option 'Find matching rows' by attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes[1:],
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0], [1, 1, 1], [2, 0, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[0], domainB[0])])
        self.widget.controls.merging.buttons[self.widget.InnerJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_attribute_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        attribute"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                             [2, 0, 2, np.nan], [3, 1, np.nan, np.nan]])
        result_Y = np.array([[0, np.nan], [1, 1], [2, 0], [np.nan, np.nan]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 1.0],
                             [np.nan, "m3", 0.0], [0.0, "m4", np.nan]]
                           ).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[0], domainB[0])])
        self.widget.controls.merging.buttons[self.widget.OuterJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_class_left(self):
        """Check output for merging option 'Append columns from Extra Data' by
        class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars, domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, np.nan, np.nan], [1, 1, 2, np.nan],
                             [2, 0, 1, 1], [3, 1, np.nan, np.nan]])
        result_Y = np.array([0, 1, 2, np.nan])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 0.0],
                             [np.nan, "m3", 1.0], [0.0, "m4", np.nan]]
                           ).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[2], domainB[2])])
        self.widget.commit()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_class_inner(self):
        """Check output for merging option 'Find matching rows' by class
        variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars, domainA.metas + domainB.metas)
        result_X = np.array([[1, 1, 2, np.nan], [2, 0, 1, 1]])
        result_Y = np.array([1, 2])
        result_M = np.array([[1.0, "m2", 0.0], [np.nan, "m3", 1.0]]
                           ).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state(
            [(domainA.class_vars[0], domainB.class_vars[0])])
        self.widget.controls.merging.buttons[1].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_class_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        class variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, np.nan, np.nan], [1, 1, 2, np.nan],
                             [2, 0, 1, 1], [3, 1, np.nan, np.nan],
                             [np.nan, np.nan, 0, 0]])
        result_Y = np.array([[0, np.nan], [1, 0], [2, 1], [np.nan, np.nan],
                             [np.nan, np.nan]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", 0.0],
                             [np.nan, "m3", 1.0], [0.0, "m4", np.nan],
                             [np.nan, "", np.nan]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state(
            [(domainA.class_vars[0], domainB.class_vars[0])])
        self.widget.controls.merging.buttons[2].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_meta_left(self):
        """Check output for merging option 'Append columns from Extra Data' by
        meta variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas)
        result_X = np.array([[0, 0, np.nan, np.nan], [1, 1, np.nan, np.nan],
                             [2, 0, np.nan, np.nan], [3, 1, 2, np.nan]])
        result_Y = np.array([[0, np.nan], [1, np.nan],
                             [2, np.nan], [np.nan, 0]])
        result_M = np.array([[0.0, "m1"], [1.0, "m2"], [np.nan, "m3"],
                             [0.0, "m4"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[-2], domainB[-1])])
        self.widget.commit()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_meta_inner(self):
        """Check output for merging option 'Find matching rows' by meta
        variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas)
        result_X = np.array([[3, 1, 2, np.nan]])
        result_Y = np.array([[np.nan, 0]])
        result_M = np.array([[0.0, "m4"]]).astype(object)
        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[-2], domainB[-1])])
        self.widget.controls.merging.buttons[self.widget.InnerJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_meta_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        meta variable"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        result_d = Domain(domainA.attributes + domainB.attributes,
                          domainA.class_vars + domainB.class_vars,
                          domainA.metas + domainB.metas)
        result_X = np.array([[0, 0, np.nan, np.nan], [1, 1, np.nan, np.nan],
                             [2, 0, np.nan, np.nan], [3, 1, 2, np.nan],
                             [np.nan, np.nan, 0, 0], [np.nan, np.nan, 1, 1]])
        result_Y = np.array([[0, np.nan], [1, np.nan], [2, np.nan],
                             [np.nan, 0], [np.nan, np.nan], [np.nan, 1]])
        result_M = np.array([[0.0, "m1", np.nan], [1.0, "m2", np.nan],
                             [np.nan, "m3", np.nan], [0.0, "m4", 0.0],
                             [np.nan, "", np.nan], [np.nan, "", 1.0]]
                           ).astype(object)

        result = Table(result_d, result_X, result_Y, result_M)

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.widget.attr_boxes.set_state([(domainA[-2], domainB[-1])])
        self.widget.controls.merging.buttons[2].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def assertTablesEqual(self, table1, table2):
        self.assertEqual(table1.domain, table2.domain)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas.astype(str),
                                      table2.metas.astype(str))

    def test_best_match(self):
        """Check default merging attributes setup"""
        widget = self.widget
        indices = list(range(101))
        indices.pop(26)
        zoo = Table("zoo")[indices]
        zoo_images = Table(test_filename("datasets/zoo-with-images.tab"))
        self.send_signal(widget.Inputs.data, zoo)
        self.send_signal(widget.Inputs.extra_data, zoo_images)
        for i in range(3):
            self.assertEqual(widget.attr_boxes.current_state(),
                             [(zoo.domain[-1], zoo_images.domain[-1])],
                             f"wrong attributes chosen for merge_type={i}")

    def test_sparse(self):
        """
        Merge should work with sparse.
        GH-2295
        GH-2155
        """
        data = Table("iris")[::25]
        data_ed_dense = Table("titanic")[::300]
        data_ed_sparse = Table("titanic")[::300].to_sparse()
        self.send_signal("Data", data)

        self.send_signal("Extra Data", data_ed_dense)
        output_dense = self.get_output("Data")
        self.assertFalse(sp.issparse(output_dense.X))
        self.assertFalse(output_dense.is_sparse())

        self.send_signal("Extra Data", data_ed_sparse)
        output_sparse = self.get_output("Data")
        self.assertTrue(sp.issparse(output_sparse.X))
        self.assertTrue(output_sparse.is_sparse())

        output_sparse.X = output_sparse.X.toarray()
        self.assertTablesEqual(output_dense, output_sparse)

    def test_commit_on_new_data(self):
        """Check that disabling auto apply doesn't block on new data"""
        self.widget.auto_apply = False
        self.widget.merging = 2
        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))

    def test_multiple_attributes_left(self):
        domainA, domainB = self.dataA.domain, self.dataB.domain

        X = np.array([[0, 1], [1, 1], [3, np.nan], [np.nan, 0]])
        Y = np.array([1, 2, 0, 0])
        metas = np.array([[0, "a"], [1, "b"], [0, "c"], [0, "d"]])
        dataA = Table(domainA, X, Y, metas)
        dataA.name = "dataA"

        X = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [3, np.nan]])
        Y = np.array([0, 0, 1, 1, 0, 1])
        metas = np.array([[0], [1], [0], [1], [1], [1]])
        dataB = Table(domainB, X, Y, metas)
        dataB.name = "dataB"

        self.send_signal(self.widget.Inputs.data, dataA)
        self.send_signal(self.widget.Inputs.extra_data, dataB)
        self.widget.attr_boxes.set_state(
            [(domainA[0], domainB[0]), (domainA[1], domainB[1])])
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)

        self.assertEqual(output.name, dataA.name)
        np.testing.assert_array_equal(output.ids, dataA.ids)
        self.assertEqual(output.attributes, dataA.attributes)
        np.testing.assert_equal(output.X, dataA.X)

    def test_nonunique(self):
        widget = self.widget
        x = ContinuousVariable("x")
        d = DiscreteVariable("d", values=list("abc"))
        domain = Domain([x, d], [])
        dataA = Table.from_numpy(
            domain, np.array([[1.0, 0], [1, 1], [2, 1]]))
        dataB = Table.from_numpy(
            domain, np.array([[1.0, 0], [2, 1], [3, 1]]))
        dataB.ids = dataA.ids
        self.send_signal(widget.Inputs.data, dataA)
        self.send_signal(widget.Inputs.extra_data, dataB)
        widget.merging = widget.InnerJoin

        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())

        widget.attr_boxes.set_state([(INSTANCEID, INSTANCEID)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        widget.attr_boxes.set_state([(INDEX, INDEX)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        widget.attr_boxes.set_state([(x, x)])
        widget.unconditional_commit()
        self.assertTrue(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))

        widget.merging = widget.LeftJoin
        widget.unconditional_commit()
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        widget.merging = widget.InnerJoin
        widget.attr_boxes.set_state([(x, x), (d, d)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        widget.attr_boxes.set_state([(d, d)])
        widget.unconditional_commit()
        self.assertTrue(widget.Error.nonunique_left.is_shown())
        self.assertTrue(widget.Error.nonunique_right.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))

        widget.merging = widget.LeftJoin
        widget.unconditional_commit()
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertTrue(widget.Error.nonunique_right.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))

        widget.merging = widget.InnerJoin
        widget.unconditional_commit()
        self.assertTrue(widget.Error.nonunique_left.is_shown())
        self.assertTrue(widget.Error.nonunique_right.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))

        self.send_signal(widget.Inputs.data, None)
        self.send_signal(widget.Inputs.extra_data, None)
        self.assertFalse(widget.Error.nonunique_left.is_shown())
        self.assertFalse(widget.Error.nonunique_right.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))

    def test_invalide_pairs(self):
        widget = self.widget
        x = ContinuousVariable("x")
        d = DiscreteVariable("d", values=list("abc"))
        domain = Domain([x, d], [])
        dataA = Table.from_numpy(
            domain, np.array([[1.0, 0], [1, 1], [2, 1]]))
        dataB = Table.from_numpy(
            domain, np.array([[1.0, 0], [2, 1], [3, 1]]))
        dataB.ids = dataA.ids
        self.send_signal(widget.Inputs.data, dataA)
        self.send_signal(widget.Inputs.extra_data, dataB)

        widget.attr_boxes.set_state([(x, x), (d, d)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.matching_id_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (INDEX, d)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.matching_id_with_sth.is_shown())
        self.assertTrue(widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (d, INDEX)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.matching_id_with_sth.is_shown())
        self.assertTrue(widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (INSTANCEID, d)])
        widget.unconditional_commit()
        self.assertTrue(widget.Error.matching_id_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (d, INSTANCEID)])
        widget.unconditional_commit()
        self.assertTrue(widget.Error.matching_id_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (INDEX, INSTANCEID)])
        widget.unconditional_commit()
        self.assertTrue(widget.Error.matching_id_with_sth.is_shown()
                        or widget.Error.matching_index_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_numeric_with_nonnum.is_shown())

        widget.attr_boxes.set_state([(x, x), (x, d)])
        widget.unconditional_commit()
        self.assertFalse(widget.Error.matching_id_with_sth.is_shown())
        self.assertFalse(widget.Error.matching_index_with_sth.is_shown())
        self.assertTrue(widget.Error.matching_numeric_with_nonnum.is_shown())
