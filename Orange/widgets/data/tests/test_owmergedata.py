# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from itertools import chain

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain, DiscreteVariable, StringVariable
from Orange.widgets.data.owmergedata import OWMergeData, INSTANCEID, INDEX
from Orange.widgets.tests.base import WidgetTest


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
        cls.dataB = Table(domainB, XB, yB, metasB)

    def setUp(self):
        self.widget = self.create_widget(OWMergeData)

    def test_input_remove(self):
        """Check widget after inputs have been removed"""
        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.extra_data, None)

    def test_combobox_items_left(self):
        """Check if combo box content is properly set for merging option
        'Append columns from Extra Data'"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        data_combo = self.widget.controls.attr_augment_data
        extra_combo = self.widget.controls.attr_augment_extra

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        data_items = list(chain([INDEX], domainA, domainA.metas))
        extra_items = list(chain(
            [INDEX], domainA.variables[::2], domainA.metas[1:]))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        extra_items = list(chain([INDEX], domainB, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.data, self.dataB)
        data_items = list(chain([INDEX], domainB, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

    def test_combobox_items_inner(self):
        """Check if combo box content is properly set for merging option
        'Find matching rows'"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        data_combo = self.widget.controls.attr_merge_data
        extra_combo = self.widget.controls.attr_merge_extra

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        self.widget.controls.merging.buttons[1].click()
        data_items = extra_items = list(chain(
            [INSTANCEID, INDEX], domainA.variables[::2], domainA.metas[1:]))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        data_items = list(chain(
            [INDEX], domainA.variables[::2], domainA.metas[1:]))
        extra_items = list(chain([INDEX], domainB, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.data, self.dataB)
        data_items = extra_items = list(chain(
            [INSTANCEID, INDEX], domainB, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

    def test_combobox_items_outer(self):
        """Check if combo box content is properly set for merging option
        'Concatenate tables, merge rows'"""
        domainA, domainB = self.dataA.domain, self.dataB.domain
        data_combo = self.widget.controls.attr_combine_data
        extra_combo = self.widget.controls.attr_combine_extra

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        self.widget.controls.merging.buttons[1].click()
        data_items = extra_items = list(chain(
            [INSTANCEID, INDEX], domainA.variables[::2], domainA.metas[1:]))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        data_items = list(chain(
            [INDEX], domainA.variables[::2], domainA.metas[1:]))
        extra_items = list(chain([INDEX], domainB, domainB.metas))
        self.assertListEqual(data_combo.model()[:], data_items)
        self.assertListEqual(extra_combo.model()[:], extra_items)

        self.send_signal(self.widget.Inputs.data, self.dataB)
        data_items = extra_items = list(chain(
            [INSTANCEID, INDEX], domainB, domainB.metas))
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
        self.widget.attr_merge_data = "Source position (index)"
        self.widget.attr_merge_extra = "Source position (index)"
        self.widget.controls.merging.buttons[1].click()
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
        self.widget.attr_combine_data = "Source position (index)"
        self.widget.attr_combine_extra = "Source position (index)"
        self.widget.controls.merging.buttons[2].click()
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
        self.widget.attr_augment_data = domainA[0]
        self.widget.attr_augment_extra = domainB[0]
        self.widget.commit()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

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
        self.widget.attr_merge_data = domainA[0]
        self.widget.attr_merge_extra = domainB[0]
        self.widget.controls.merging.buttons[1].click()
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
        self.widget.attr_combine_data = domainA[0]
        self.widget.attr_combine_extra = domainB[0]
        self.widget.controls.merging.buttons[2].click()
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
        self.widget.attr_augment_data = domainA[2]
        self.widget.attr_augment_extra = domainB[2]
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
        self.widget.attr_merge_data = domainA.class_vars[0]
        self.widget.attr_merge_extra = domainB.class_vars[0]
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
        self.widget.attr_combine_data = domainA.class_vars[0]
        self.widget.attr_combine_extra = domainB.class_vars[0]
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
        self.widget.attr_augment_data = domainA[-2]
        self.widget.attr_augment_extra = domainB[-1]
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
        self.widget.attr_merge_data = domainA[-2]
        self.widget.attr_merge_extra = domainB[-1]
        self.widget.controls.merging.buttons[1].click()
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
        self.widget.attr_combine_data = domainA[-2]
        self.widget.attr_combine_extra = domainB[-1]
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
        indices = list(range(101))
        indices.pop(26)
        zoo = Table("zoo")[indices]
        zoo_images = Table("zoo-with-images")
        self.send_signal(self.widget.Inputs.data, zoo)
        self.send_signal(self.widget.Inputs.extra_data, zoo_images)
        self.assertEqual(self.widget.attr_augment_data, zoo.domain[-1])
        self.assertEqual(self.widget.attr_augment_extra, zoo_images.domain[-1])
        self.assertEqual(self.widget.attr_merge_data, zoo.domain[-1])
        self.assertEqual(self.widget.attr_merge_extra, zoo_images.domain[-1])
        self.assertEqual(self.widget.attr_combine_data, zoo.domain[-1])
        self.assertEqual(self.widget.attr_combine_extra, zoo_images.domain[-1])

    def test_sparse(self):
        """
        Merge should work with sparse.
        GH-2295
        GH-2155
        """
        data = Table("iris")[::25]
        data_ed_dense = Table("titanic")[::300]
        data_ed_sparse = Table("titanic")[::300]
        data_ed_sparse.X = sp.csr_matrix(data_ed_sparse.X)
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
