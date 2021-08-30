# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access

import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import (Table, Domain, StringVariable,
                         ContinuousVariable)
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.utils.annotated_data import (ANNOTATED_DATA_FEATURE_NAME)
from Orange.widgets.visualize.owvenndiagram import (
    OWVennDiagram, get_perm, arrays_equal, pad_columns,
    IDENTITY_STR, EQUALITY_STR)


class TestOWVennDiagram(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls, False)

        cls.signal_data = cls.data[:25]

    def setUp(self):
        self.widget = self.create_widget(OWVennDiagram)
        self.signal_name = self.widget.Inputs.data

    def _select_data(self):
        self.widget.vennwidget.vennareas()[1].setSelected(True)
        return list(range(len(self.signal_data)))

    def test_rows_id(self):
        data = Table('zoo')
        data1 = deepcopy(data)
        with data1.unlocked():
            data1[:, 1] = 1
        self.widget.rowwise = True
        self.send_signal(self.signal_name, data1[:10], 1)
        self.widget.selected_feature = IDENTITY_STR
        self.send_signal(self.signal_name, data[5:10], 2)

        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())

        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected = self.get_output(self.widget.Outputs.selected_data)
        sel_atrs = [atr.name for atr in selected.domain.attributes]
        true_atrs = ['hair', 'feathers (1)', 'feathers (2)', 'eggs', 'milk', 'airborne', 'aquatic',
                     'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs',
                     'tail', 'domestic', 'catsize']
        self.assertEqual(sel_atrs, true_atrs)
        self.assertEqual(selected.domain.metas, data.domain.metas)
        self.assertEqual(selected.domain.class_vars, data.domain.class_vars)

    def test_output_duplicates(self):
        self.widget.rowwise = True
        self.widget.output_duplicates = True
        self.send_signal(self.signal_name, self.data[:2], 1)
        self.send_signal(self.signal_name, self.data[:4], 2)

        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected = self.get_output(self.widget.Outputs.selected_data)
        in_ids = self.data[:2].ids
        true_ids = np.array([in_ids[0], in_ids[0], in_ids[1], in_ids[1]])
        np.testing.assert_array_equal(selected.ids.flatten(), true_ids)

    def test_multiple_input_rows_id(self):
        """Over rows"""
        self.widget.rowwise = True
        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[50:], 2)

        # check selected data output
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # check annotated data output
        feature_name = ANNOTATED_DATA_FEATURE_NAME
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(0, np.sum([i[feature_name] for i in annotated]))

        # select data instances
        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected_indices = list(range(50, 100))

        # check selected data output
        selected = self.get_output(self.widget.Outputs.selected_data)
        n_sel, n_attr = len(selected), len(self.data.domain.attributes)
        self.assertGreater(n_sel, 0)
        self.assertEqual(selected.domain == self.data.domain,
                         self.same_input_output_domain)
        np.testing.assert_array_equal(selected.X[:, :n_attr],
                                      self.data.X[selected_indices])

        # check annotated data output
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(n_sel, np.sum([i[feature_name] for i in annotated]))

        # compare selected and annotated data domains
        self._compare_selected_annotated_domains(selected, annotated)

        # check output when data is removed
        self.send_signal(self.signal_name, None, 1)
        self.send_signal(self.signal_name, None, 2)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_equality(self):
        self.widget.rowwise = True
        data1 = Table.from_numpy(None, np.arange(20).reshape(20, 1))
        data2 = Table.from_numpy(None, np.arange(10, 25).reshape(15, 1))
        self.send_signal(self.signal_name, data1, 1)
        self.send_signal(self.signal_name, data2, 2)

        self.widget.vennwidget.vennareas()[3].setSelected(True)
        self.assertFalse(bool(self.get_output(self.widget.Outputs.selected_data)))

        self.widget.selected_feature = EQUALITY_STR
        self.widget._on_inputAttrActivated()
        self.assertEqual(
            set(self.get_output(self.widget.Outputs.selected_data).X.flatten()),
            set(range(10, 20)))

    def test_disable_duplicates(self):
        cb = self.widget.output_duplicates_cb
        data1 = Table.from_numpy(None, np.arange(20).reshape(20, 1))
        data2 = Table.from_numpy(None, np.arange(10, 25).reshape(15, 1))
        self.send_signal(self.signal_name, data1, 1)
        self.send_signal(self.signal_name, data2, 2)
        self.assertFalse(cb.isEnabled())

        data = Table("zoo")
        self.send_signal(self.signal_name, data, 1)
        self.send_signal(self.signal_name, data, 2)
        for self.widget.selected_feature in (IDENTITY_STR, EQUALITY_STR):
            self.widget._on_inputAttrActivated()
            self.assertFalse(cb.isEnabled())

        self.widget.selected_feature = \
            self.widget.controls.selected_feature.model()[2]
        self.widget._on_inputAttrActivated()
        self.assertTrue(cb.isEnabled())

        self.widget.rowwise = False
        self.widget._on_matching_changed()
        self.assertFalse(cb.isEnabled())

    def test_disable_match_equality(self):
        model = self.widget.controls.selected_feature.model()
        row0 = model.index(0, 0)
        row1 = model.index(1, 0)
        data = Table("zoo")

        self.assertTrue(model.flags(row0) & Qt.ItemIsEnabled)
        self.assertTrue(model.flags(row1) & Qt.ItemIsEnabled)

        self.send_signal(self.signal_name, data, 1)
        self.assertTrue(model.flags(row0) & Qt.ItemIsEnabled)
        self.assertTrue(model.flags(row1) & Qt.ItemIsEnabled)

        self.send_signal(self.signal_name, data, 2)
        self.assertTrue(model.flags(row0) & Qt.ItemIsEnabled)
        self.assertTrue(model.flags(row1) & Qt.ItemIsEnabled)

        data1 = Table.from_numpy(None, np.arange(20).reshape(20, 1))
        self.widget.selected_feature = EQUALITY_STR
        self.send_signal(self.signal_name, data1, 3)
        self.assertEqual(self.widget.selected_feature, IDENTITY_STR)
        self.assertTrue(model.flags(row0) & Qt.ItemIsEnabled)
        self.assertFalse(model.flags(row1) & Qt.ItemIsEnabled)

    def test_multiple_input_over_cols(self):
        self.widget.rowwise = False
        selected_atr_name = 'Selected'
        input2 = self.data.transform(Domain([self.data.domain.attributes[0]],
                                            self.data.domain.class_vars,
                                            self.data.domain.metas)).copy()
        self.send_signal(self.signal_name, self.data, (1, 'Data', None))
        self.send_signal(self.signal_name, input2, (2, 'Data', None))

        selected = self.get_output(self.widget.Outputs.selected_data)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNone(selected)
        self.assertEqual(len(annotated), len(self.data))
        self.assertEqual(annotated.domain, self.data.domain)
        for atr in annotated.domain.attributes:
            self.assertFalse(atr.attributes)

        # select data instances
        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X,
                                      input2.X)
        np.testing.assert_array_equal(selected.Y,
                                      input2.Y)
        np.testing.assert_array_equal(selected.metas,
                                      input2.metas)

        #domain matches but the values do not
        with input2.unlocked(input2.X):
            input2.X = input2.X - 1
        self.send_signal(self.signal_name, input2, (2, 'Data', None))
        self.widget.vennwidget.vennareas()[3].setSelected(True)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        selected = self.get_output(self.widget.Outputs.selected_data)
        atrs = {atr.name for atr in selected.domain.attributes}
        true_atrs = {'sepal length (2)', 'sepal length (1)'}
        self.assertTrue(atrs == true_atrs)

        out_domain = annotated.domain.attributes
        self.assertTrue(out_domain[0].attributes[selected_atr_name])
        self.assertTrue(out_domain[1].attributes[selected_atr_name])
        self.assertFalse(out_domain[2].attributes[selected_atr_name])
        self.assertFalse(out_domain[3].attributes[selected_atr_name])
        self.assertFalse(out_domain[4].attributes[selected_atr_name])

    def test_test_explicit_closing(self):
        data = self.data[:3]
        self.widget.rowwise = True
        self.send_signal(self.signal_name, data[:1], 1)
        self.send_signal(self.signal_name, data[1:2], 2)
        self.send_signal(self.signal_name, data[2:3], 3)
        out = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(out.ids, data[:3].ids)

        self.send_signal(self.signal_name, None, 2)
        out = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(out.ids, data[0:3:2].ids)

        self.send_signal(self.signal_name, data[1:2], 2)
        out = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(out.ids, data[:3].ids)

        self.send_signal(self.signal_name,
                         self.widget.Inputs.data.closing_sentinel, 1)
        out = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(out.ids, data[1:3].ids)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.signal_name, self.data[:0], 1)
        self.send_signal(self.signal_name, self.data[:100], 2)
        self.send_signal(self.signal_name, self.data[50:], 3)

        for i in range(1, 4):
            self.send_signal(self.signal_name, None, i)

        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[:0], 1)
        self.send_signal(self.signal_name, self.data[50:], 3)

        for i in range(1, 4):
            self.send_signal(self.signal_name, None, i)

        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[50:], 2)
        self.send_signal(self.signal_name, self.data[:0], 3)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.autocommit = False
            commit.reset_mock()
            self.send_signal(self.signal_name, self.data[:100], 1)
            commit.assert_called()

    def test_input_compatibility(self):
        self.widget.rowwise = True
        self.send_signal(self.signal_name, self.data, 1)
        self.send_signal(self.signal_name,
                         self.data.transform(Domain([self.data.domain.attributes[0]],
                                                    self.data.domain.class_vars,
                                                    self.data.domain.metas)), 2)
        self.assertFalse(self.widget.Error.instances_mismatch.is_shown())

        self.widget.rowwise = False
        self.send_signal(self.signal_name, self.data[:100, :], 2)
        self.assertTrue(self.widget.Error.instances_mismatch.is_shown())

        self.send_signal(self.signal_name, None, 2)
        self.assertFalse(self.widget.Error.instances_mismatch.is_shown())

    def test_rows_identifiers(self):
        self.widget.rowwise = True
        data = Table('zoo')
        self.send_signal(self.signal_name, data, (1, 'Data', None))
        self.widget.selected_feature = data.domain.metas[0]
        self.send_signal(self.signal_name, data[:5], (2, 'Data', None))

        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected.domain.attributes, data.domain.attributes)
        self.assertEqual(selected.domain.class_vars, data.domain.class_vars)

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(annotated), 100)

    def test_too_many_inputs(self):
        self.send_signal(self.signal_name, self.data, 1)
        self.send_signal(self.signal_name, self.data, 2)
        self.send_signal(self.signal_name, self.data, 3)
        self.send_signal(self.signal_name, self.data, 4)
        self.send_signal(self.signal_name, self.data, 5)
        self.send_signal(self.signal_name, self.data, 6)
        self.assertTrue(self.widget.Error.too_many_inputs.is_shown())

        self.send_signal(self.signal_name,
                         self.widget.Inputs.data.closing_sentinel, 6)
        self.assertFalse(self.widget.Error.too_many_inputs.is_shown())

    def test_no_attributes(self):
        domain = Domain([], class_vars=self.data.domain.attributes)
        n = len(self.data)
        table = Table.from_numpy(domain, np.empty((n, 0)), self.data.X)

        self.widget.rowwise = True
        self.send_signal(self.signal_name, table, 1)
        out = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(out), len(table))

    def test_migration_to_3(self):
        settings = {"selected_feature": None}
        OWVennDiagram.migrate_settings(settings, 1)
        self.assertEqual(settings["selected_feature"], IDENTITY_STR)

        settings = {}
        OWVennDiagram.migrate_settings(settings, 1)
        self.assertEqual(settings["selected_feature"], IDENTITY_STR)


class TestVennUtilities(unittest.TestCase):
    def test_array_equals_cols(self):
        a = np.array([1, 2], dtype=np.float64)
        b = np.array([1, np.nan], dtype=np.float64)
        self.assertTrue(arrays_equal(None, None, None))
        self.assertFalse(arrays_equal(None, a, None))
        self.assertFalse(arrays_equal(a, None, None))
        self.assertFalse(arrays_equal(a, b, ContinuousVariable))
        a[1] = np.nan
        self.assertTrue(arrays_equal(a, b, ContinuousVariable))
        self.assertTrue(arrays_equal(a.astype(str), a.astype(str), StringVariable))
        a[1] = 2
        b[1] = 3
        self.assertFalse(arrays_equal(a, b, ContinuousVariable))
        self.assertFalse(arrays_equal(a.astype(str), b.astype(str), StringVariable))

    def test_pad_columns(self):
        l = 5
        mask = [2, 3]
        values = np.array([7.2, 77.3]).reshape(-1, 1)
        res = pad_columns(values, mask, l)
        true_res = np.array([np.nan, np.nan, 7.2, 77.3, np.nan]).reshape(-1, 1)
        np.testing.assert_array_equal(res, true_res)

    def test_get_perm(self):
        all_ids = [1, 7, 22]
        res = get_perm([7, 33], all_ids)
        true_res = [1]
        self.assertEqual(res, true_res)

        res = get_perm([22, 1, 7], all_ids)
        true_res = [2, 0, 1]
        self.assertEqual(res, true_res)


if __name__ == "__main__":
    unittest.main()
