# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from itertools import chain
import unittest

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt

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
                         DiscreteVariable("clsA", ("aaa", "bbb", "ccc")),
                         [DiscreteVariable("mA1", ("cc", "dd")),
                          StringVariable("mA2")])
        XA = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        yA = np.array([0, 1, 2, np.nan])
        metasA = np.array([[0.0, "m1"], [1.0, "m2"], [np.nan, "m3"],
                           [0.0, "m4"]]).astype(object)

        domainB = Domain([DiscreteVariable("dB1", values=("a", "b", "c")),
                          DiscreteVariable("dB2", values=("aa", "bb"))],
                         DiscreteVariable("clsB", values=("bbb", "ccc")),
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

    def test_combobox_items(self):
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

    def test_combo_box_sync(self):
        row = self.widget.attr_boxes.rows[0]
        data_combo, extra_combo = row.left_combo, row.right_combo

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)

        extra_combo.setCurrentIndex(3)
        data_combo.setCurrentIndex(0)
        data_combo.activated.emit(0)
        self.assertEqual(data_combo.currentIndex(), 0)
        self.assertEqual(extra_combo.currentIndex(), 0)

        data_combo.setCurrentIndex(1)
        data_combo.activated.emit(1)
        self.assertEqual(data_combo.currentIndex(), 1)
        self.assertEqual(extra_combo.currentIndex(), 1)

        data_combo.setCurrentIndex(2)
        data_combo.activated.emit(2)

        extra_combo.setCurrentIndex(0)
        extra_combo.activated.emit(0)
        self.assertEqual(data_combo.currentIndex(), 0)
        self.assertEqual(extra_combo.currentIndex(), 0)

        extra_combo.setCurrentIndex(1)
        extra_combo.activated.emit(1)
        self.assertEqual(data_combo.currentIndex(), 1)
        self.assertEqual(extra_combo.currentIndex(), 1)

    def test_attr_combo_tooltips(self):
        row = self.widget.attr_boxes.rows[0]
        model = row.left_combo.model()
        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)

        tip = model.data(model.index(2, 0), Qt.ToolTipRole)
        # Test the test; if general tooltips ever change and the following
        # assert fails, the rest of this test has to be modified accordingly
        self.assertTrue(tip.startswith("<b>"))

        # Just test that tooltip is a string (implicitly) and that it's not
        # a generic DomainModel tooltip
        tip = model.data(model.index(0, 0), Qt.ToolTipRole)
        self.assertFalse(tip.startswith("<b>"))

        tip = model.data(model.index(1, 0), Qt.ToolTipRole)
        self.assertFalse(tip.startswith("<b>"))

    def test_match_attr_name(self):
        widget = self.widget
        row = widget.attr_boxes.rows[0]
        data_combo, extra_combo = row.left_combo, row.right_combo

        domainA = Domain([DiscreteVariable("dA1", ("a", "b", "c", "d")),
                          DiscreteVariable("dA2", ("aa", "bb")),
                          DiscreteVariable("dA3", ("aa", "bb"))],
                         DiscreteVariable("cls", ("aaa", "bbb", "ccc")),
                         [DiscreteVariable("mA1", ("cc", "dd")),
                          StringVariable("mA2")])
        XA = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]])
        yA = np.array([0, 1, 2, np.nan])
        metasA = np.array([[0.0, "m1"], [1.0, "m2"], [np.nan, "m3"],
                           [0.0, "m4"]]).astype(object)

        domainB = Domain([DiscreteVariable("dB1", values=("a", "b", "c")),
                          ContinuousVariable("dA2")],
                         None,
                         [StringVariable("cls"),
                          DiscreteVariable("dA1", ("m4", "m5"))])
        XB = np.array([[0, 0], [1, 1], [2, np.nan]])
        yB = np.empty((3, 0))
        metasB = np.array([[np.nan, np.nan], [1, 1], [0, 0]]).astype(object)
        dataA = Table(domainA, XA, yA, metasA)
        dataA.name = 'dataA'
        dataA.attributes = 'dataA attributes'
        dataB = Table(domainB, XB, yB, metasB)
        dataB.name = 'dataB'
        dataB.attributes = 'dataB attributes'

        self.send_signal(widget.Inputs.data, dataA)
        self.send_signal(widget.Inputs.extra_data, dataB)

        # match variable if available and the other combo is Row Index
        extra_combo.setCurrentIndex(0)
        extra_combo.activated.emit(0)
        data_combo.setCurrentIndex(2)
        data_combo.activated.emit(2)
        self.assertEqual(extra_combo.currentIndex(), 5)

        # match variable if available and the other combo is ID
        extra_combo.setCurrentIndex(1)
        extra_combo.activated.emit(1)
        data_combo.setCurrentIndex(2)
        data_combo.activated.emit(2)
        self.assertEqual(extra_combo.currentIndex(), 5)

        # don't match variable if other combo is set
        extra_combo.setCurrentIndex(4)
        extra_combo.activated.emit(4)
        data_combo.setCurrentIndex(2)
        data_combo.activated.emit(2)
        self.assertEqual(extra_combo.currentIndex(), 4)

        # don't match if nothing to match to
        extra_combo.setCurrentIndex(0)
        extra_combo.activated.emit(0)
        data_combo.setCurrentIndex(4)
        data_combo.activated.emit(4)
        self.assertEqual(extra_combo.currentIndex(), 0)

        # don't match numeric with non-numeric
        extra_combo.setCurrentIndex(0)
        extra_combo.activated.emit(0)
        data_combo.setCurrentIndex(3)
        data_combo.activated.emit(3)
        self.assertEqual(extra_combo.currentIndex(), 0)

        # allow matching string with discrete
        extra_combo.setCurrentIndex(0)
        extra_combo.activated.emit(0)
        data_combo.setCurrentIndex(5)
        data_combo.activated.emit(5)
        self.assertEqual(extra_combo.currentIndex(), 4)

    def test_add_row_button(self):
        boxes = self.widget.attr_boxes
        boxes.set_state([(INSTANCEID, INSTANCEID), (INSTANCEID, INSTANCEID)])
        boxes.rows[-1].add_button.clicked.emit()
        self.assertEqual(len(boxes.rows), 3)
        self.assertEqual(boxes.layout().count(), 3)

    def test_remove_row(self):
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)

        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])
        for i, row in enumerate(boxes.rows):
            self.assertTrue(row.remove_button.isEnabled())
            self.assertEqual(row.remove_button.text(), "×")
            self.assertEqual(row.add_button.isEnabled(), i == 2)
            self.assertEqual(row.add_button.text(), ["", "+"][i == 2])

        boxes.rows[1].remove_button.clicked.emit()
        self.assertEqual(boxes.current_state(), [(INDEX, INDEX), (var0, var1)])
        for i, row in enumerate(boxes.rows):
            self.assertTrue(row.remove_button.isEnabled())
            self.assertEqual(row.remove_button.text(), "×")
            self.assertEqual(row.add_button.isEnabled(), i == 1)
            self.assertEqual(row.add_button.text(), ["", "+"][i])

        boxes.rows[1].remove_button.clicked.emit()
        self.assertEqual(boxes.current_state(), [(INDEX, INDEX)])
        row = boxes.rows[0]
        self.assertFalse(row.remove_button.isEnabled())
        self.assertEqual(row.remove_button.text(), "")
        self.assertTrue(row.add_button.isEnabled())
        self.assertEqual(row.add_button.text(), "+")

        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])
        boxes.rows[2].remove_button.clicked.emit()
        self.assertEqual(
            boxes.current_state(), [(INDEX, INDEX), (INSTANCEID, INSTANCEID)])
        for i, row in enumerate(boxes.rows):
            self.assertTrue(row.remove_button.isEnabled())
            self.assertEqual(row.remove_button.text(), "×")
            self.assertEqual(row.add_button.isEnabled(), i == 1)
            self.assertEqual(row.add_button.text(), ["", "+"][i == 1])

    def test_dont_remove_single_row(self):
        widget = self.widget
        rows = widget.attr_boxes.rows
        self.assertEqual(len(rows), 1)
        rows[0].remove_button.clicked.emit()
        self.assertEqual(len(rows), 1)

    def test_retrieve_settings(self):
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)

        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])

        settings = widget.settingsHandler.pack_data(widget)

        widget2 = self.create_widget(OWMergeData, stored_settings=settings)
        widget2.attr_boxes.set_state([(INDEX, INDEX)])
        self.send_signals(
            [(widget2.Inputs.data, self.dataA),
             (widget2.Inputs.extra_data, self.dataA)],
            widget=widget2)
        self.assertEqual(
            widget2.attr_boxes.current_state(),
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])

    def test_retrieve_settings_with_missing_var(self):
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)

        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])

        domain = self.dataA.domain

        # The left combo in the last row must change to INDEX due to missing var
        data2 = self.dataA.transform(
            Domain(domain.attributes[1:], domain.class_var, domain.metas))
        settings = widget.settingsHandler.pack_data(widget)
        widget2 = self.create_widget(OWMergeData, stored_settings=settings)
        widget2.attr_boxes.set_state([(INDEX, INDEX)])
        self.send_signals(
            [(widget2.Inputs.data, data2),
             (widget2.Inputs.extra_data, data2)],
            widget=widget2)
        self.assertEqual(
            widget2.attr_boxes.current_state(),
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (INDEX, var1)])

        # The last row is changed to INDEX, INDEX and removed as duplicate
        data2 = self.dataA.transform(
            Domain(domain.attributes[2:], domain.class_var, domain.metas))
        settings = widget.settingsHandler.pack_data(widget)
        widget2 = self.create_widget(OWMergeData, stored_settings=settings)
        widget2.attr_boxes.set_state([(INDEX, INDEX)])
        self.send_signals(
            [(widget2.Inputs.data, data2),
             (widget2.Inputs.extra_data, data2)],
            widget=widget2)
        self.assertEqual(
            widget2.attr_boxes.current_state(),
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID)])

    def test_migrate_settings(self):
        attr1, attr2, attr3, attr4, attr5 = [object() for _ in range(5)]
        orig_settings = dict(
            attr_augment_data=attr1, attr_augment_extra=attr2,
            attr_merge_data=attr3, attr_merge_extra=attr4,
            attr_combine_data=attr5, attr_combine_extra='Position (index)')

        widget = self.create_widget(
            OWMergeData, stored_settings=dict(merging=0, **orig_settings))
        self.assertEqual(widget.attr_pairs, (True, True, [(attr1, attr2)]))

        widget = self.create_widget(
            OWMergeData, stored_settings=dict(merging=1, **orig_settings))
        self.assertEqual(widget.attr_pairs, (True, True, [(attr3, attr4)]))

        widget = self.create_widget(
            OWMergeData, stored_settings=dict(merging=2, **orig_settings))
        self.assertEqual(widget.attr_pairs, (True, True, [(attr5, INDEX)]))

        orig_settings["attr_combine_extra"] = "Source position (index)"
        widget = self.create_widget(
            OWMergeData, stored_settings=dict(merging=2, **orig_settings))
        self.assertEqual(widget.attr_pairs, (True, True, [(attr5, INSTANCEID)]))

    @unittest.skip("widget doesn't work this way, but could in the future")
    def test_switch_domain(self):  # pragma: no cover
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])

        self.send_signal(self.widget.Inputs.data, self.dataB)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.assertEqual(
            boxes.current_state(),
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID)])

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        boxes.set_state([(var0, var1), (var1, var0)])

        self.send_signal(self.widget.Inputs.data, self.dataB)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.assertEqual(boxes.current_state(), [(INDEX, INDEX)])

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        boxes.set_state([(var1, var1), (var0, var0)])

        domain2 = Domain(self.dataA.domain.attributes[:1],
                         self.dataA.domain.class_var)
        data2 = self.dataA.transform(domain2)
        self.send_signal(self.widget.Inputs.data, data2)
        self.send_signal(self.widget.Inputs.extra_data, data2)
        self.assertEqual(boxes.current_state(), [(var0, var0)])

    def test_domain_switch_cleanup(self):
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])

        self.send_signal(self.widget.Inputs.data, self.dataB)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        self.assertEqual(
            boxes.current_state(),
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID)])

    def test_report(self):
        widget = self.widget
        boxes = widget.attr_boxes
        var0, var1 = self.dataA.domain.attributes[:2]

        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataA)
        boxes.set_state(
            [(INDEX, INDEX), (INSTANCEID, INSTANCEID), (var0, var1)])
        widget.send_report()
        # Don't crash, that's it

    def test_no_matches(self):
        """Check output is None when there are no matches in inner join"""
        self.send_signal(self.widget.Inputs.data, self.dataA)
        self.send_signal(self.widget.Inputs.extra_data, self.dataB)
        domA = self.dataA.domain
        domB = self.dataB.domain

        self.widget.attr_boxes.set_state([(domA["dA1"], domB["dB2"])])
        self.widget.controls.merging.buttons[self.widget.LeftJoin].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))

        self.widget.controls.merging.buttons[self.widget.InnerJoin].click()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.widget.controls.merging.buttons[self.widget.OuterJoin].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))

    def test_output_merge_by_ids_inner(self):
        """Check output for merging option 'Find matching rows' by
        Source position (index)"""
        domain = self.dataA.domain
        result = Table(domain, np.array([[1, 1], [2, 0]]), np.array([1, 2]),
                       np.array([[1.0, "m2"], [np.nan, "m3"]]).astype(object))
        self.send_signal(self.widget.Inputs.data, self.dataA[:3, [0, "clsA", -1]])
        self.send_signal(self.widget.Inputs.extra_data, self.dataA[1:, [1, "clsA", -2]])
        self.widget.attr_boxes.set_state([(INSTANCEID, INSTANCEID)])
        self.widget.controls.merging.buttons[self.widget.InnerJoin].click()
        self.assertTablesEqual(self.get_output(self.widget.Outputs.data), result)

    def test_output_merge_by_ids_outer(self):
        """Check output for merging option 'Concatenate tables, merge rows' by
        Source position (index)"""
        domain = self.dataA.domain
        result = Table(domain,
                       np.array([[1, 1], [2, 0], [3, np.nan], [np.nan, 0]]),
                       np.array([1, 2, np.nan, 0]),
                       np.array([[1.0, "m2"], [np.nan, "m3"],
                                 [0.0, ""], [np.nan, "m1"]]).astype(object))
        self.widget.attr_boxes.set_state([(INSTANCEID, INSTANCEID)])
        self.widget.merging = 2
        self.widget.controls.merging.buttons[self.widget.OuterJoin].click()
        self.send_signal(self.widget.Inputs.data, self.dataA[1:, [0, "clsA", -1]])
        self.send_signal(self.widget.Inputs.extra_data, self.dataA[:3, [1, "clsA", -2]])
        out = self.get_output(self.widget.Outputs.data)
        self.assertTablesEqual(out, result)
        np.testing.assert_equal(
            out.ids, np.hstack((self.dataA.ids[1:], self.dataA.ids[:1])))

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

    def test_output_merge_by_attribute_outer_same_attr(self):
        """Values of columns from extra aata are copied to left part if they
        match"""
        name = StringVariable("name")
        domainA = Domain([ContinuousVariable("x")], None, [name])
        domainB = Domain([ContinuousVariable("y")], None, [name])
        xA = np.array([[0], [1], [2]])
        mA = np.array([["a"], ["b"], ["c"]])
        xB = np.array([[4], [5], [6], [7]])
        mB = np.array([["b"], ["d"], ["a"], ["c"]])
        dataA = Table(domainA, xA, None, mA)
        dataB = Table(domainB, xB, None, mB)

        self.send_signal(self.widget.Inputs.data, dataA)
        self.send_signal(self.widget.Inputs.extra_data, dataB)
        self.widget.attr_boxes.set_state([(name, name)])
        self.widget.controls.merging.buttons[self.widget.OuterJoin].click()
        out = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(
            out.X,
            np.array([[0, 6], [1, 4], [2, 7], [np.nan, 5]]))
        self.assertEqual(" ".join(out.metas.flatten()), "a b c d")

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

    def test_duplicate_names(self):
        domain = Domain([ContinuousVariable("C1")],
                        metas=[DiscreteVariable("Feature", values=["A", "B"])])
        data = Table(domain, np.array([[1.], [0.]]),
                     metas=np.array([[1.], [0.]]))
        domain = Domain([ContinuousVariable("C1")],
                        metas=[StringVariable("Feature")])
        extra_data = Table(domain, np.array([[1.], [0.]]),
                           metas=np.array([["A"], ["B"]]))
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.extra_data, extra_data)
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())
        merged_data = self.get_output(self.widget.Outputs.data)
        self.assertListEqual([m.name for m in merged_data.domain.metas],
                             ["Feature (1)", "Feature (2)"])


if __name__ == "__main__":
    unittest.main()
