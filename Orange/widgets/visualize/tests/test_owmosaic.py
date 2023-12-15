# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import patch

import numpy as np

from AnyQt.QtCore import QEvent, QPointF, Qt
from AnyQt.QtGui import QMouseEvent
from AnyQt.QtTest import QSignalSpy

from Orange.data import Table, DiscreteVariable, Domain, ContinuousVariable, \
    StringVariable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owmosaic import OWMosaicDisplay
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.utils.vizrank import RunState


class TestOWMosaicDisplay(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = OWMosaicDisplay.Inputs.data
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, self.data[:0])

    def test_empty_column(self):
        """Check that the widget doesn't crash if the columns are empty"""
        self.send_signal(self.widget.Inputs.data, self.data[:, :0])

    def _select_data(self):
        self.widget.select_area(1, QMouseEvent(
            QEvent.MouseButtonPress, QPointF(), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        return [2, 3, 9, 23, 29, 30, 34, 35, 37, 42, 47, 49]

    def test_continuous_metas(self):
        """Check widget for dataset with continuous metas"""
        domain = Domain([ContinuousVariable("c1")],
                        metas=[ContinuousVariable("m")])
        data = Table(domain, np.arange(6).reshape(6, 1),
                     metas=np.arange(6).reshape(6, 1))
        self.send_signal(self.widget.Inputs.data, data)

    def test_string_meta(self):
        """Check widget for dataset with only one string meta"""
        domain = Domain([], metas=[StringVariable("m")])
        data = Table(domain, np.empty((6, 0)),
                     metas=np.array(["meta"] * 6).reshape(6, 1))
        self.send_signal(self.widget.Inputs.data, data)

    def test_missing_values(self):
        """Check widget for dataset with missing values"""
        data = Table(Domain([DiscreteVariable("c1", [])]),
                     np.array([np.nan] * 6)[:, None])
        self.send_signal(self.widget.Inputs.data, data)

        # missings in class variable
        attrs = [DiscreteVariable("c1", ["a", "b", "c"])]
        class_var = DiscreteVariable("cls", [])
        X = np.array([1, 2, 0, 1, 0, 2])[:, None]
        data = Table(Domain(attrs, class_var), X, np.array([np.nan] * 6))
        self.send_signal(self.widget.Inputs.data, data)

    def test_keyerror(self):
        """gh-2014
        Check if it works when a table has only one row or duplicates.
        Discretizer must have remove_const set to False.
        """
        data = Table("iris")
        data = data[0:1]
        self.send_signal(self.widget.Inputs.data, data)

    def test_combos_and_mosaic(self):
        """
        Text in combos is wiped away when there is no data and mosaic disappears as well.
        GH-2459
        GH-2462
        """
        def assertCount(cb_color, cb_attr, areas):
            self.assertEqual(len(self.widget.areas), areas)
            self.assertEqual(self.widget.cb_attr_color.count(), cb_color)
            for combo, cb_count in zip(self.widget.attr_combos, cb_attr):
                self.assertEqual(combo.count(), cb_count)

        data = Table("iris")
        assertCount(1, [0, 1, 1, 1], 0)
        self.send_signal(self.widget.Inputs.data, data)
        assertCount(6, [5, 6, 6, 6], 16)
        self.send_signal(self.widget.Inputs.data, None)
        assertCount(1, [0, 1, 1, 1], 0)

    @patch('Orange.widgets.visualize.owmosaic.CanvasRectangle')
    @patch('Orange.widgets.visualize.owmosaic.QGraphicsItemGroup.addToGroup')
    def test_different_number_of_attributes(self, _, canvas_rectangle):
        domain = Domain([DiscreteVariable(c, values="01") for c in "abcd"])
        data = Table.from_list(
            domain,
            [list("{:04b}".format(i)[-4:]) for i in range(16)])
        self.send_signal(self.widget.Inputs.data, data)
        widget = self.widget
        widget.variable2 = widget.variable3 = widget.variable4 = None
        for i, attr in enumerate(data.domain[:4], start=1):
            canvas_rectangle.reset_mock()
            setattr(self.widget, "variable" + str(i), attr)
            self.widget.update_graph()
            self.assertEqual(canvas_rectangle.call_count, 7 + 2 ** (i + 1))

    def test_change_domain(self):
        """Test for GH-3419 fix"""
        self.send_signal(self.widget.Inputs.data, self.data[:, :2])
        subset = self.data[:1, 2:3]
        self.send_signal(self.widget.Inputs.data, subset)
        output = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(output.X, subset.X)

    def test_subset(self):
        """Test for GH-3416 fix"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.data_subset, self.data[10:])
        self.send_signal(self.widget.Inputs.data, self.data[:1])
        output = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(output.X, self.data[:1].X)

    @patch('Orange.widgets.visualize.owmosaic.MosaicVizRank.auto_select')
    def test_vizrank_receives_manual_change(self, auto_select):
        # Recreate the widget so the patch kicks in
        self.widget = self.create_widget(OWMosaicDisplay)
        data = Table("iris.tab")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.variable1 = data.domain[0]
        self.widget.variable2 = data.domain[1]
        simulate.combobox_activate_index(self.widget.controls.variable2, 3)
        self.assertEqual(self.widget.variable2, data.domain[2])
        call_args = auto_select.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0].name, data.domain[0].name)
        self.assertEqual(call_args[1].name, data.domain[2].name)

    def test_selection_setting(self):
        widget = self.widget
        data = Table("iris.tab")
        self.send_signal(widget.Inputs.data, data)

        widget.select_area(
            1,
            QMouseEvent(QEvent.MouseButtonPress, QPointF(), Qt.LeftButton,
                        Qt.LeftButton, Qt.NoModifier))

        # Changing the data must reset the selection
        self.send_signal(widget.Inputs.data, Table("titanic"))
        self.assertFalse(bool(widget.selection))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

        self.send_signal(widget.Inputs.data, data)
        self.assertFalse(bool(widget.selection))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

        widget.select_area(
            1,
            QMouseEvent(QEvent.MouseButtonPress, QPointF(), Qt.LeftButton,
                        Qt.LeftButton, Qt.NoModifier))
        settings = self.widget.settingsHandler.pack_data(self.widget)

        # Setting data to None must reset the selection
        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(bool(widget.selection))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

        self.send_signal(widget.Inputs.data, data)
        self.assertFalse(bool(widget.selection))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

        w = self.create_widget(OWMosaicDisplay, stored_settings=settings)
        self.assertFalse(bool(widget.selection))
        self.send_signal(w.Inputs.data, data, widget=w)
        self.assertEqual(w.selection, {1})
        self.assertIsNotNone(self.get_output(w.Outputs.selected_data, widget=w))


# Derive from WidgetTest to simplify creation of the Mosaic widget, although
# we are actually testing the MosaicVizRank dialog and not the widget

# These tests are rather crude: the main challenge of this widget is to handle
# user interactions and interrupts, e.g. changing the widget settings or
# getting new data while the computation is running.
class MosaicVizRankTests(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris.tab")
        dom = Domain(cls.iris.domain.attributes, [])
        cls.iris_no_class = cls.iris.transform(dom)

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)
        self.widget.vizrank_attr_range_index = 0

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_count(self):
        """MosaicVizrank correctly computes the number of combinations"""

        data = self.iris
        attributes = [v for v in data.domain.attributes[1:]]
        metas = [data.domain.attributes[0]]
        domain = Domain(attributes, data.domain.class_var, metas)
        new_data = data.from_table(domain, data)

        for data in [self.iris, new_data]:
            self.send_signal(self.widget.Inputs.data, data)

            simulate.combobox_activate_index(self.widget.controls.variable_color, 0, 0)
            vizrank = self.widget.vizrank_dialog
            vizrank.attr_range_index = 1
            self.assertEqual(vizrank.state_count(), 10)  # 5x4 / 2
            vizrank.attr_range_index = 2
            self.assertEqual(vizrank.state_count(), 10)  # 5x4x3 / 2x3
            vizrank.attr_range_index = 3
            self.assertEqual(vizrank.state_count(), 5)  # 5x4x3x2 / 2x3x4
            vizrank.attr_range_index = 4
            self.assertEqual(vizrank.state_count(), 10)  # 5x4 / 2
            vizrank.attr_range_index = 5
            self.assertEqual(vizrank.state_count(), 20)  # above + 5x4x3 / 2x3
            vizrank.attr_range_index = 6
            self.assertEqual(vizrank.state_count(), 25)  # above + 5x4x3x2 / 2x3x4

            simulate.combobox_activate_index(self.widget.controls.variable_color, 2, 0)
            vizrank = self.widget.vizrank_dialog
            vizrank.attr_range_index = 0
            self.assertEqual(vizrank.state_count(), 4)  # 4
            vizrank.attr_range_index = 1
            self.assertEqual(vizrank.state_count(), 6)  # 4x3 / 2
            vizrank.attr_range_index = 2
            self.assertEqual(vizrank.state_count(), 4)  # 4x3x2 / 3x2
            vizrank.attr_range_index = 3
            self.assertEqual(vizrank.state_count(), 1)  # 4x3x2x1 / 2x3x4
            vizrank.attr_range_index = 4
            self.assertEqual(vizrank.state_count(), 10)  # 4 + 4x3 / 2
            vizrank.attr_range_index = 5
            self.assertEqual(vizrank.state_count(), 14)  # above + 4x3x2 / 3x2
            vizrank.attr_range_index = 6
            self.assertEqual(vizrank.state_count(), 15)  # above + 4x3x2x1 / 2x3x4

            self.send_signal(self.widget.Inputs.data, self.iris_no_class)
            simulate.combobox_activate_index(self.widget.controls.variable_color, 0, 0)
            vizrank = self.widget.vizrank_dialog
            vizrank.attr_range_index = 1
            self.assertEqual(vizrank.state_count(), 6)  # 4x3 / 2
            vizrank.attr_range_index = 2
            self.assertEqual(vizrank.state_count(), 4)  # 4x3x2 / 3x2
            vizrank.attr_range_index = 3
            self.assertEqual(vizrank.state_count(), 1)  # 4x3x2x1 / 2x3x4
            vizrank.attr_range_index = 4
            self.assertEqual(vizrank.state_count(), 6)  # 4x3 / 2
            vizrank.attr_range_index = 5
            self.assertEqual(vizrank.state_count(), 10)  # above + 4x3x2 / 3x2
            vizrank.attr_range_index = 6
            self.assertEqual(vizrank.state_count(), 11)  # above + 4x3x2x1 / 2x3x4

    def test_iteration(self):
        """MosaicVizrank correctly iterates through states"""
        widget = self.widget
        widget.vizrank_attr_range_index = 1
        self.send_signal(self.widget.Inputs.data, self.iris)
        vizrank = self.widget.vizrank_dialog

        self.assertEqual(set(vizrank.state_generator()),
                         {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)})

        vizrank.attr_range_index = 6
        self.assertEqual(set(vizrank.state_generator()),
                         {(0, ), (1, ), (2, ), (3, ),
                          (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
                          (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
                          (0, 1, 2, 3)})

        vizrank.attr_range_index = 4
        self.assertEqual(set(vizrank.state_generator()),
                         {(0, ), (1, ), (2, ), (3, ),
                          (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)})

        widget.vizrank_attr_range_index = 6
        widget.variable_color = None
        self.widget.cb_attr_color.setCurrentIndex(0)
        simulate.combobox_activate_index(self.widget.controls.variable_color, 0, 0)
        vizrank = self.widget.vizrank_dialog
        self.assertEqual(set(vizrank.state_generator()),
                         {(0, 1), (0, 2), (0, 3), (0, 4),
                          (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 4),
                          (3, 4),
                          (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4),
                          (1, 2, 3), (1, 2, 4), (1, 3, 4),
                          (2, 3, 4),
                          (0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 2, 4), (0, 2, 3, 4), (1, 2, 3, 4)})

        vizrank.attr_range_index = 4
        self.assertEqual(set(vizrank.state_generator()),
                         {(0, 1), (0, 2), (0, 3), (0, 4),
                          (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 4),
                          (3, 4)})

    def test_row_for_state(self):
        """MosaicVizrank returns table row corresponding to the state"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        vizrank = self.widget.vizrank_dialog
        vizrank._attr_order = [DiscreteVariable(n) for n in "abcd"]
        items = vizrank.row_for_state(0, [1, 3, 0])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.text(), "a, b, d")
        self.assertEqual(
            item.data(vizrank._AttrRole),
            tuple(vizrank.attr_order[i] for i in [0, 1, 3]))

    def test_does_not_crash_cont_class(self):
        """MosaicVizrank computes rankings without crashing"""
        data = Table("housing.tab")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.vizrank_button().click()

    def test_max_attr_combo_1_disabling(self):
        widget = self.widget
        enabled = Qt.ItemIsSelectable | Qt.ItemIsEnabled

        data = Table("iris.tab")
        self.send_signal(self.widget.Inputs.data, data)
        vizrank = widget.vizrank_dialog
        combo = vizrank.attrs_combo
        model = combo.model()
        self.assertEqual(model.item(0).flags() & enabled, enabled)

        vizrank.attr_range_index = 0
        simulate.combobox_activate_index(self.widget.controls.variable_color, 0)
        vizrank = widget.vizrank_dialog
        combo = vizrank.attrs_combo
        model = combo.model()
        self.assertEqual(vizrank.attr_range_index, 1)
        self.assertEqual(model.item(0).flags() & enabled, Qt.NoItemFlags)

        simulate.combobox_activate_index(self.widget.controls.variable_color, 1)
        vizrank = widget.vizrank_dialog
        combo = vizrank.attrs_combo
        model = combo.model()
        self.assertEqual(model.item(0).flags() & enabled, enabled)

    def test_disable_combo_3_4(self):
        def assert_enabled(*args, sel=1):
            enabled = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            vizrank = widget.vizrank_dialog
            combo = vizrank.attrs_combo
            model = combo.model()
            for opt in (2, 3):
                self.assertEqual(model.item(opt).flags() & enabled,
                                 enabled if opt + 1 in args else Qt.NoItemFlags)
            self.assertEqual(combo.currentIndex(), sel)

        widget = self.widget

        data = Table("iris.tab")
        widget.vizrank_attr_range_index = 3
        self.send_signal(self.widget.Inputs.data, data)
        assert_enabled(3, 4, sel=3)

        self.send_signal(self.widget.Inputs.data, data[:, :3])
        assert_enabled(3, sel=2)

        self.send_signal(self.widget.Inputs.data, data[:, :2])
        assert_enabled(sel=1)

    def test_attr_range(self):
        data = Table("iris.tab")
        domain = data.domain

        self.send_signal(self.widget.Inputs.data, data)
        vizrank = self.widget.vizrank_dialog
        for vizrank.attr_range_index, rge in (
                (0, (1, 2)), (1, (2, 3)), (2, (3, 4)), (3, (4, 5)),
                (4, (1, 3)), (5, (1, 4)), (6, (1, 5))):
            self.assertEqual(vizrank.attr_range(), rge,
                             f"failed at attr_range_index={vizrank.attr_range_index}")

        reduced = data.transform(Domain(domain.attributes[:2], domain.class_var))
        self.send_signal(self.widget.Inputs.data, reduced)
        vizrank = self.widget.vizrank_dialog
        for vizrank.attr_range_index, rge in (
                (0, (1, 2)), (1, (2, 3)), (2, (3, 3)), (3, (4, 3)),
                (4, (1, 3)), (5, (1, 3)), (6, (1, 3))):
            self.assertEqual(vizrank.attr_range(), rge,
                             f"failed at attr_range_index={vizrank.attr_range_index}")
            self.assertIs(vizrank.state_count() == 0, rge[0] >= rge[1])

        simulate.combobox_activate_index(self.widget.controls.variable_color, 0)
        vizrank = self.widget.vizrank_dialog
        for vizrank.attr_range_index, rge in (
                (0, (2, 3)), (1, (2, 3)), (2, (3, 4)), (3, (4, 4)),
                (4, (2, 3)), (5, (2, 4)), (6, (2, 4))):
            self.assertEqual(vizrank.attr_range(), rge,
                             f"failed at attr_range_index={vizrank.attr_range_index}")
            self.assertIs(vizrank.state_count() == 0, rge[0] >= rge[1])

    def test_nan_column(self):
        """
        A column with only NaN-s used to throw an error
        (ZeroDivisionError) when loaded into widget.
        GH-2046
        """
        table = Table(
            Domain(
                [ContinuousVariable("a"), ContinuousVariable("b"), ContinuousVariable("c")]),
            np.array([
                [0, np.NaN, 0],
                [0, np.NaN, 0],
                [0, np.NaN, 0]
            ])
        )
        self.send_signal(self.widget.Inputs.data, table)

    def test_color_combo(self):
        """
        Color combo enables to select class values. Checks if class values
        are selected correctly.
        """
        table = Table("titanic")
        self.send_signal(self.widget.Inputs.data, table)
        color_vars = ["(Pearson residuals)"] + [str(x) for x in table.domain.variables]
        for cv in color_vars:
            idx = self.widget.cb_attr_color.findText(cv)
            self.widget.cb_attr_color.setCurrentIndex(idx)
            color = self.widget.cb_attr_color.currentText()
            simulate.combobox_activate_index(self.widget.controls.variable_color, idx, 0)
            discrete_data = self.widget.discrete_data

            if color == "(Pearson residuals)":
                self.assertIsNone(discrete_data.domain.class_var)
            else:
                self.assertEqual(color, str(discrete_data.domain.class_var))

            output = self.get_output(self.widget.Outputs.annotated_data)
            self.assertEqual(output.domain.class_var, table.domain.class_var)

    def test_scores(self):
        SCORES = {('status', ): 4.35e-40,
                  ('sex', ): 6.18e-100,
                  ('age', ): 2.82e-05,
                  ('sex', 'status'): 1.06e-123,
                  ('age', 'status'): 4.15e-47,
                  ('age', 'sex'): 5.49e-102,
                  ('age', 'sex', 'status'): 5.3e-128}
        table = Table("titanic")
        self.send_signal(self.widget.Inputs.data, table)
        vizrank = self.widget.vizrank_dialog
        vizrank.prepare_run()
        vizrank.attr_range_index = 3
        for state in vizrank.state_generator():
            attrlist = tuple(sorted(vizrank.attr_order[i].name for i in state))
            sc = vizrank.compute_score(state)
            self.assertTrue(np.allclose(sc, SCORES[attrlist], rtol=0.003, atol=0))

    def test_subset_data(self):
        """
        Mosaic should not crash on subset data and should properly interpret it.
        GH-2515
        GH-2528
        """
        table_titanic = Table("titanic")
        self.send_signal(self.widget.Inputs.data, table_titanic)
        self.send_signal(self.widget.Inputs.data_subset, table_titanic[::20])
        table_housing = Table("housing")
        self.send_signal(self.widget.Inputs.data, table_housing)
        self.send_signal(self.widget.Inputs.data_subset, table_housing[::20])

    def test_incompatible_subset(self):
        """
        Show warning when subset data is not compatible with data.
        GH-2528
        """
        table_titanic = Table("titanic")
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data_subset, table_titanic)
        self.assertTrue(self.widget.Warning.incompatible_subset.is_shown())
        self.send_signal(self.widget.Inputs.data_subset, self.iris)
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())

    def test_autoselect(self):
        data = Table("iris.tab")
        self.widget.vizrank_attr_range_index = 2
        self.send_signal(self.widget.Inputs.data, data)
        spy = QSignalSpy(self.widget.vizrankRunStateChanged)
        self.widget.vizrank_button().click()
        # This takes 0.5 s on my M1 Mac (2022), but let us tolerate 20x longer
        # on CI
        while spy.wait(timeout=10000) and spy[-1][0] != RunState.Done:
            pass

        vizrank = self.widget.vizrank_dialog
        model = vizrank.rank_model
        attrs = model.data(model.index(3, 0), vizrank._AttrRole)
        vizrank.auto_select(attrs)
        selection = vizrank.rank_table.selectedIndexes()
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0].row(), 3)

        vizrank.auto_select(attrs[::-1])
        selection = vizrank.rank_table.selectedIndexes()
        self.assertEqual(len(selection), 0)


if __name__ == "__main__":
    unittest.main()
