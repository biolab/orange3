# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import time
import unittest

import numpy as np

from AnyQt.QtCore import QEvent, QPoint, Qt
from AnyQt.QtGui import QMouseEvent

from Orange.data import Table, DiscreteVariable, Domain, ContinuousVariable, \
    StringVariable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owmosaic import OWMosaicDisplay
from Orange.widgets.tests.utils import simulate

class TestOWMosaicDisplay(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
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
            QEvent.MouseButtonPress, QPoint(), Qt.LeftButton,
            Qt.LeftButton, Qt.KeyboardModifiers()))
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
        cls.iris_no_class = Table(dom, cls.iris)

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)
        self.vizrank = self.widget.vizrank

    def test_count(self):
        """MosaicVizrank correctly computes the number of combinations"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)

        widget.interior_coloring = widget.PEARSON
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 10)  # 5x4 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 20)  # above + 5x4x3 / 2x3
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 25)  # above + 5x4x3x2 / 2x3x4

        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 10)  # 4 + 4x3 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 14)  # above + 4x3x2 / 3x2
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 15)  # above + 4x3x2x1 / 2x3x4

        widget.set_data(self.iris_no_class)
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 6)  # 4x3 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 10)  # above + 4x3x2 / 3x2
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 11)  # above + 4x3x2x1 / 2x3x4

    def test_iteration(self):
        """MosaicVizrank correctly iterates through states"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)
        vizrank.compute_attr_order()

        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.max_attrs = 4
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0], [1], [2], [3],
                          [0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])

        vizrank.max_attrs = 2
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0], [1], [2], [3],
                          [0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3]])

        widget.interior_coloring = widget.PEARSON
        vizrank.max_attrs = 4
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])

        vizrank.max_attrs = 2
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3]])

    def test_row_for_state(self):
        """MosaicVizrank returns table row corresponding to the state"""
        self.widget.set_data(self.iris)
        self.vizrank.attr_ordering = [DiscreteVariable(n) for n in "abcd"]
        items = self.vizrank.row_for_state(0, [1, 3, 0])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.text(), "a, b, d")
        self.assertEqual(
            item.data(self.vizrank._AttrRole),
            tuple(self.vizrank.attr_ordering[i] for i in [0, 1, 3]))

    @unittest.skip("Appveyor sometimes fails.")
    def test_does_not_crash(self):
        """MosaicVizrank computes rankings without crashing"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)
        vizrank.max_attrs = 2

        widget.interior_coloring = widget.PEARSON
        vizrank.toggle()
        time.sleep(0.5)
        self.assertEqual(vizrank.rank_model.rowCount(), 10)  # 4x5 / 2
        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.toggle()
        time.sleep(0.5)
        self.assertEqual(vizrank.rank_model.rowCount(), 10)  # 4 + 4x5 / 2

        widget.set_data(self.iris_no_class)
        vizrank.toggle()

    def test_does_not_crash_cont_class(self):
        """MosaicVizrank computes rankings without crashing"""
        data = Table("housing.tab")
        self.widget.set_data(data)
        self.vizrank.toggle()

    def test_pause_continue(self):
        data = Table("housing.tab")
        self.widget.set_data(data)
        self.vizrank.toggle()  # start
        self.process_events(until=lambda: self.vizrank.saved_progress > 5)
        self.vizrank.toggle()  # stop
        self.process_events(until=lambda: not self.vizrank.keep_running)
        self.vizrank.toggle()  # continue
        self.process_events(until=lambda: self.vizrank.saved_progress > 20)

    def test_finished(self):
        data = Table("iris.tab")
        self.widget.set_data(data)
        self.vizrank.toggle()
        self.process_events(until=lambda: not self.vizrank.keep_running)
        self.assertEqual(len(self.vizrank.scores), self.vizrank.state_count())

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
        GH-2133
        GH-2036
        """
        RESULTS = [[0, 2, 6], [0, 3, 10], [0, 4, 11],
                   [1, 2, 6], [1, 3, 7], [1, 4, 7]]
        table = Table("titanic")
        self.send_signal(self.widget.Inputs.data, table)
        color_vars = ["(Pearson residuals)"] + [str(x) for x in table.domain]
        for i, cv in enumerate(color_vars):
            idx = self.widget.cb_attr_color.findText(cv)
            self.widget.cb_attr_color.setCurrentIndex(idx)
            color = self.widget.cb_attr_color.currentText()
            simulate.combobox_activate_index(self.widget.controls.variable_color, idx, 0)
            discrete_data = self.widget.discrete_data

            if color == "(Pearson residuals)":
                self.widget.interior_coloring = self.widget.PEARSON
                self.assertIsNone(discrete_data.domain.class_var)
            else:
                self.widget.interior_coloring = self.widget.CLASS_DISTRIBUTION
                self.assertEqual(color, str(discrete_data.domain.class_var))

            output = self.get_output("Data")
            self.assertEqual(output.domain.class_var, table.domain.class_var)

            for ma in range(2, 5):
                self.vizrank.max_attrs = ma
                sc = self.vizrank.state_count()
                self.assertTrue([i > 0, ma, sc] in RESULTS)

    def test_scores(self):
        """
        Test scores without running vizrank.
        GH-2299
        GH-2036
        """
        SCORES = {('status', ): 4.35e-40,
                  ('sex', ): 6.18e-100,
                  ('age', ): 2.82e-05,
                  ('sex', 'status'): 1.06e-123,
                  ('age', 'status'): 4.15e-47,
                  ('age', 'sex'): 5.49e-102,
                  ('age', 'sex', 'status'): 5.3e-128}
        table = Table("titanic")
        self.send_signal(self.widget.Inputs.data, table)
        self.vizrank.compute_attr_order()
        self.widget.vizrank.max_attrs = 3
        state = None
        for state in self.vizrank.iterate_states(state):
            self.vizrank.iterate_states(state)
            attrlist = tuple(sorted(self.vizrank.attr_ordering[i].name for i in state))
            sc = self.vizrank.compute_score(state)
            self.assertTrue(np.allclose(sc, SCORES[attrlist], rtol=0.003, atol=0))
