# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access
from functools import partial

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import QItemSelection, Qt, QEvent
from AnyQt.QtGui import QKeyEvent
from AnyQt.QtWidgets import QCheckBox

from orangewidget.utils.combobox import qcombobox_emit_activated

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.preprocess import BinDefinition
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owdistributions import OWDistributions


class TestOWDistributions(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDistributions)  #: OWDistributions
        self.iris = Table("iris")
        self.heart = Table("heart_disease")

    def _set_cvar(self, cvar):
        combo = self.widget.controls.cvar
        self.widget.cvar = cvar
        qcombobox_emit_activated(combo, combo.currentIndex())

    def _set_fitter(self, i):
        combo = self.widget.controls.fitted_distribution
        combo.setCurrentIndex(i)
        qcombobox_emit_activated(combo, i)

    def _set_var(self, var):
        listview = self.widget.controls.var
        model = listview.model()
        index = var if isinstance(var, int) else model.indexOf(var)
        selectionmodel = listview.selectionModel()
        oldselection = selectionmodel.selection()
        newselection = QItemSelection()
        newselection.select(model.index(index, 0), model.index(index, 0))
        selectionmodel.select(newselection, selectionmodel.ClearAndSelect)
        selectionmodel.selectionChanged.emit(newselection, oldselection)

    @staticmethod
    def _set_check(checkbox: QCheckBox, value: bool):
        state = Qt.Checked if value else Qt.Unchecked
        checkbox.setCheckState(state)
        checkbox.toggled[bool].emit(value)

    def _set_slider(self, i):
        slider = self.widget.controls.number_of_bins
        slider.setValue(i)
        slider.valueChanged[int].emit(i)
        slider.sliderReleased.emit()

    def test_set_data(self):
        """Basic test of set_data and removal of data"""
        widget = self.widget
        var_model = widget.controls.var.model()
        cvar_model = widget.controls.cvar.model()

        domain = self.iris.domain
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertEqual({var.name for var in var_model},
                         {var.name for var in domain.variables})
        self.assertEqual(list(cvar_model),
                         [None, DomainModel.Separator, domain.class_var])
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, domain.class_var)
        np.testing.assert_equal(widget.valid_data, self.iris.X[:, 0])
        np.testing.assert_equal(widget.valid_group_data, self.iris.Y)
        self.assertIsNotNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNotNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

        # Data gone: clean up
        widget.selected_bars.add(widget.ordered_values[0])
        widget._clear_plot = Mock()
        self.send_signal(widget.Inputs.data, None)
        self.assertEqual(len(var_model), 0)
        self.assertEqual(list(cvar_model), [None])
        self.assertIsNone(widget.var)
        self.assertIsNone(widget.cvar)

        self.assertEqual(widget.selected_bars, set())
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertIsNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))
        widget._clear_plot.assert_called()

    def test_set_data_no_class_no_discrete(self):
        """Widget is properly set up when there is no class and discrete vars"""
        widget = self.widget
        var_model = widget.controls.var.model()
        cvar_model = widget.controls.cvar.model()

        domain = Domain(self.iris.domain.attributes, [])
        data = self.iris.transform(domain)
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual({var.name for var in var_model},
                         {var.name for var in domain.attributes})
        self.assertEqual(list(cvar_model), [None])
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, None)
        np.testing.assert_equal(widget.valid_data, self.iris.X[:, 0])
        self.assertIsNone(widget.valid_group_data)
        self.assertIsNotNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNotNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

    def test_set_data_no_class(self):
        """Widget is properly set up when there is no class"""
        widget = self.widget
        var_model = widget.controls.var.model()
        cvar_model = widget.controls.cvar.model()
        iris = self.iris

        domain = Domain(iris.domain.attributes + iris.domain.class_vars)
        data = iris.transform(domain)
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual({var.name for var in var_model},
                         {var.name for var in domain.attributes})
        self.assertEqual(list(cvar_model),
                         [None, DomainModel.Separator, iris.domain.class_var])
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, None)
        np.testing.assert_equal(widget.valid_data, self.iris.X[:, 0])
        self.assertIsNone(widget.valid_group_data)
        self.assertIsNotNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNotNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

    def test_set_data_reg_class(self):
        """Widget is properly set up when the target is numeric"""
        widget = self.widget
        var_model = widget.controls.var.model()
        cvar_model = widget.controls.cvar.model()
        iris = self.iris

        domain = Domain(iris.domain.attributes[:3] + iris.domain.class_vars,
                        iris.domain.attributes[3])
        data = iris.transform(domain)
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual({var.name for var in var_model},
                         {var.name for var in domain.variables})
        self.assertEqual(list(cvar_model),
                         [None, DomainModel.Separator, iris.domain.class_var])
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, None)
        np.testing.assert_equal(widget.valid_data, self.iris.X[:, 0])
        self.assertIsNone(widget.valid_group_data)
        self.assertIsNotNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNotNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

    def test_set_data_reg_class_no_discrete(self):
        """Widget is properly set up with regression and no discrete vars"""
        widget = self.widget
        var_model = widget.controls.var.model()
        cvar_model = widget.controls.cvar.model()
        iris = self.iris

        domain = Domain(iris.domain.attributes[:3], iris.domain.attributes[3])
        data = iris.transform(domain)
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual({var.name for var in var_model},
                         {var.name for var in domain.variables})
        self.assertEqual(list(cvar_model), [None])
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, None)
        np.testing.assert_equal(widget.valid_data, self.iris.X[:, 0])
        self.assertIsNone(widget.valid_group_data)
        self.assertIsNotNone(self.get_output(widget.Outputs.histogram_data))
        self.assertIsNotNone(self.get_output(widget.Outputs.annotated_data))
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))

    def test_histogram_data(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        self._set_var(self.iris.domain["sepal length"])
        self._set_cvar(self.iris.domain["iris"])
        hist = self.get_output(widget.Outputs.histogram_data)
        self.assertTrue(len(hist) > 0 and len(hist) % 3 == 0)

    def test_switch_var(self):
        """Widget reset and recomputes when changing var"""
        widget = self.widget

        self.send_signal(widget.Inputs.data, self.iris)
        binnings = widget.binnings.copy()
        valid_data = widget.valid_data.copy()
        widget.selected_bars.add(widget.ordered_values[1])
        widget._clear_plot = Mock()
        widget.apply.now = widget.apply.deferred = Mock()

        self._set_var(2)
        self.assertFalse(
            binnings[0].thresholds.shape == widget.binnings[0].thresholds.shape
            and np.allclose(binnings[0].threshold, widget.binnings[0].threshold)
        )
        self.assertFalse(valid_data.shape == widget.valid_data.shape
                         and np.allclose(valid_data, widget.valid_data))
        self.assertEqual(widget.selected_bars, set())
        widget._clear_plot.assert_called()
        widget.apply.now.assert_called()

    def test_switch_cvar(self):
        """Widget reset and recomputes when changing splitting variable"""
        widget = self.widget

        y = self.iris.domain.class_var
        extra = DiscreteVariable("foo", values=("a", "b"))
        domain = Domain(self.iris.domain.attributes + (extra, ), y)
        data = self.iris.transform(domain).copy()
        with data.unlocked():
            data.X[:75, -1] = 0
            data.X[75:120, -1] = 1
        self.send_signal(widget.Inputs.data, data)
        self._set_var(2)
        self._set_cvar(y)

        binnings = widget.binnings
        valid_data = widget.valid_data.copy()
        widget.selected_bars.add(widget.ordered_values[1])
        widget._clear_plot = Mock()
        widget.apply.now = widget.apply.deferred = Mock()

        self.assertEqual(len(widget.valid_group_data), 150)

        self._set_cvar(extra)
        self.assertIs(binnings, widget.binnings)
        np.testing.assert_equal(valid_data[:120], widget.valid_data)
        self.assertEqual(len(widget.valid_group_data), 120)
        self.assertEqual(widget.selected_bars, {widget.ordered_values[1]})
        widget._clear_plot.assert_called()
        widget.apply.now.assert_called()
        widget._clear_plot.reset_mock()
        widget.apply.now.reset_mock()

        self._set_cvar(None)
        self.assertIs(binnings, widget.binnings)
        np.testing.assert_equal(valid_data, widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertEqual(widget.selected_bars, {widget.ordered_values[1]})
        widget._clear_plot.assert_called()
        widget.apply.now.assert_called()

    def test_on_bins_changed(self):
        """Widget replots and outputs data when the number of bins is changed"""
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)

        self._set_slider(0)
        widget.selected_bars.add(widget.ordered_values[1])
        n_bars = len(widget.bar_items)
        widget.apply.now = widget.apply.deferred = Mock()

        self._set_slider(1)
        self.assertEqual(widget.selected_bars, set())
        self.assertGreater(n_bars, len(widget.bar_items))
        widget.apply.now.assert_called_once()

    def test_set_valid_data(self):
        """Widget handles nans in data"""
        widget = self.widget
        err_def_var = widget.Error.no_defined_values_var
        err_def_pair = widget.Error.no_defined_values_pair
        warn_nans = widget.Warning.ignored_nans

        domain = self.iris.domain

        self.assertIsNone(widget.var)
        self.assertIsNone(widget.cvar)
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertFalse(widget.is_valid)

        widget.valid_data = Mock()
        widget.group_valid_data = Mock()
        widget.set_valid_data()
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertFalse(widget.is_valid)

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertIsNotNone(widget.valid_data)
        self.assertIsNotNone(widget.valid_group_data)
        self.assertTrue(widget.is_valid)

        with self.iris.unlocked():
            X, Y = self.iris.X, self.iris.Y
            X[:, 0] = np.nan
            X[:50, 1] = np.nan
            X[:100, 2] = np.nan
            Y[75:] = np.nan
        self.send_signal(widget.Inputs.data, self.iris)

        self._set_var(domain[0])
        self._set_cvar(domain.class_var)
        self.assertIs(widget.var, domain[0])
        self.assertIs(widget.cvar, domain.class_var)
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertFalse(widget.is_valid)
        self.assertTrue(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertFalse(warn_nans.is_shown())

        self._set_var(domain[1])
        self.assertIs(widget.var, domain[1])
        self.assertIs(widget.cvar, domain.class_var)
        np.testing.assert_equal(widget.valid_data, X[50:75, 1])
        np.testing.assert_equal(widget.valid_group_data, Y[50:75])
        self.assertTrue(widget.is_valid)
        self.assertFalse(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertTrue(warn_nans.is_shown())

        self._set_var(domain[2])
        self.assertIs(widget.var, domain[2])
        self.assertIs(widget.cvar, domain.class_var)
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertFalse(widget.is_valid)
        self.assertFalse(err_def_var.is_shown())
        self.assertTrue(err_def_pair.is_shown())
        self.assertFalse(warn_nans.is_shown())

        self._set_var(domain[3])
        self.assertIs(widget.var, domain[3])
        self.assertIs(widget.cvar, domain.class_var)
        np.testing.assert_equal(widget.valid_data, X[:75, 3])
        np.testing.assert_equal(widget.valid_group_data, Y[:75])
        self.assertTrue(widget.is_valid)
        self.assertFalse(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertTrue(warn_nans.is_shown())

        self._set_var(domain[0])
        self._set_cvar(None)
        self.assertIs(widget.var, domain[0])
        self.assertIsNone(widget.cvar)
        self.assertIsNone(widget.valid_data)
        self.assertIsNone(widget.valid_group_data)
        self.assertFalse(widget.is_valid)
        self.assertTrue(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertFalse(warn_nans.is_shown())

        self._set_var(domain[1])
        self.assertIs(widget.var, domain[1])
        self.assertIsNone(widget.cvar)
        np.testing.assert_equal(widget.valid_data, X[50:, 1])
        self.assertIsNone(widget.valid_group_data)
        self.assertTrue(widget.is_valid)
        self.assertFalse(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertTrue(warn_nans.is_shown())

        self._set_var(domain[3])
        self.assertIs(widget.var, domain[3])
        self.assertIsNone(widget.cvar)
        np.testing.assert_equal(widget.valid_data, X[:, 3])
        self.assertIsNone(widget.valid_group_data)
        self.assertTrue(widget.is_valid)
        self.assertFalse(err_def_var.is_shown())
        self.assertFalse(err_def_pair.is_shown())
        self.assertFalse(warn_nans.is_shown())

    def test_controls_disabling(self):
        """Widget changes gui for continuous/discrete variables and grouping"""
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)

        cont = self.iris.domain[0]
        disc = self.iris.domain.class_var
        cont_box = widget.continuous_box
        sort_by_freq = widget.controls.sort_by_freq
        show_probs = widget.controls.show_probs
        stacked = widget.controls.stacked_columns

        self._set_var(cont)
        self._set_cvar(disc)
        self.assertFalse(sort_by_freq.isEnabled())
        self.assertTrue(cont_box.isEnabled())
        self.assertTrue(show_probs.isEnabled())
        self.assertTrue(stacked.isEnabled())

        self._set_var(cont)
        self._set_cvar(None)
        self.assertFalse(sort_by_freq.isEnabled())
        self.assertTrue(cont_box.isEnabled())
        self.assertFalse(show_probs.isEnabled())
        self.assertFalse(stacked.isEnabled())

        self._set_var(disc)
        self._set_cvar(None)
        self.assertTrue(sort_by_freq.isEnabled())
        self.assertFalse(cont_box.isEnabled())
        self.assertFalse(show_probs.isEnabled())
        self.assertFalse(stacked.isEnabled())

        self._set_var(disc)
        self._set_cvar(disc)
        self.assertTrue(sort_by_freq.isEnabled())
        self.assertFalse(cont_box.isEnabled())
        self.assertTrue(show_probs.isEnabled())
        self.assertTrue(stacked.isEnabled())

    if os.getenv("CI"):
        # Testing all combinations takes almost a minute; this should take < 2s
        # Code for fitter, stacked_columns and show_probs is independent, so
        # changing them simultaneously doesn't significantly degrade the tests
        def test_plot_types_combinations(self):
            """Check that the widget doesn't crash at any plot combination"""
            widget = self.widget
            c = widget.controls
            self.send_signal(widget.Inputs.data, self.iris)
            cont = self.iris.domain[0]
            disc = self.iris.domain.class_var
            for var in (cont, disc):
                for cvar in (disc, None):
                    for b in [0, 1]:
                        self._set_var(var)
                        self._set_cvar(cvar)
                        self._set_fitter(2 * b)
                        self._set_check(c.stacked_columns, b)
                        self._set_check(c.show_probs, b)
                        self._set_check(c.sort_by_freq, b)
                        widget.grab()  # run layout and paint
    else:
        def test_plot_types_combinations(self):
            """Check that the widget doesn't crash at any plot combination"""
            # pylint: disable=too-many-nested-blocks
            widget = self.widget
            c = widget.controls
            set_chk = self._set_check
            self.send_signal(widget.Inputs.data, self.iris)
            cont = self.iris.domain[0]
            disc = self.iris.domain.class_var
            for var in (cont, disc):
                for cvar in (disc, None):
                    for fitter in [0, 2]:
                        for cumulative in [False, True]:
                            for stack in [False, True]:
                                for show_probs in [False, True]:
                                    for sort_by_freq in [False, True]:
                                        self._set_var(var)
                                        self._set_cvar(cvar)
                                        self._set_fitter(fitter)
                                        set_chk(c.cumulative_distr, cumulative)
                                        set_chk(c.stacked_columns, stack)
                                        set_chk(c.show_probs, show_probs)
                                        set_chk(c.sort_by_freq, sort_by_freq)
                                        widget.grab()  # run layout and paint

    def test_selection_grouping(self):
        """Widget groups consecutive selected bars"""
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        self._set_slider(0)
        widget.selected_bars = {widget.ordered_values[x]
                                for x in [1, 2, 3, 5, 6, 9]}
        widget.plot_mark.addItem = Mock()
        widget.show_selection()
        widget._on_end_selecting()
        self.assertEqual(widget.plot_mark.addItem.call_count, 3)
        out_selected = self.get_output(widget.Outputs.selected_data)
        self.assertEqual(
            len(out_selected.domain[ANNOTATED_DATA_FEATURE_NAME].values), 3)

    def test_disable_hide_bars(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        domain = self.iris.domain

        cb = widget.controls.hide_bars

        for var in ("petal length", "iris"):
            for fitter in (0, 1):
                self._set_var(domain[var])
                self._set_fitter(fitter)
                self.assertEqual(cb.isEnabled(),
                                 var == "petal length" and fitter == 1)

    def test_hide_bars(self):
        # bar is a valid name in this context, pylint: disable=blacklisted-name
        widget = self.widget
        cb = widget.controls.hide_bars
        self._set_check(cb, True)
        self._set_fitter(1)

        self.send_signal(widget.Inputs.data, self.iris)

        domain = self.iris.domain
        self._set_var(domain["petal length"])

        for cvar in (None, domain["iris"]):
            self._set_cvar(cvar)
            self.assertTrue(cb.isEnabled())
            self.assertTrue(all(bar.hidden
                                for bar in widget.bar_items))
            self.assertTrue(all(curve.opts["brush"] is not None
                                for curve in widget.curve_items))

            self._set_check(cb, False)
            self.assertTrue(all(not bar.hidden
                                for bar in widget.bar_items))
            self.assertTrue(all(curve.opts["brush"] is None or
                                curve.opts["brush"].style() == Qt.NoBrush
                                for curve in widget.curve_items))

            self._set_check(cb, True)
            self.assertTrue(all(bar.hidden
                                for bar in widget.bar_items))
            self.assertTrue(all(curve.opts["brush"] is not None
                                for curve in widget.curve_items))

        self._set_fitter(1)
        self._set_check(widget.controls.hide_bars, True)
        self.assertTrue(all(bar.hidden for bar in widget.bar_items))

        self._set_fitter(0)
        self.assertTrue(all(not bar.hidden for bar in widget.bar_items))

        self._set_fitter(1)
        self.assertTrue(all(bar.hidden for bar in widget.bar_items))

    def test_report(self):
        """Report doesn't crash"""
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        widget.send_report()

    def test_sort_by_freq_no_split(self):
        domain = self.heart.domain
        sort_by_freq = self.widget.controls.sort_by_freq

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._set_var(domain["gender"])
        self._set_cvar(None)

        self._set_check(sort_by_freq, False)
        out = self.get_output(self.widget.Outputs.histogram_data)
        self.assertEqual(out[0][0], "female")
        self.assertEqual(out[0][1], 97)
        self.assertEqual(out[1][0], "male")
        self.assertEqual(out[1][1], 206)

        self._set_check(sort_by_freq, True)
        out = self.get_output(self.widget.Outputs.histogram_data)
        self.assertEqual(out[0][0], "male")
        self.assertEqual(out[0][1], 206)
        self.assertEqual(out[1][0], "female")
        self.assertEqual(out[1][1], 97)

    def test_sort_by_freq_split(self):
        domain = self.heart.domain
        sort_by_freq = self.widget.controls.sort_by_freq

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._set_var(domain["gender"])
        self._set_cvar(domain["rest ECG"])

        self._set_check(sort_by_freq, False)
        out = self.get_output(self.widget.Outputs.histogram_data)
        self.assertEqual(out[0][0], "female")
        self.assertEqual(out[0][1], "normal")
        self.assertEqual(out[0][2], 49)
        self.assertEqual(out[4][0], "male")
        self.assertEqual(out[4][1], "left vent hypertrophy")
        self.assertEqual(out[4][2], 103)

        self._set_check(sort_by_freq, True)
        out = self.get_output(self.widget.Outputs.histogram_data)
        self.assertEqual(out[0][0], "male")
        self.assertEqual(out[0][1], "normal")
        self.assertEqual(out[0][2], 102)
        self.assertEqual(out[4][0], "female")
        self.assertEqual(out[4][1], "left vent hypertrophy")
        self.assertEqual(out[4][2], 45)

    def test_sort_by_freq_output_selection(self):
        widget = self.widget
        sort_by_freq = self.widget.controls.sort_by_freq
        var = self.heart.domain["chest pain"]

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._set_var(var)

        sort_by_freq.setChecked(False)
        assert not widget.sort_by_freq

        # Select value[1]
        widget._on_item_clicked(widget.bar_items[1], Qt.NoModifier, False)
        widget._on_end_selecting()
        cp = self.get_output(widget.Outputs.selected_data).get_column(var)
        self.assertTrue(np.all(cp == 1))

        sort_by_freq.setChecked(True)
        assert widget.sort_by_freq

        # Select value[2] (because of reordering)
        # value[1] remains selected
        widget._on_item_clicked(widget.bar_items[1], Qt.ControlModifier, False)
        widget._on_end_selecting()
        cp = self.get_output(widget.Outputs.selected_data).get_column(var)
        self.assertFalse(np.any(cp == 0))
        self.assertTrue(np.any(cp == 1))
        self.assertTrue(np.any(cp == 2))
        self.assertFalse(np.any(cp == 3))

        # deselect value[1]
        widget._on_item_clicked(widget.bar_items[2], Qt.ControlModifier, False)
        widget._on_end_selecting()
        cp = self.get_output(widget.Outputs.selected_data).get_column(var)
        self.assertTrue(np.all(cp == 2))

        # Select value[0] and also value[1] (!), because it's in between
        # This tests checks that shift-selecting works with ordered values
        widget._on_item_clicked(widget.bar_items[2], Qt.NoModifier, False)
        widget._on_item_clicked(widget.bar_items[0], Qt.ShiftModifier, False)
        widget._on_end_selecting()
        cp = self.get_output(widget.Outputs.selected_data).get_column(var)
        self.assertTrue(np.any(cp == 0))
        self.assertTrue(np.any(cp == 1))
        self.assertTrue(np.any(cp == 2))
        self.assertFalse(np.any(cp == 3))

    def test_keyboard_interaction_unsorted(self):
        press = partial(QKeyEvent, QEvent.KeyPress)
        left, right = Qt.Key_Left, Qt.Key_Right

        widget = self.widget
        sort_by_freq = self.widget.controls.sort_by_freq
        widget.sort_by_freq = False
        var = self.heart.domain["chest pain"]

        for ordered in [False, True]:
            with self.subTest(ordered=ordered):
                sort_by_freq.setChecked(ordered)
                assert widget.sort_by_freq is ordered
                assert not ordered or list(widget.ordered_values) != list(var.values)

                values = widget.ordered_values if ordered else var.values

                self.send_signal(self.widget.Inputs.data, self.heart)
                self._set_var(var)

                # Start selecting by pressing right
                for i in [0, 1, 2, 3, 3, 3]:
                    widget.keyPressEvent(press(right, Qt.NoModifier))
                    self.assertEqual(widget.selected_bars, {values[i]}, f"at i={i}")

                # Going left
                for i in [2, 1, 0, 0, 0]:
                    widget.keyPressEvent(press(left, Qt.NoModifier))
                    self.assertEqual(widget.selected_bars, {values[i]}, f"at i={i}")

                # Deselect first item (= clear selection)
                widget._on_item_clicked(widget.bar_items[0], Qt.NoModifier, False)
                assert not widget.selected_bars

                # Going left from clear
                for i in [3, 2, 1, 0, 0, 0]:
                    widget.keyPressEvent(press(left, Qt.NoModifier))
                    self.assertEqual(widget.selected_bars, {values[i]}, f"at i={i}")

                widget.keyPressEvent(press(right, Qt.NoModifier))
                assert widget.selected_bars == {values[1]}

                # Shift selecting to the right
                widget.keyPressEvent(press(right, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1], values[2]}

                widget.keyPressEvent(press(right, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1], values[2], values[3]}

                widget.keyPressEvent(press(right, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1], values[2], values[3]}

                # Shift deselecting to the left
                widget.keyPressEvent(press(left, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1], values[2]}

                widget.keyPressEvent(press(left, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1]}

                # Now we have a single item, so going left should select the one to the left
                widget.keyPressEvent(press(left, Qt.ShiftModifier))
                assert widget.selected_bars == {values[0], values[1]}

                # Right deselects it
                widget.keyPressEvent(press(right, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1]}

                # Right again selects the item to the right of the single itsm
                widget.keyPressEvent(press(right, Qt.ShiftModifier))
                assert widget.selected_bars == {values[1], values[2]}

    @patch("Orange.widgets.visualize.owdistributions.decimal_binnings")
    def test_selection_with_offset_cont_hist(self, dec_bin):
        widget = self.widget

        dec_bin.return_value = [BinDefinition(np.arange(0, 1000, 100))]
        self.send_signal(Table.from_numpy(Domain([ContinuousVariable("y")]),
                                          np.arange(1000)[:, np.newaxis]))
        widget._on_item_clicked(widget.bar_items[2], Qt.NoModifier, False)
        widget._on_end_selecting()
        np.testing.assert_equal(
            self.get_output(widget.Outputs.selected_data).X,
            np.arange(200, 300)[:, np.newaxis])


if __name__ == "__main__":
    unittest.main()
