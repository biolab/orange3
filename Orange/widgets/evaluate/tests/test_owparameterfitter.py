# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import patch, Mock

import pyqtgraph as pg

from AnyQt.QtCore import QPointF
from AnyQt.QtGui import QFont
from AnyQt.QtWidgets import QToolTip

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain
from Orange.modelling import RandomForestLearner
from Orange.regression import PLSRegressionLearner
from Orange.widgets.evaluate.owparameterfitter import OWParameterFitter
from Orange.widgets.model.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate


class DummyLearner(PLSRegressionLearner):
    @property
    def fitted_parameters(self):
        return [
            self.FittedParameter("n_components", "Foo", int, 5, None),
            self.FittedParameter("n_components", "Bar", int, 5, 10),
            self.FittedParameter("n_components", "Baz", int, None, 10),
            self.FittedParameter("n_components", "Qux", int, None, None)
        ]


class TestOWParameterFitter(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._heart = Table("heart_disease")
        cls._housing = Table("housing")
        cls._naive_bayes = NaiveBayesLearner()
        cls._pls = PLSRegressionLearner()
        cls._rf = RandomForestLearner()
        cls._dummy = DummyLearner()

    def setUp(self):
        self.widget = self.create_widget(OWParameterFitter)

    def test_init(self):
        self.widget.controls.minimum.setValue(3)
        self.widget.controls.maximum.setValue(6)

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.assertEqual(self.widget.controls.parameter_index.currentText(),
                         "Components")
        self.assertEqual(self.widget.minimum, 3)
        self.assertEqual(self.widget.maximum, 6)

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertEqual(self.widget.controls.parameter_index.currentText(),
                         "")

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_input_no_params(self):
        self.send_signal(self.widget.Inputs.data, self._heart)
        self.send_signal(self.widget.Inputs.learner, self._naive_bayes)
        self.wait_until_finished()
        self.assertTrue(self.widget.Warning.no_parameters.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())

    def test_random_forest(self):
        rf_widget = self.create_widget(OWRandomForest)
        learner = self.get_output(rf_widget.Outputs.learner)

        self.send_signal(self.widget.Inputs.learner, learner)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_classless_data(self):
        data = self._housing
        classless_data = data.transform(Domain(data.domain.attributes))

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.send_signal(self.widget.Inputs.data, classless_data)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.missing_target.is_shown())

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.missing_target.is_shown())

    def test_multiclass_data(self):
        data = self._housing
        multiclass_data = data.transform(Domain(data.domain.attributes[2:],
                                                data.domain.attributes[:2]))

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.send_signal(self.widget.Inputs.data, multiclass_data)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.multiple_targets_data.is_shown())

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.multiple_targets_data.is_shown())

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        x = self.widget.graph._FitterPlot__bar_item_tr.opts["x"]
        self.assertEqual(list(x), [-0.2, 0.8])
        x = self.widget.graph._FitterPlot__bar_item_cv.opts["x"]
        self.assertEqual(list(x), [0.2, 1.2])

    @patch.object(QToolTip, "showText")
    def test_tooltip(self, show_text):
        graph = self.widget.graph

        self.assertFalse(self.widget.graph.help_event(Mock()))
        self.assertIsNone(show_text.call_args)

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        for item in graph.items():
            if isinstance(item, pg.BarGraphItem):
                item.mapFromScene = Mock(return_value=QPointF(0.2, 0.2))

        self.assertTrue(self.widget.graph.help_event(Mock()))
        self.assertIn("Train:", show_text.call_args[0][1])
        self.assertIn("CV:", show_text.call_args[0][1])

        for item in graph.items():
            if isinstance(item, pg.BarGraphItem):
                item.mapFromScene = Mock(return_value=QPointF(0.5, 0.5))
        self.assertFalse(self.widget.graph.help_event(Mock()))

    def test_manual_steps(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        self.widget.controls.manual_steps.setText("1, 2, 3")
        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()

        x = self.widget.graph._FitterPlot__bar_item_tr.opts["x"]
        self.assertEqual(list(x), [-0.2, 0.8, 1.8])
        x = self.widget.graph._FitterPlot__bar_item_cv.opts["x"]
        self.assertEqual(list(x), [0.2, 1.2, 2.2])

    def test_manual_steps_limits(self):
        w = self.widget

        def check(cases):
            for setting, steps in cases:
                w.controls.manual_steps.setText(setting)
                w.controls.manual_steps.returnPressed.emit()
                self.assertEqual(w.steps, steps, f"setting: {setting}")
                self.assertIs(w.Error.manual_steps_error.is_shown(), not steps,
                              f"setting: {setting}")

        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()

        # 5 to None
        simulate.combobox_activate_index(w.controls.parameter_index, 0)
        self.wait_until_finished()
        check([("6, 9, 7", (6, 7, 9)),
               ("6, 9, 7, 3", ()),
               ("6, 9, 7", (6, 7, 9)),
               ("6, 9, 7, 3", ())])

        # None to 10
        simulate.combobox_activate_index(w.controls.parameter_index, 2)
        self.wait_until_finished()
        self.assertFalse(w.Error.manual_steps_error.is_shown())

        check([("12, 1, 3, -5", ()),
               ("1, 3, -5", (-5, 1, 3)),
               ("12, 1, 3, -5", ())])

        # No limits
        simulate.combobox_activate_index(w.controls.parameter_index, 3)
        self.wait_until_finished()

        self.assertEqual(w.steps, (-5, 1, 3, 12))
        self.assertFalse(w.Error.manual_steps_error.is_shown())

        # 5 to 10
        simulate.combobox_activate_index(w.controls.parameter_index, 1)
        self.wait_until_finished()

        self.assertEqual(w.steps, ())
        self.assertTrue(w.Error.manual_steps_error.is_shown())

        check([("12, 8, 7, 5", ()),
               ("8, 7, -5", ()),
               ("8, 7, 5", (5, 7, 8))])

    def test_steps_preview(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.assertEqual(self.widget.range_preview.steps(), (1, 2))

        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()
        self.assertIsNone(self.widget.range_preview.steps())

    def test_on_parameter_changed(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._dummy)
        self.wait_until_finished()

        self.widget.commit.deferred = Mock()

        for i in range(1, 4):
            self.widget.commit.deferred.reset_mock()
            simulate.combobox_activate_index(
                self.widget.controls.parameter_index, i)
            self.wait_until_finished()
            self.widget.commit.deferred.assert_called_once()

    def test_not_enough_data(self):
        self.send_signal(self.widget.Inputs.data, self._housing[:5])
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.not_enough_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())

    def test_unknown_err(self):
        self.send_signal(self.widget.Inputs.data, Table("iris")[:50])
        self.send_signal(self.widget.Inputs.learner, self._rf)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.unknown_err.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

    def test_fitted_parameters(self):
        self.assertEqual(self.widget.fitted_parameters, [])

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.assertEqual(self.widget.fitted_parameters, [])

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.assertEqual(len(self.widget.fitted_parameters), 1)
        self.wait_until_finished()

    def test_initial_parameters(self):
        self.assertEqual(self.widget.initial_parameters, {})

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.assertEqual(self.widget.initial_parameters, {})

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.assertEqual(len(self.widget.initial_parameters), 3)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.learner, self._rf)
        self.assertEqual(len(self.widget.initial_parameters), 13)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(len(self.widget.initial_parameters), 14)

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertEqual(self.widget.initial_parameters, {})

    def test_bounds(self):
        self.widget.controls.minimum.setValue(-3)
        self.widget.controls.maximum.setValue(6)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, None)
        self.widget.controls.minimum.setValue(-3)
        self.widget.controls.maximum.setValue(6)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

    def test_saved_workflow(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._dummy)
        self.wait_until_finished()
        simulate.combobox_activate_index(
            self.widget.controls.parameter_index, 2)
        self.widget.controls.minimum.setValue(3)
        self.widget.controls.maximum.setValue(6)
        self.wait_until_finished()

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWParameterFitter,
                                    stored_settings=settings)
        self.send_signal(widget.Inputs.data, self._housing, widget=widget)
        self.send_signal(widget.Inputs.learner, self._dummy, widget=widget)
        self.wait_until_finished(widget=widget)
        self.assertEqual(widget.controls.parameter_index.currentText(), "Baz")
        self.assertEqual(widget.minimum, 3)
        self.assertEqual(widget.maximum, 6)

    def test_retain_settings(self):
        self.send_signal(self.widget.Inputs.learner, self._dummy)

        controls = self.widget.controls

        def _test():
            self.assertEqual(controls.parameter_index.currentText(), "Bar")
            self.assertEqual(controls.minimum.value(), 6)
            self.assertEqual(controls.maximum.value(), 8)
            self.assertEqual(self.widget.parameter_index, 1)
            self.assertEqual(self.widget.minimum, 6)
            self.assertEqual(self.widget.maximum, 8)

        simulate.combobox_activate_index(controls.parameter_index, 1)
        controls.minimum.setValue(6)
        controls.maximum.setValue(8)

        self.send_signal(self.widget.Inputs.data, self._housing)
        _test()

        self.send_signal(self.widget.Inputs.learner,
                         DummyLearner(n_components=6))
        _test()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self._housing)
        _test()

        self.send_signal(self.widget.Inputs.learner, self._rf)
        self.assertEqual(controls.parameter_index.currentText(),
                         "Number of trees")
        self.assertEqual(controls.minimum.value(), 1)
        self.assertEqual(controls.maximum.value(), 10)
        self.assertEqual(self.widget.parameter_index, 0)
        self.assertEqual(self.widget.minimum, 1)
        self.assertEqual(self.widget.maximum, 10)

    def test_visual_settings(self):
        graph = self.widget.graph

        def test_settings():
            font = QFont("Helvetica", italic=True, pointSize=20)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.label.font(), font)
            font.setPointSize(15)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.style["tickFont"], font)
            font.setPointSize(17)
            for legend_item in graph.parameter_setter.legend_items:
                self.assertFontEqual(legend_item[1].item.font(), font)
            self.assertFalse(graph.getAxis("left").grid)

        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Legend", "Font size"), 17
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Legend", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Gridlines", "Show"), False
        self.widget.set_visual_settings(key, value)
        key, value = ("Figure", "Gridlines", "Opacity"), 20
        self.widget.set_visual_settings(key, value)

        test_settings()

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.send_signal(self.widget.Inputs.data, self._heart[:10])
        test_settings()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.learner, None)

        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.send_signal(self.widget.Inputs.data, self._heart[:10])
        test_settings()

    def assertFontEqual(self, font1: QFont, font2: QFont):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    def test_send_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.send_signal(self.widget.Inputs.learner, self._naive_bayes)
        self.wait_until_finished()
        self.widget.send_report()

    def test_steps_from_range_error(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._heart)
        self.send_signal(w.Inputs.learner, self._dummy)
        w.type = w.FROM_RANGE

        w.minimum = 10
        w.maximum = 5
        self.assertEqual(w.steps, ())
        self.assertTrue(w.Error.min_max_error.is_shown())

        w.maximum = 15
        self.assertNotEqual(w.steps, ())
        self.assertFalse(w.Error.min_max_error.is_shown())

        w.minimum = 10
        w.maximum = 5
        w.steps  # pylint: disable=pointless-statement
        self.assertTrue(w.Error.min_max_error.is_shown())

        self.send_signal(w.Inputs.learner, None)
        self.assertFalse(w.Error.min_max_error.is_shown())

    def test_steps_from_ranges_steps(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._heart)
        self.send_signal(w.Inputs.learner, self._dummy)
        w.type = w.FROM_RANGE

        for mini, maxi, exp in [
            (1, 2, (1, 2)),
            (1, 5, (1, 2, 3, 4, 5)),
            (1, 10, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
            (2, 14, (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)),
            (2, 20, (2, 10, 20)),
            (2, 22, (2, 10, 20, 22)),
            (2, 10, (2, 3, 4, 5, 6, 7, 8, 9, 10)),
            (2, 5, (2, 3, 4, 5)),
            (2, 4, (2, 3, 4)),
            (1, 1, (1,)),
            (1, 50, (1, 10, 20, 30, 40, 50)),
            (3, 49, (3, 10, 20, 30, 40, 49)),
            (9, 31, (9, 10, 20, 30, 31)),
            (90, 398, (90, 100, 200, 300, 398)),
            (90, 1010,
             (90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1010)),
            (810, 1234, (810, 900, 1000, 1100, 1200, 1234)),
            (4980, 18030,
             (4980, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
              13000, 14000, 15000, 16000, 17000, 18000, 18030))]:
            w.minimum = mini
            w.maximum = maxi
            self.assertEqual(w.steps, exp, f"min={mini}, max={maxi}")

    def test_steps_from_manual_error(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()
        simulate.combobox_activate_index(w.controls.parameter_index, 3)
        w.type = w.MANUAL

        w.manual_steps = "1, 2, 3, asdf, 4, 5"
        self.assertEqual(w.steps, ())
        self.assertTrue(w.Error.manual_steps_error.is_shown())

        w.manual_steps = "1, 2, 3, 4, 5"
        self.assertNotEqual(w.steps, ())
        self.assertFalse(w.Error.manual_steps_error.is_shown())

        w.manual_steps = "1, 2, 3, asdf, 4, 5"
        w.steps  # pylint: disable=pointless-statement
        self.assertTrue(w.Error.manual_steps_error.is_shown())

        self.send_signal(w.Inputs.learner, None)
        self.assertFalse(w.Error.manual_steps_error.is_shown())

    def test_steps_from_manual_no_dots(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()
        simulate.combobox_activate_index(w.controls.parameter_index, 3)
        w.type = w.MANUAL

        w.manual_steps = "1, 2, 3, 4, 5"
        self.assertEqual(w.steps, (1, 2, 3, 4, 5))

        w.manual_steps = "1, 2, 3, 4, 5, 6"
        self.assertEqual(w.steps, (1, 2, 3, 4, 5, 6))

        w.manual_steps = "1, 2, 10,   3, 4, 123, 5, 6"
        self.assertEqual(w.steps, (1, 2, 3, 4, 5, 6, 10, 123))

    def test_steps_from_manual_dots(self):
        def check(cases):
            for settings, steps in cases:
                w.manual_steps = settings
                self.assertEqual(w.steps, steps, f"setting: {settings}")
                self.assertIs(w.Error.manual_steps_error.is_shown(), not steps,
                              f"setting: {settings}")

        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()
        w.type = w.MANUAL

        # No limits
        simulate.combobox_activate_index(w.controls.parameter_index, 3)
        self.wait_until_finished()
        check([("1, 2, ..., 5", (1, 2, 3, 4, 5)),
               ("1, 2, 3, ..., 5, 6, 7", (1, 2, 3, 4, 5, 6, 7)),
               ("3, ..., 5, 6", (3, 4, 5, 6)),
               ("..., 5, 6", ()),
               ("5, 6, ...", ()),
               ("1, 2, 3, 4, 5, ...", ()),
               ("1, ..., 5", ()),
               ("1, 2, ..., 5, 6, ..., 8", ())])

        # 5 to 10
        simulate.combobox_activate_index(w.controls.parameter_index, 1)
        self.wait_until_finished()
        check([("4, 5, ..., 8", ()),
               ("5, 6, ..., 12", ()),
               ("5, 6, ..., 9", (5, 6, 7, 8, 9)),
               ("6, 7, ..., 9", (6, 7, 8, 9)),
               ("6, 7, ..., 8, 9", (6, 7, 8, 9)),
               ("..., 8, 9", (5, 6, 7, 8, 9)),
               ("6, 7, ...", (6, 7, 8, 9, 10)),
               ("6, 7, 8, 9, ...", (6, 7, 8, 9, 10)),
               ("..., 8, 9", (5, 6, 7, 8, 9)),
               ])

        # 5 to None
        simulate.combobox_activate_index(w.controls.parameter_index, 0)
        self.wait_until_finished()
        check([("4, 5, ..., 8", ()),
               ("5, 6, ..., 12", (5, 6, 7, 8, 9, 10, 11, 12)),
               ("6, 7, ..., 9", (6, 7, 8, 9)),
               ("6, 7, ..., 8, 9", (6, 7, 8, 9)),
               ("..., 8, 9", (5, 6, 7, 8, 9)),
               ("6, 7, ...", ())
               ])

        # None to 10
        simulate.combobox_activate_index(w.controls.parameter_index, 2)
        self.wait_until_finished()
        check([("4, 5, ..., 8", (4, 5, 6, 7, 8)),
               ("5, 6, ..., 12", ()),
               ("5, 6, ..., 9", (5, 6, 7, 8, 9)),
               ("6, 7, ..., 9", (6, 7, 8, 9)),
               ("..., 8, 9", ()),
               ("6, 7, ...", (6, 7, 8, 9, 10)),
               ("6, 7, 8, 9, ...", (6, 7, 8, 9, 10)),
               ("..., 8, 9", ()),
               ])

    def test_steps_from_manual_dots_corrections(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()
        w.type = w.MANUAL

        # 5 to 10
        simulate.combobox_activate_index(w.controls.parameter_index, 1)
        self.wait_until_finished()

        for settings, steps in [
            ("5, 6..., 8", (5, 6, 7, 8)),
            ("5,6...,8", (5, 6, 7, 8)),
            ("5,6...8", (5, 6, 7, 8)),
            ("5,   6  ...  8", (5, 6, 7, 8)),
            ("5,   6  ...  8", (5, 6, 7, 8)),
            ("5,   6  ...  ", (5, 6, 7, 8, 9, 10)),
            ("..., 7, 8", (5, 6, 7, 8)),
            ("..., 7, 8, ...", ()),
            ("5, 6, ..., 7, 8, ...", ()),
            ("5, 6, 8, ...", ()),
            ("5, 6, 8, ...", ()),
            ("5, 6, ..., 8, 10", ()),
            ("5, 7, ..., 8, 10", ()),
            ("8, 7, 6, ...", ()),
            ("5, 6, 7, ..., 7, 8", ()),
        ]:
            w.manual_steps = settings
            self.assertEqual(w.steps, steps, f"setting: {settings}")
            self.assertIs(w.Error.manual_steps_error.is_shown(), not steps,
                          f"setting: {settings}")

    def test_manual_tooltip(self):
        w: OWParameterFitter = self.widget
        self.send_signal(w.Inputs.data, self._housing)
        self.send_signal(w.Inputs.learner, self._dummy)
        self.wait_until_finished()

        simulate.combobox_activate_index(w.controls.parameter_index, 0)
        self.assertIn("greater or equal to 5", w.edit.toolTip())

        simulate.combobox_activate_index(w.controls.parameter_index, 1)
        self.assertIn("between 5 and 10", w.edit.toolTip())

        simulate.combobox_activate_index(w.controls.parameter_index, 2)
        self.assertIn("smaller or equal to 10", w.edit.toolTip())

        simulate.combobox_activate_index(w.controls.parameter_index, 3)
        self.assertEqual("", w.edit.toolTip())


if __name__ == "__main__":
    unittest.main()
