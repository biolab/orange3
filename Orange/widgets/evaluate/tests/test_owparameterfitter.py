# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import patch, Mock

import pyqtgraph as pg

from AnyQt.QtCore import QPointF
from AnyQt.QtGui import QFont
from AnyQt.QtWidgets import QToolTip

from Orange.classification import NaiveBayesLearner
from Orange.data import Table
from Orange.modelling import RandomForestLearner
from Orange.regression import PLSRegressionLearner
from Orange.widgets.evaluate.owparameterfitter import OWParameterFitter
from Orange.widgets.model.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate


class DummyLearner(PLSRegressionLearner):
    def fitted_parameters(self):
        return [
            self.FittedParameter("n_components", "Foo", "foo", int, 1, None),
            self.FittedParameter("n_components", "Bar", "bar", int, 1, 10),
            self.FittedParameter("n_components", "Baz", "baz", int, None, 10),
            self.FittedParameter("n_components", "Qux", "qux", int, None, None)
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

    def test_steps_preview(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[1, 2]")

        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[]")

        self.widget.controls.manual_steps.setText("10, 15, 20, 25")
        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[10, 15, 20, 25]")

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

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.fitted_parameters, [])

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
        self.assertEqual(self.widget.initial_parameters, {})

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
        self.assertEqual(1, 2)


if __name__ == "__main__":
    unittest.main()
