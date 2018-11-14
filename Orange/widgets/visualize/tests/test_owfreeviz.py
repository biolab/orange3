# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, AnchorProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owfreeviz import OWFreeViz


class TestOWFreeViz(WidgetTest, AnchorProjectionWidgetTestMixin,
                    WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False
        cls.heart_disease = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWFreeViz)

    def test_error_msg(self):
        data = self.data[:, list(range(len(self.data.domain.attributes)))]
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        data = self.data[:40]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())

    def test_optimization(self):
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.widget.btn_start.click()

    def test_optimization_cancelled(self):
        self.test_optimization()
        self.widget.btn_start.click()

    def test_optimization_reset(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        init = self.widget.controls.initialization
        simulate.combobox_activate_index(init, 0)
        simulate.combobox_activate_index(init, 1)

    def test_constant_data(self):
        data = Table("titanic")[56:59]
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.btn_start.click()
        self.assertTrue(self.widget.Error.constant_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_set_radius_no_data(self):
        """
        Widget should not crash when there is no data and radius slider is moved.
        GH-2780
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        self.widget.graph.controls.hide_radius.setSliderPosition(3)

    def test_output_components(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        components = self.get_output(self.widget.Outputs.components)
        domain = components.domain
        self.assertEqual(domain.attributes, self.data.domain.attributes)
        self.assertEqual(domain.class_vars, ())
        self.assertEqual([m.name for m in domain.metas], ["component"])
        X = np.array([[1, 0, -1, 0], [0, 1, 0, -1]]).astype(float)
        np.testing.assert_array_almost_equal(components.X, X)
        metas = [["freeviz-x"], ["freeviz-y"]]
        np.testing.assert_array_equal(components.metas, metas)

    def test_manual_move(self):
        super().test_manual_move()
        array = np.array([[1, 2], [0, 1], [-1, 0], [0, -1]])
        np.testing.assert_array_almost_equal(
            self.get_output(self.widget.Outputs.components).X, array.T)

    def test_discrete_attributes(self):
        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertTrue(self.widget.Warning.removed_features.is_shown())
        self.widget.btn_start.click()
