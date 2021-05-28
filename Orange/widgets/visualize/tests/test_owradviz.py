# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock
import numpy as np

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin,
    AnchorProjectionWidgetTestMixin, datasets
)
from Orange.widgets.visualize.owradviz import OWRadviz, RadvizVizRank


class TestOWRadviz(WidgetTest, AnchorProjectionWidgetTestMixin,
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
        self.widget = self.create_widget(OWRadviz)

    def test_btn_vizrank(self):
        def check_vizrank(data):
            self.send_signal(self.widget.Inputs.data, data)
            if data is not None and data.domain.class_var in \
                    self.widget.controls.attr_color.model():
                self.widget.attr_color = data.domain.class_var
            if self.widget.btn_vizrank.isEnabled():
                vizrank = RadvizVizRank(self.widget)
                states = [state for state in vizrank.iterate_states(None)]
                self.assertIsNotNone(vizrank.compute_score(states[0]))

        check_vizrank(self.data)
        check_vizrank(self.data[:, :3])
        check_vizrank(None)
        for ds in datasets.datasets():
            check_vizrank(ds)

    def test_no_features(self):
        self.send_signal(self.widget.Inputs.data, self.data[:, :0])

    def test_not_enough_instances(self):
        w = self.widget
        self.assertFalse(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, self.data[:1])
        self.assertTrue(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, self.data)
        self.assertFalse(w.Error.no_instances.is_shown())

    def test_saved_features(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        del self.widget.selected_vars[0]
        selected = self.widget.selected_vars[:]

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWRadviz, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.data, widget=w)
        self.assertListEqual(selected, self.widget.selected_vars)
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        selected = self.widget.selected_vars[:]
        names = [a for a in self.heart_disease.domain.attributes
                 if a.is_continuous or a.is_discrete and len(a.values) == 2]
        self.assertListEqual(selected, names[:5])

    def test_output_components(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        components = self.get_output(self.widget.Outputs.components)
        domain = components.domain
        self.assertEqual(domain.attributes, self.data.domain.attributes)
        self.assertEqual(domain.class_vars, ())
        self.assertEqual([m.name for m in domain.metas], ["component"])
        X = np.array([[1, 0, -1, 0], [0, 1, 0, -1],
                      [0, 1.57, 3.14, -1.57]])
        np.testing.assert_array_almost_equal(components.X, X, 2)
        metas = [["radviz-x"], ["radviz-y"], ["angle"]]
        np.testing.assert_array_equal(components.metas, metas)

    def test_manual_move(self):
        super().test_manual_move()
        array = np.array([[0.4472136, 0.894427], [0, 1], [-1, 0], [0, -1]])
        np.testing.assert_array_almost_equal(
            self.get_output(self.widget.Outputs.components).X[:2], array.T)

    def test_discrete_attributes(self):
        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertTrue(self.widget.Warning.removed_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.removed_vars.is_shown())

    def test_saved_selected_vars(self):
        self.send_signal(self.widget.Inputs.data, self.data)

        self.widget.selected_vars[:] = self.data.domain[:1]
        self.widget.model_selected.selection_changed.emit()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(len(self.widget.selected_vars[:]), 1)

        self.widget.selected_vars[:] = self.data.domain[:0]
        self.widget.model_selected.selection_changed.emit()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(len(self.widget.selected_vars[:]), 0)

    def test_invalidated_model_selected(self):
        self.widget.setup_plot = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.widget.selected_vars[:] = self.data.domain[2:]
        self.widget.model_selected.selection_changed.emit()
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data[:, 2:])
        self.widget.setup_plot.assert_not_called()

        self.widget.selected_vars[:] = self.data.domain[3:]
        self.widget.model_selected.selection_changed.emit()
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()

    def test_score_plots_feature_update(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        selected_vars = set(self.widget.selected_vars)
        output1 = self.get_output(self.widget.Outputs.components)
        self.widget.vizrank.toggle()
        self.process_events(until=lambda: not self.widget.vizrank.keep_running)
        self.assertNotEqual(selected_vars, set(self.widget.selected_vars))
        output2 = self.get_output(self.widget.Outputs.components)
        self.assertNotEqual(output1, output2)


if __name__ == "__main__":
    unittest.main()
