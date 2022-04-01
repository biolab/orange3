# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import Mock
import numpy as np

from AnyQt.QtCore import QItemSelectionModel

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.settings import Context
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, datasets,
    AnchorProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owlinearprojection import (
    OWLinearProjection, LinearProjectionVizRank, Placement
)
from Orange.widgets.visualize.utils import run_vizrank


class TestOWLinearProjection(WidgetTest, AnchorProjectionWidgetTestMixin,
                             WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)  # type: OWLinearProjection

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        self.send_signal(self.widget.Inputs.data, data)
        simulate.combobox_run_through_all(self.widget.controls.attr_color)
        simulate.combobox_run_through_all(self.widget.controls.attr_size)

        with data.unlocked():
            data.X[:, 0] = np.nan
            data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data[2:3])
        simulate.combobox_run_through_all(self.widget.controls.attr_color)
        simulate.combobox_run_through_all(self.widget.controls.attr_size)

    def test_buttons(self):
        for btn in self.widget.radio_placement.buttons[:3]:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(btn.isEnabled())
            btn.click()

    def test_btn_vizrank(self):
        def check_vizrank(data):
            self.send_signal(self.widget.Inputs.data, data)
            if data is not None and data.domain.class_var in \
                    self.widget.controls.attr_color.model():
                self.widget.attr_color = data.domain.class_var
            if self.widget.btn_vizrank.isEnabled():
                vizrank = LinearProjectionVizRank(self.widget)
                states = [state for state in vizrank.iterate_states(None)]
                self.assertIsNotNone(vizrank.compute_score(states[0]))

        check_vizrank(self.data)
        check_vizrank(self.data[:, :3])
        check_vizrank(None)
        for ds in datasets.datasets():
            check_vizrank(ds)

    def test_bad_data(self):
        w = self.widget
        data = Table("iris")[:20]
        domain = data.domain
        domain = Domain(
            attributes=domain.attributes[:4], class_vars=DiscreteVariable("class", values=("a", )))
        data = Table.from_numpy(domain=domain, X=data.X, Y=data.Y)
        self.assertTrue(w.radio_placement.buttons[1].isEnabled())
        self.send_signal(w.Inputs.data, data)
        self.assertFalse(w.radio_placement.buttons[1].isEnabled())

    def test_no_data_for_lda(self):
        buttons = self.widget.radio_placement.buttons
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.radio_placement.buttons[Placement.LDA].click()
        self.assertTrue(buttons[Placement.LDA].isEnabled())
        output = self.get_output(self.widget.Outputs.components)
        self.assertTrue(output and len(output) == 2)
        self.send_signal(self.widget.Inputs.data, Table("housing"))
        self.assertFalse(buttons[Placement.LDA].isEnabled())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertTrue(buttons[Placement.LDA].isEnabled())

    def test_data_no_cont_features(self):
        data = Table("titanic")
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

    def test_radius(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.radio_placement.buttons[Placement.LDA].click()
        self.widget.controls.graph.hide_radius.setValue(5)

    def test_invalid_data(self):
        def assertErrorShown(data, is_shown):
            self.send_signal(self.widget.Inputs.data, data)
            self.assertEqual(is_shown, self.widget.Error.no_valid_data.is_shown())

        data = Table("iris")[::30].copy()
        with data.unlocked():
            data[:, 0] = np.nan
        for data, is_shown in zip([None, data, Table("iris")[:30]], [False, True, False]):
            assertErrorShown(data, is_shown)

    def test_migrate_settings_from_version_1(self):
        # Settings from Orange 3.4.0
        settings = {
            '__version__': 1,
            'alpha_value': 255,
            'auto_commit': True,
            'class_density': False,
            'context_settings': [
                Context(attributes={'iris': 1,
                                    'petal length': 2, 'petal width': 2,
                                    'sepal length': 2, 'sepal width': 2},
                        metas={},
                        ordered_domain=[('sepal length', 2),
                                        ('sepal width', 2),
                                        ('petal length', 2),
                                        ('petal width', 2),
                                        ('iris', 1)],
                        time=1504865133.098991,
                        values={'__version__': 1,
                                'color_index': (5, -2),
                                'shape_index': (1, -2),
                                'size_index': (1, -2),
                                'variable_state': ({}, -2)})],
            'jitter_value': 0,
            'legend_anchor': ((1, 0), (1, 0)),
            'point_size': 8,
            'savedWidgetGeometry': None
        }
        w = self.create_widget(OWLinearProjection, stored_settings=settings)
        iris = Table("iris")
        self.send_signal(w.Inputs.data, iris, widget=w)
        self.assertEqual(w.graph.point_width, 8)
        self.assertEqual(w.attr_color, iris.domain["iris"])
        self.assertEqual(w.attr_shape, iris.domain["iris"])
        self.assertEqual(w.attr_size, iris.domain["sepal length"])

    def test_set_radius_no_data(self):
        """
        Widget should not crash when there is no data and radius slider is moved.
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        w.controls.graph.hide_radius.setSliderPosition(3)

    def test_invalidated_model_selected(self):
        model = self.widget.model_selected

        self.widget.setup_plot = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.widget.selected_vars[:] = self.data.domain[2:3]
        model.selection_changed.emit()
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data[:, 2:])
        self.widget.setup_plot.assert_not_called()

        self.widget.selected_vars[:] = [self.data.domain[3]]
        model.selection_changed.emit()
        self.widget.setup_plot.assert_called_once()

        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()

    def test_two_classes_dataset(self):
        self.widget.radio_placement.buttons[1].click()
        self.send_signal(self.widget.Inputs.data, Table("heart_disease"))
        self.assertFalse(self.widget.radio_placement.buttons[1].isEnabled())

    def test_unique_name(self):
        data = Table("iris")
        new = ContinuousVariable("C-y")
        d = Table.from_numpy(Domain(list(data.domain.attributes[:3])+[new],
                                    class_vars=data.domain.class_vars), data.X,
                             data.Y)
        self.send_signal(self.widget.Inputs.data, d)
        output = self.get_output(self.widget.Outputs.annotated_data)
        metas = ["C-x (1)", "C-y (1)", "Selected"]
        self.assertEqual([meta.name for meta in output.domain.metas], metas)


class LinProjVizRankTests(WidgetTest):
    """
    Linear Projection VizRank tests are mostly done without threading.
    This is because threads created with module coverage are not traced.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)
        self.vizrank = self.widget.vizrank

    def test_discrete_class(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        run_vizrank(self.vizrank.compute_score,
                    self.vizrank.iterate_states, None,
                    [], 0, self.vizrank.state_count(), Mock())

    def test_continuous_class(self):
        data = Table("housing")[::100]
        self.send_signal(self.widget.Inputs.data, data)
        run_vizrank(self.vizrank.compute_score,
                    self.vizrank.iterate_states, None,
                    [], 0, self.vizrank.state_count(), Mock())

    def test_set_attrs(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        prev_selected = self.widget.selected_vars[:]
        c1 = self.get_output(self.widget.Outputs.components)
        self.vizrank.toggle()
        self.process_events(until=lambda: not self.vizrank.keep_running)
        self.assertEqual(len(self.vizrank.scores), self.vizrank.state_count())
        self.vizrank.rank_table.selectionModel().select(
            self.vizrank.rank_model.item(0, 0).index(),
            QItemSelectionModel.ClearAndSelect
        )
        self.assertNotEqual(self.widget.selected_vars, prev_selected)
        c2 = self.get_output(self.widget.Outputs.components)
        self.assertNotEqual(c1.domain.attributes, c2.domain.attributes)


if __name__ == "__main__":
    import unittest
    unittest.main()
