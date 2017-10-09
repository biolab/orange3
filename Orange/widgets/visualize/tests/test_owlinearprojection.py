# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import time

import numpy as np

from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets.settings import Context
from Orange.widgets.visualize.owlinearprojection import OWLinearProjection
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets
from Orange.widgets.tests.utils import EventSpy, excepthook_catch, simulate
from Orange.widgets.visualize.utils import Worker


class TestOWLinearProjection(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False
        cls.projection_table = cls._get_projection_table()

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)  # type: OWLinearProjection

    def _select_data(self):
        self.widget.graph.select_by_rectangle(QRectF(QPointF(-20, -20), QPointF(20, 20)))
        return self.widget.graph.get_selection()

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, Table(Table("iris").domain))

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        espy = EventSpy(self.widget, OWLinearProjection.ReplotRequest)
        with excepthook_catch():
            self.send_signal(self.widget.Inputs.data, data)
            # ensure delayed replot request is processed
            if not espy.events():
                assert espy.wait(1000)

        cb = self.widget.graph.controls
        simulate.combobox_run_through_all(cb.attr_color)
        simulate.combobox_run_through_all(cb.attr_size)

        data = data.copy()
        data.X[:, 0] = np.nan
        data.Y[:] = np.nan

        spy = EventSpy(self.widget, OWLinearProjection.ReplotRequest)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data[2:3])
        if not spy.events():
            assert spy.wait()

        with excepthook_catch():
            simulate.combobox_activate_item(cb.attr_color, "X1")

        with excepthook_catch():
            simulate.combobox_activate_item(cb.attr_size, "X1")

    def test_points_combo_boxes(self):
        self.send_signal("Data", self.data)
        graph = self.widget.controls.graph
        self.assertEqual(len(graph.attr_color.model()), 8)
        self.assertEqual(len(graph.attr_shape.model()), 3)
        self.assertEqual(len(graph.attr_size.model()), 6)
        self.assertEqual(len(graph.attr_label.model()), 8)

    def test_buttons(self):
        for btn in self.widget.radio_placement.buttons[:3]:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(btn.isEnabled())
            btn.click()

    def test_rank(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.vizrank.button.click()
        time.sleep(1)

    @classmethod
    def _get_projection_table(cls):
        domain = Domain(attributes=[ContinuousVariable("Attr {}".format(i)) for i in range(4)],
                        metas=[StringVariable("Component")])
        table = Table.from_numpy(domain,
                                 X=np.array([[0.522, -0.263, 0.581, 0.566],
                                             [0.372, 0.926, 0.021, 0.065]]),
                                 metas=[["PC1"], ["PC2"]])
        return table

    def test_projection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.radio_placement.buttons[3].isEnabled())
        self.send_signal(self.widget.Inputs.projection, self.projection_table)
        self.assertTrue(self.widget.radio_placement.buttons[3].isEnabled())
        self.widget.radio_placement.buttons[3].click()
        self.send_signal(self.widget.Inputs.projection, None)
        self.assertFalse(self.widget.radio_placement.buttons[3].isChecked())
        self.assertTrue(self.widget.radio_placement.buttons[0].isChecked())

    def test_projection_error(self):
        domain = Domain(attributes=[ContinuousVariable("Attr {}".format(i)) for i in range(4)],
                        metas=[StringVariable("Component")])
        table = Table.from_numpy(domain,
                                 X=np.array([[0.522, -0.263, 0.581, 0.566]]),
                                 metas=[["PC1"]])
        self.assertFalse(self.widget.Warning.not_enough_components.is_shown())
        self.send_signal(self.widget.Inputs.projection, table)
        self.assertTrue(self.widget.Warning.not_enough_components.is_shown())

    def test_bad_data(self):
        w = self.widget
        data = Table("iris")[:20]
        domain = data.domain
        domain = Domain(
            attributes=domain.attributes[:4], class_vars=DiscreteVariable("class", values=["a"]))
        data = Table.from_numpy(domain=domain, X=data.X, Y=data.Y)
        self.assertTrue(w.radio_placement.buttons[1].isEnabled())
        self.send_signal(w.Inputs.data, data)
        self.assertFalse(w.radio_placement.buttons[1].isEnabled())

    def test_no_data_for_lda(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.radio_placement.buttons[self.widget.Placement.LDA].click()
        self.assertTrue(self.widget.radio_placement.buttons[self.widget.Placement.LDA].isEnabled())
        data = Table("housing")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.radio_placement.buttons[self.widget.Placement.LDA].isEnabled())

    def test_data_no_cont_features(self):
        data = Table("titanic")
        self.assertFalse(self.widget.Warning.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_cont_features.is_shown())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.send_report()

    def test_radius(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.radio_placement.buttons[self.widget.Placement.LDA].click()
        self.widget.rslider.setValue(5)

    def test_metas(self):
        data = Table("iris")
        domain = data.domain
        domain = Domain(attributes=domain.attributes[:3],
                        class_vars=domain.class_vars,
                        metas=domain.attributes[3:])
        data = data.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)

    def test_invalid_data(self):
        def assertErrorShown(data, is_shown):
            self.send_signal(self.widget.Inputs.data, data)
            self.assertEqual(is_shown, self.widget.Error.no_valid_data.is_shown())

        data = Table("iris")[::30]
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
        self.assertEqual(w.graph.attr_color, iris.domain["iris"])
        self.assertEqual(w.graph.attr_shape, iris.domain["iris"])
        self.assertEqual(w.graph.attr_size, iris.domain["sepal length"])

    def test_add_variables(self):
        w = self.widget
        w.variables_selection.add_remove.buttons[1].click()

    def test_set_radius_no_data(self):
        """
        Widget should not crash when there is no data and radius slider is moved.
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        w.rslider.setSliderPosition(3)


class LinProjVizRankTests(WidgetTest):
    """
    Linear Projection VizRank tests are mostly done without threading.
    This is because threads created with module coverage are not traced.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("iris")
        # dom = Domain(cls.iris.domain.attributes, [])
        # cls.iris_no_class = Table(dom, cls.iris)

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)
        self.vizrank = self.widget.vizrank

    def test_discrete_class(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        worker = Worker(self.vizrank)
        self.vizrank.keep_running = True
        worker.do_work()

    def test_continuous_class(self):
        data = Table("housing")[::100]
        self.send_signal(self.widget.Inputs.data, data)
        worker = Worker(self.vizrank)
        self.vizrank.keep_running = True
        worker.do_work()
