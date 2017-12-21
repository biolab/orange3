# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, \
    datasets
from Orange.widgets.visualize.owradviz import OWRadviz


class TestOWFreeViz(WidgetTest, WidgetOutputsTestMixin):
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

    def test_points_combo_boxes(self):
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        graph = self.widget.controls.graph
        self.assertEqual(len(graph.attr_color.model()), 17)
        self.assertEqual(len(graph.attr_shape.model()), 11)
        self.assertEqual(len(graph.attr_size.model()), 8)
        self.assertEqual(len(graph.attr_label.model()), 17)

    def test_ugly_datasets(self):
        self.send_signal(self.widget.Inputs.data, Table(datasets.path("testing_dataset_cls")))
        self.send_signal(self.widget.Inputs.data, Table(datasets.path("testing_dataset_reg")))

    def test_btn_vizrank(self):
        # TODO: fix this
        w = self.widget
        def assertEnabled(data, is_enabled):
            self.send_signal(w.Inputs.data, data)
            self.assertEqual(is_enabled, w.btn_vizrank.isEnabled())

        data = self.data
        for data, is_enabled in zip([data[:, :3], data, None], [False, False, False]):
            assertEnabled(data, is_enabled)

    def _select_data(self):
        self.widget.graph.select_by_rectangle(QRectF(QPointF(-20, -20), QPointF(20, 20)))
        return self.widget.graph.get_selection()

    def test_subset_data(self):
        w = self.widget
        data = Table("iris")
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.data_subset, data[::30])

    def test_no_features(self):
        w = self.widget
        data = Table("iris")
        domain = Domain(attributes=data.domain.attributes[:1], class_vars=data.domain.class_vars)
        data2 = data.transform(domain)
        self.assertFalse(w.Error.no_features.is_shown())
        self.send_signal(w.Inputs.data, data2)
        self.assertTrue(w.Error.no_features.is_shown())
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.no_features.is_shown())

    def test_not_enough_instances(self):
        w = self.widget
        data = Table("iris")
        self.assertFalse(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, data[:1])
        self.assertTrue(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, data)
        self.assertFalse(w.Error.no_instances.is_shown())
