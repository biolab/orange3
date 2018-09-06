# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table, Domain
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.visualize.owradviz import OWRadviz


class TestOWRadviz(WidgetTest, WidgetOutputsTestMixin,
                   ProjectionWidgetTestMixin):
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

    def test_no_features(self):
        w = self.widget
        data2 = self.data.transform(Domain(self.data.domain.attributes[:1],
                                           self.data.domain.class_vars))
        self.assertFalse(w.Error.no_features.is_shown())
        self.send_signal(w.Inputs.data, data2)
        self.assertTrue(w.Error.no_features.is_shown())
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.no_features.is_shown())

    def test_not_enough_instances(self):
        w = self.widget
        self.assertFalse(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, self.data[:1])
        self.assertTrue(w.Error.no_instances.is_shown())
        self.send_signal(w.Inputs.data, self.data)
        self.assertFalse(w.Error.no_instances.is_shown())
