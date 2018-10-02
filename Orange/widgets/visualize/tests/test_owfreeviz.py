# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owfreeviz import OWFreeViz


class TestOWFreeViz(WidgetTest, WidgetOutputsTestMixin,
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
        self.widget = self.create_widget(OWFreeViz)

    def test_error_msg(self):
        data = self.data[:, list(range(len(self.data.domain.attributes)))]
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        data = self.data[:40]
        domain = self.data.domain.copy()
        domain.class_var.values = self.data.domain.class_var.values[:1]
        data = data.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())

    def _select_data(self):
        rect = QRectF(QPointF(-20, -20), QPointF(20, 20))
        self.widget.graph.select_by_rectangle(rect)
        return self.widget.graph.get_selection()

    def test_optimization(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.btn_start.click()

    def test_optimization_cancelled(self):
        self.test_optimization()
        self.widget.btn_start.click()

    def test_optimization_reset(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        init = self.widget.controls.initialization
        simulate.combobox_activate_index(init, 0)
        simulate.combobox_activate_index(init, 1)

    def test_sparse(self):
        table = Table("iris")
        table.X = sp.csr_matrix(table.X)
        self.assertTrue(sp.issparse(table.X))
        self.assertFalse(self.widget.Error.sparse_data.is_shown())
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.sparse_data.is_shown())

    def test_set_radius_no_data(self):
        """
        Widget should not crash when there is no data and radius slider is moved.
        GH-2780
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        self.widget.graph.controls.radius.setSliderPosition(3)
