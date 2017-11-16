# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owfreeviz import OWFreeViz
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, \
    datasets


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
        self.widget = self.create_widget(OWFreeViz)

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

    def test_error_msg(self):
        data = self.data[:, list(range(len(self.data.domain.attributes)))]
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        data = self.data[:40]
        data.domain.class_var.values = data.domain.class_var.values[0:1]
        data = data.transform(self.data.domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())

    def _select_data(self):
        self.widget.graph.select_by_rectangle(QRectF(QPointF(-20, -20), QPointF(20, 20)))
        return self.widget.graph.get_selection()

    def test_optimization(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.widget.btn_start.click()

    def test_optimization_cancelled(self):
        self.test_optimization()
        self.widget.btn_start.click()

    def test_reset_optimization(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        simulate.combobox_activate_index(self.widget.controls.initialization, 0)
        simulate.combobox_activate_index(self.widget.controls.initialization, 1)

    def test_size_hint(self):
        self.widget.show()

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_subset_data(self):
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.send_signal(self.widget.Inputs.data_subset, self.heart_disease)

    def test_sparse(self):
        table = Table("iris")
        table.X = sp.csr_matrix(table.X)
        self.assertTrue(sp.issparse(table.X))
        self.assertFalse(self.widget.Warning.sparse_not_supported.is_shown())
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Warning.sparse_not_supported.is_shown())

    def test_none_data(self):
        table = Table(
            Domain(
                [ContinuousVariable("a"),
                 DiscreteVariable("b", values=["y", "n"])]
            ),
            list(zip(
                [],
                ""))
        )
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.reset_graph_data()

    def test_class_density(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.widget.cb_class_density.click()

    def test_set_radius_no_data(self):
        """
        Widget should not crash when there is no data and radius slider is moved.
        GH-2780
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        w.rslider.setSliderPosition(3)

    def test_update_graph_no_data(self):
        """
        Widget should not crash when there is no data and one wants to change class density etc.
        GH-2780
        """
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        w.cb_class_density.click()
