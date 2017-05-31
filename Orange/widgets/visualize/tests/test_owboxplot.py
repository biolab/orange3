# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import numpy as np

from Orange.data import Table, ContinuousVariable, StringVariable, Domain
from Orange.widgets.visualize.owboxplot import OWBoxPlot, FilterGraphicsRectItem
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWBoxPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.iris = Table("iris")
        cls.zoo = Table("zoo")
        cls.housing = Table("housing")
        cls.titanic = Table("titanic")
        cls.heart = Table("heart_disease")
        cls.data = cls.iris
        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWBoxPlot)

    def test_input_data(self):
        """Check widget's data"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(len(self.widget.attrs), 5)
        self.assertEqual(len(self.widget.group_vars), 2)
        self.assertFalse(self.widget.display_box.isHidden())
        self.assertTrue(self.widget.stretching_box.isHidden())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(len(self.widget.attrs), 0)
        self.assertEqual(len(self.widget.group_vars), 0)
        self.assertTrue(self.widget.display_box.isHidden())
        self.assertFalse(self.widget.stretching_box.isHidden())

    def test_input_data_missings_cont_group_var(self):
        """Check widget with continuous data with missing values and group variable"""
        data = self.iris.copy()
        data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        # used to crash, see #1568

    def test_input_data_missings_cont_no_group_var(self):
        """Check widget with continuous data with missing values and no group variable"""
        data = self.housing
        data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        # used to crash, see #1568

    def test_input_data_missings_disc_group_var(self):
        """Check widget with discrete data with missing values and group variable"""
        data = self.zoo
        data.X[:, 1] = np.nan
        data.domain.attributes[1].values = []
        self.send_signal("Data", data)
        self.widget.controls.order_by_importance.setChecked(True)
        self._select_list_items(self.widget.controls.attribute)
        self._select_list_items(self.widget.controls.group_var)

    def test_input_data_missings_disc_no_group_var(self):
        """Check widget discrete data with missing values and no group variable"""
        data = self.zoo
        data.domain.class_var = ContinuousVariable("cls")
        data.X[:, 1] = np.nan
        data.domain.attributes[1].values = []
        self.send_signal("Data", data)
        self.widget.controls.order_by_importance.setChecked(True)
        self._select_list_items(self.widget.controls.attribute)
        self._select_list_items(self.widget.controls.group_var)

    def test_attribute_combinations(self):
        data = Table("anneal")
        self.send_signal(self.widget.Inputs.data, data)
        group_list = self.widget.controls.group_var
        m = group_list.selectionModel()
        for i in range(len(group_list.model())):
            m.setCurrentIndex(group_list.model().index(i), m.ClearAndSelect)
            self._select_list_items(self.widget.controls.attribute)

    def _select_list_items(self, _list):
        model = _list.selectionModel()
        for i in range(len(_list.model())):
            model.setCurrentIndex(_list.model().index(i), model.ClearAndSelect)

    def test_apply_sorting(self):
        controls = self.widget.controls
        group_list = controls.group_var
        order_check = controls.order_by_importance
        attributes = self.widget.attrs

        def select_group(i):
            group_selection = group_list.selectionModel()
            group_selection.setCurrentIndex(
                group_list.model().index(i),
                group_selection.ClearAndSelect)

        data = self.titanic
        self.send_signal("Data", data)

        select_group(0)
        self.assertFalse(order_check.isEnabled())
        select_group(1)
        self.assertTrue(order_check.isEnabled())

        order_check.setChecked(False)
        self.assertEqual(tuple(attributes), data.domain.variables)
        order_check.setChecked(True)
        self.assertEqual([x.name for x in attributes],
                         ['sex', 'survived', 'age', 'status'])
        select_group(4)
        self.assertEqual([x.name for x in attributes],
                         ['sex', 'status', 'age', 'survived'])

        data = self.heart
        self.send_signal("Data", data)
        select_group(len(group_list.model()) - 1)
        order_check.setChecked(True)
        self.assertEqual([x.name for x in attributes],
                         ['thal', 'major vessels colored', 'chest pain',
                          'ST by exercise', 'max HR', 'exerc ind ang',
                          'slope peak exc ST', 'gender', 'age', 'rest SBP',
                          'rest ECG', 'cholesterol',
                          'fasting blood sugar > 120', 'diameter narrowing'])

    def test_box_order_when_missing_stats(self):
        self.widget.compare = 1
        # The widget can't do anything smart here, but shouldn't crash
        self.send_signal(self.widget.Inputs.data, self.iris[49:51])

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        selected_indices = self._select_data()
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.data, self.data)
        np.testing.assert_array_equal(self.get_output(self.widget.Outputs.selected_data).X,
                                      self.data.X[selected_indices])

    def _select_data(self):
        items = [item for item in self.widget.box_scene.items()
                 if isinstance(item, FilterGraphicsRectItem)]
        items[0].setSelected(True)
        return [100, 103, 104, 108, 110, 111, 112, 115, 116,
                120, 123, 124, 126, 128, 132, 133, 136, 137,
                139, 140, 141, 143, 144, 145, 146, 147, 148]

    def test_continuous_metas(self):
        domain = self.iris.domain
        metas = domain.attributes[:-1] + (StringVariable("str"),)
        domain = Domain([], domain.class_var, metas)
        data = Table.from_table(domain, self.iris)
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.controls.order_by_importance.setChecked(True)
