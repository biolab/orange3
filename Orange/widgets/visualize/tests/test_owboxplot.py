# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
from unittest.mock import patch

import numpy as np
from AnyQt.QtCore import QItemSelectionModel

from Orange.data import Table, ContinuousVariable, StringVariable, Domain, \
    DiscreteVariable
from Orange.widgets.visualize.owboxplot import (
    OWBoxPlot, FilterGraphicsRectItem, _quantiles
)
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.tests import test_filename


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
        self.assertEqual(len(self.widget.group_vars), 1)
        self.assertTrue(self.widget.display_box.isHidden())
        self.assertFalse(self.widget.stretching_box.isHidden())

    def test_dont_show_hidden_attrs(self):
        """Check widget's data"""
        iris = Table("iris")
        iris.domain["iris"].attributes["hidden"] = True
        iris.domain["petal length"].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, iris)
        self.assertEqual(len(self.widget.attrs), 3)
        self.assertEqual(len(self.widget.group_vars), 1)

    def test_primitive_metas(self):
        new_domain = Domain(attributes=[], class_vars=[], metas=(
            self.data.domain.attributes + self.data.domain.class_vars))
        attrs_as_metas = self.data.transform(new_domain)
        self.send_signal(self.widget.Inputs.data, attrs_as_metas)
        self.assertTrue(self.widget.display_box.isEnabled())

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
        # This is a test and does it at its own risk:
        # pylint: disable=protected-access
        data.domain.attributes[1]._values = []
        self.send_signal("Data", data)
        self.widget.controls.order_by_importance.setChecked(True)
        self._select_list_items(self.widget.controls.attribute)
        self._select_list_items(self.widget.controls.group_var)

    def test_input_data_missings_disc_no_group_var(self):
        """Check widget discrete data with missing values and no group variable"""
        data = self.zoo
        data.domain.class_var = ContinuousVariable("cls")
        data.X[:, 1] = np.nan
        # This is a test and does it at its own risk:
        # pylint: disable=protected-access
        data.domain.attributes[1]._values = []
        self.send_signal("Data", data)
        self._select_list_items(self.widget.controls.attribute)
        self._select_list_items(self.widget.controls.group_var)

    def test_attribute_combinations(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        group_list = self.widget.controls.group_var
        m = group_list.selectionModel()
        for i in range(len(group_list.model())):
            m.setCurrentIndex(group_list.model().index(i), m.ClearAndSelect)
            self._select_list_items(self.widget.controls.attribute)

    def test_apply_sorting_group(self):
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

        select_group(2)  # First attribute

        order_check.setChecked(False)
        self.assertEqual(tuple(attributes),
                         data.domain.class_vars + data.domain.attributes)
        order_check.setChecked(True)
        self.assertEqual([x.name for x in attributes],
                         ['sex', 'survived', 'age', 'status'])
        select_group(1)  # Class
        self.assertEqual([x.name for x in attributes],
                         ['sex', 'status', 'age', 'survived'])

        data = self.heart
        self.send_signal("Data", data)
        select_group(1)  # Class
        order_check.setChecked(True)
        self.assertEqual([x.name for x in attributes],
                         ['thal',
                          'chest pain',
                          'major vessels colored',
                          'ST by exercise',
                          'max HR',
                          'exerc ind ang',
                          'slope peak exc ST',
                          'gender',
                          'age',
                          'rest ECG',
                          'rest SBP',
                          'cholesterol',
                          'fasting blood sugar > 120',
                          'diameter narrowing'])

    def test_apply_sorting_vars(self):
        controls = self.widget.controls
        attr_list = self.widget.attrs
        order_check = controls.order_grouping_by_importance
        groups = self.widget.group_vars

        def select_attr(i):
            attr_selection = controls.attribute.selectionModel()
            attr_selection.setCurrentIndex(
                attr_list.index(i),
                attr_selection.ClearAndSelect)

        data = self.titanic
        self.send_signal("Data", data)

        select_attr(1)  # First attribute

        order_check.setChecked(False)
        self.assertEqual(
            tuple(groups),
            (None, ) + data.domain.class_vars + data.domain.attributes)
        order_check.setChecked(True)
        self.assertIsNone(groups[0])
        self.assertEqual([x.name for x in groups[1:]],
                         ['sex', 'survived', 'age', 'status'])
        select_attr(0)  # Class
        self.assertIsNone(groups[0])
        self.assertEqual([x.name for x in groups[1:]],
                         ['sex', 'status', 'age', 'survived'])

        data = self.heart
        self.send_signal("Data", data)
        select_attr(0)  # Class
        self.assertIsNone(groups[0])
        self.assertEqual([x.name for x in groups[1:]],
                         ['thal',
                          'chest pain',
                          'exerc ind ang',
                          'slope peak exc ST',
                          'gender',
                          'rest ECG',
                          'fasting blood sugar > 120',
                          'diameter narrowing'])

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

    def test_continuous_metas(self):
        domain = self.iris.domain
        metas = domain.attributes[:-1] + (StringVariable("str"),)
        domain = Domain([], domain.class_var, metas)
        data = Table.from_table(domain, self.iris)
        self.send_signal(self.widget.Inputs.data, data)

    def test_label_overlap(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.widget.stretched = False
        self.__select_variable("chest pain")
        self.__select_group("gender")
        self.widget.adjustSize()
        self.widget.layout().activate()
        self.widget.grab()  # ensure that the painting code is run

    def test_empty_groups(self):
        """Test if groups with zero elements are not shown"""
        table = Table(test_filename("datasets/cyber-security-breaches.tab"))
        self.send_signal(self.widget.Inputs.data, table)
        self.__select_variable("US State")
        self.__select_group("US State")
        self.assertEqual(52, len(self.widget.boxes))

        # select rows with US State equal to TX or MO
        use_indexes = np.array([0, 1, 25, 26, 27])
        table.X = table.X[use_indexes]
        self.send_signal(self.widget.Inputs.data, table)
        self.assertEqual(2, len(self.widget.boxes))

    def test_sorting_disc_group_var(self):
        """Test if subgroups are sorted by their size"""
        table = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, table)
        self.__select_variable("gender")
        self.__select_group("chest pain")

        # checkbox not checked - preserve original order of selected grouping attribute
        self.assertListEqual(self.widget.order, [0, 1, 2, 3])

        # checkbox checked - sort by frequencies
        self.widget.controls.sort_freqs.setChecked(True)
        self.assertListEqual(self.widget.order, [0, 2, 1, 3])

    def _select_data(self):
        items = [item for item in self.widget.box_scene.items()
                 if isinstance(item, FilterGraphicsRectItem)]
        items[0].setSelected(True)
        return [100, 103, 104, 108, 110, 111, 112, 115, 116,
                120, 123, 124, 126, 128, 132, 133, 136, 137,
                139, 140, 141, 143, 144, 145, 146, 147, 148]

    def _select_list_items(self, _list):
        model = _list.selectionModel()
        for i in range(len(_list.model())):
            model.setCurrentIndex(_list.model().index(i), model.ClearAndSelect)

    def __select_variable(self, name, widget=None):
        if widget is None:
            widget = self.widget

        self.__select_value(widget.controls.attribute, name)

    def __select_group(self, name, widget=None):
        if widget is None:
            widget = self.widget

        self.__select_value(widget.controls.group_var, name)

    def __select_value(self, list, value):
        m = list.model()
        for i in range(m.rowCount()):
            idx = m.index(i)
            if m.data(idx) == value:
                list.selectionModel().setCurrentIndex(
                    idx, QItemSelectionModel.ClearAndSelect)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'commit') as apply:
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.zoo)
            apply.assert_called()

    def test_stretching(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        enabled = self.widget.controls.stretched.isEnabled

        self.__select_variable("chest pain")
        self.__select_group("gender")
        self.assertTrue(enabled())

        self.__select_variable("gender")
        self.__select_group("gender")
        self.assertFalse(enabled())

        self.__select_variable("gender")
        self.__select_group("chest pain")
        self.assertTrue(enabled())

    def test_value_all_missing_for_group(self):
        """
        This is one of the extreme cases when we have a subgroup value
        where all values in selected variable are missing. Box plot should
        handle this.
        """
        data = Table.from_list(
            Domain([DiscreteVariable("a", values=("v1", "v2", "v3")),
                    DiscreteVariable("b", values=("v3", "v4"))]),
            [[0., 0.],
             [0., 1.],
             [1., np.nan],
             [1., np.nan],
             [2., 0.],
             [2., 0.]])
        self.send_signal(self.widget.Inputs.data, data)

        self.__select_variable("b")
        self.__select_group("a")
        self.assertTupleEqual(self.widget.conts.shape, (3, 2))


class TestUtils(unittest.TestCase):
    def test(self):
        np.testing.assert_array_equal(
            _quantiles(range(1, 8 + 1), [1.] * 8, [0.0, 0.25, 0.5, 0.75, 1.0]),
            [1., 2.5, 4.5, 6.5, 8.]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 8 + 1), [1.] * 8, [0.0, 0.25, 0.5, 0.75, 1.0]),
            [1., 2.5, 4.5, 6.5, 8.]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 4 + 1), [1., 2., 1., 2],
                       [0.0, 0.25, 0.5, 0.75, 1.0]),
            [1.0, 2.0, 2.5, 4.0, 4.0]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 4 + 1), [2., 1., 1., 2.],
                       [0.0, 0.25, 0.5, 0.75, 1.0]),
            [1.0, 1.0, 2.5, 4.0, 4.0]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 4 + 1), [1., 1., 1., 1.],
                       [0.0, 0.25, 0.5, 0.75, 1.0]),
            [1.0, 1.5, 2.5, 3.5, 4.0]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 4 + 1), [1., 1., 1., 1.],
                       [0.0, 0.25, 0.5, 0.75, 1.0], interpolation="higher"),
            [1, 2, 3, 4, 4]
        )
        np.testing.assert_array_equal(
            _quantiles(range(1, 4 + 1), [1., 1., 1., 1.],
                       [0.0, 0.25, 0.5, 0.75, 1.0], interpolation="lower"),
            [1, 1, 2, 3, 4]
        )
