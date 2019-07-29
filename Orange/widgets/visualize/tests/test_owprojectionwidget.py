# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import (
    Table, ContinuousVariable, DiscreteVariable, StringVariable, Domain
)
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.visualize.utils.widget import (
    OWDataProjectionWidget, OWProjectionWidgetBase
)


class TestOWProjectionWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWProjectionWidgetBase)

    def test_get_column(self):
        widget = self.widget
        get_column = widget.get_column

        cont = ContinuousVariable("cont")
        disc = DiscreteVariable("disc", list("abcdefghijklmno"))
        disc2 = DiscreteVariable("disc2", list("abc"))
        disc3 = DiscreteVariable("disc3", list("abc"))
        string = StringVariable("string")
        domain = Domain([cont, disc], disc2, [disc3, string])

        widget.data = Table.from_numpy(
            domain,
            np.array([[1, 4], [2, 15], [6, 7]], dtype=float),
            np.array([2, 1, 0], dtype=float),
            np.array([[0, "foo"], [2, "bar"], [1, "baz"]])
        )

        self.assertIsNone(get_column(None))
        np.testing.assert_almost_equal(get_column(cont), [1, 2, 6])
        np.testing.assert_almost_equal(get_column(disc), [4, 15, 7])
        np.testing.assert_almost_equal(get_column(disc2), [2, 1, 0])
        np.testing.assert_almost_equal(get_column(disc3), [0, 2, 1])
        self.assertEqual(list(get_column(string)), ["foo", "bar", "baz"])

        widget.valid_data = np.array([True, False, True])

        self.assertIsNone(get_column(None))
        np.testing.assert_almost_equal(get_column(cont), [1, 6])
        self.assertEqual(list(get_column(string)), ["foo", "baz"])

        self.assertIsNone(get_column(None, False))
        np.testing.assert_almost_equal(get_column(cont, False), [1, 2, 6])
        self.assertEqual(list(get_column(string, False)), ["foo", "bar", "baz"])

        self.assertIsNone(get_column(None, return_labels=True))
        self.assertEqual(get_column(disc, return_labels=True), disc.values)
        self.assertEqual(get_column(disc2, return_labels=True), disc2.values)
        self.assertEqual(get_column(disc3, return_labels=True), disc3.values)
        with self.assertRaises(AssertionError):
            get_column(cont, return_labels=True)
        with self.assertRaises(AssertionError):
            get_column(cont, return_labels=True, max_categories=4)
        with self.assertRaises(AssertionError):
            get_column(string, return_labels=True)
        with self.assertRaises(AssertionError):
            get_column(string, return_labels=True, max_categories=4)

    def test_get_column_merge_infrequent(self):
        widget = self.widget
        get_column = widget.get_column

        disc = DiscreteVariable("disc", list("abcdefghijklmno"))
        disc2 = DiscreteVariable("disc2", list("abc"))
        domain = Domain([disc], disc2)

        x = np.array(
            [1, 1, 1, 5, 4, 1, 1, 5, 8, 5, 5, 0, 0, 0, 4, 5, 10], dtype=float)
        y = np.ones(len(x))
        widget.data = Table.from_numpy(domain, np.atleast_2d(x).T, y)

        np.testing.assert_almost_equal(get_column(disc), x)
        self.assertEqual(get_column(disc, return_labels=True), disc.values)
        np.testing.assert_almost_equal(get_column(disc2), y)
        self.assertEqual(get_column(disc2, return_labels=True), disc2.values)

        np.testing.assert_almost_equal(
            get_column(disc, max_categories=4),
            [1, 1, 1, 2, 3, 1, 1, 2, 3, 2, 2, 0, 0, 0, 3, 2, 3])
        self.assertEqual(
            get_column(disc, max_categories=4, return_labels=True),
            [disc.values[0], disc.values[1], disc.values[5], "Other"])
        np.testing.assert_almost_equal(
            get_column(disc2, max_categories=4), y)
        self.assertEqual(
            get_column(disc2, return_labels=True, max_categories=4),
            disc2.values)

        # Test that get_columns modify a copy of the data and not the data
        np.testing.assert_almost_equal(get_column(disc), x)
        self.assertEqual(get_column(disc, return_labels=True), disc.values)

    def test_get_tooltip(self):
        widget = self.widget
        domain = Domain([ContinuousVariable("v")])
        widget.data = Table.from_numpy(domain, [[1], [2], [3]])
        widget.valid_data = np.array([True, False, True])
        self.assertTrue("3" in widget.get_tooltip([1]))
        self.assertTrue("1" in widget.get_tooltip([0, 1])
                        and "3" in widget.get_tooltip([0, 1]))
        self.assertEqual(widget.get_tooltip([]), "")


class TestableDataProjectionWidget(OWDataProjectionWidget):
    def get_embedding(self):
        self.valid_data = None
        if self.data is None:
            return None

        x_data = self.data.X.toarray() if self.data.is_sparse() \
            else self.data.X
        self.valid_data = np.any(np.isfinite(x_data), 1)
        if not len(x_data[self.valid_data]):
            return None

        x_data[x_data == np.inf] = np.nan
        x_data = np.nanmean(x_data[self.valid_data], 1)
        y_data = np.ones(len(x_data))
        return np.vstack((x_data, y_data)).T


class TestOWDataProjectionWidget(WidgetTest, ProjectionWidgetTestMixin,
                                 WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(TestableDataProjectionWidget)

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.graph.select_by_indices(list(range(0, len(self.data), 10)))
        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(TestableDataProjectionWidget,
                               stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.data, widget=w)
        self.assertEqual(np.sum(w.graph.selection), 15)
        np.testing.assert_equal(self.widget.graph.selection, w.graph.selection)

    def test_too_many_labels(self):
        w = self.widget.Warning.too_many_labels
        self.assertFalse(w.is_shown())

        self.widget.graph.too_many_labels.emit(True)
        self.assertTrue(w.is_shown())

        self.widget.graph.too_many_labels.emit(False)
        self.assertFalse(w.is_shown())

    def test_invalid_subset(self):
        widget = self.widget

        data = Table("iris")
        self.send_signal(widget.Inputs.data_subset, data[40:60])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertFalse(widget.Warning.subset_not_subset.is_shown())

        self.send_signal(widget.Inputs.data, data[30:70])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertFalse(widget.Warning.subset_not_subset.is_shown())

        self.send_signal(widget.Inputs.data, data[30:50])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertTrue(widget.Warning.subset_not_subset.is_shown())

        self.send_signal(widget.Inputs.data, data[20:30])
        self.assertTrue(widget.Warning.subset_independent.is_shown())
        self.assertFalse(widget.Warning.subset_not_subset.is_shown())

        self.send_signal(widget.Inputs.data, data[30:70])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertFalse(widget.Warning.subset_not_subset.is_shown())

        self.send_signal(widget.Inputs.data, data[30:50])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertTrue(widget.Warning.subset_not_subset.is_shown())

        self.send_signals([(widget.Inputs.data, Table("titanic")),
                           (widget.Inputs.data_subset, None)])
        self.assertFalse(widget.Warning.subset_independent.is_shown())
        self.assertFalse(widget.Warning.subset_not_subset.is_shown())

    def test_get_coordinates_data(self):
        self.widget.get_embedding = Mock(return_value=np.ones((10, 2)))
        self.widget.valid_data = np.ones((10,), dtype=bool)
        self.widget.valid_data[0] = False
        self.assertIsNotNone(self.widget.get_coordinates_data())
        self.assertEqual(len(self.widget.get_coordinates_data()[0]), 9)
        self.widget.valid_data = np.zeros((10,), dtype=bool)
        self.assertIsNone(self.widget.get_coordinates_data()[0])

    def test_sparse_data_reload(self):
        table = Table("heart_disease").to_sparse()
        self.widget.setup_plot = Mock()
        self.send_signal(self.widget.Inputs.data, table)
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.setup_plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
