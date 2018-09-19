# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import patch

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
            get_column(cont, return_labels=True, merge_infrequent=True)
            get_column(cont, merge_infrequent=True)
        with self.assertRaises(AssertionError):
            get_column(string, return_labels=True)
            get_column(string, return_labels=True, merge_infrequent=True)
            get_column(string, merge_infrequent=True)

    @patch("Orange.widgets.visualize.utils.widget.MAX_CATEGORIES", 4)
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
            get_column(disc, merge_infrequent=True),
            [1, 1, 1, 2, 3, 1, 1, 2, 3, 2, 2, 0, 0, 0, 3, 2, 3])
        self.assertEqual(
            get_column(disc, merge_infrequent=True, return_labels=True),
            [disc.values[0], disc.values[1], disc.values[5], "Other"])
        np.testing.assert_almost_equal(
            get_column(disc2, merge_infrequent=True), y)
        self.assertEqual(
            get_column(disc2, return_labels=True, merge_infrequent=True),
            disc2.values)

        # Test that get_columns modify a copy of the data and not the data
        np.testing.assert_almost_equal(get_column(disc), x)
        self.assertEqual(get_column(disc, return_labels=True), disc.values)


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
