# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from math import isnan
import numpy as np

from AnyQt.QtCore import QEvent, QPoint, Qt
from AnyQt.QtGui import QMouseEvent

from Orange.data import DiscreteVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owsieve import OWSieveDiagram
from Orange.widgets.visualize.owsieve import ChiSqStats


class TestOWSieveDiagram(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWSieveDiagram)

    def _select_data(self):
        self.widget.attr_x, self.widget.attr_y = self.data.domain[:2]
        area = self.widget.areas[0]
        self.widget.select_area(area, QMouseEvent(
            QEvent.MouseButtonPress, QPoint(), Qt.LeftButton,
            Qt.LeftButton, Qt.KeyboardModifiers()))
        return [0, 4, 6, 7, 11, 17, 19, 21, 22, 24, 26, 39, 40, 43, 44, 46]

    def test_missing_values(self):
        """Check widget for dataset with missing values"""
        attrs = [DiscreteVariable("c1", ["a", "b", "c"])]
        class_var = DiscreteVariable("cls", [])
        X = np.array([1, 2, 0, 1, 0, 2])[:, None]
        data = Table(Domain(attrs, class_var), X, np.array([np.nan] * 6))
        self.send_signal("Data", data)

    def test_keyerror(self):
        """gh-2007
        Check if it works when a table has only one row or duplicates.
        Discretizer must have remove_const set to False.
        """
        data = Table("iris")
        data = data[0:1]
        self.send_signal("Data", data)

    def test_chisquare(self):
        """
         gh-2031
         Check if it can calculate chi square when there are no attributes which suppose to be.
        """
        a = DiscreteVariable("a", values=["y", "n"])
        b = DiscreteVariable("b", values=["y", "n", "o"])
        table = Table(Domain([a, b]), list(zip("yynny", "ynyyn")))
        chi = ChiSqStats(table, 0, 1)
        self.assertFalse(isnan(chi.chisq))
