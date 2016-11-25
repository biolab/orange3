# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from AnyQt.QtCore import QEvent, QPoint, Qt
from AnyQt.QtGui import QMouseEvent

from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owsieve import OWSieveDiagram


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
