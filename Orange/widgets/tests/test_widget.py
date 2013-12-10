from unittest import TestCase
from PyQt4.QtGui import QApplication
from Orange.widgets.widget import OWWidget


class Dummy:
    b = None


class MyWidget(OWWidget):
    def __init__(self, depth=1):
        super().__init__()

        self.field = 42
        self.non_widget = Dummy()
        if depth:
            self.another_widget = MyWidget(depth=depth-1)
        else:
            self.another_widget = None


class WidgetTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qApp = QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.qApp.quit()

    def test_setattr(self):
        widget = MyWidget()

        setattr(widget, 'field', 1)
        self.assertEqual(widget.field, 1)

        setattr(widget, 'non_widget.b', 2)
        self.assertEqual(widget.non_widget.b, 2)

        setattr(widget, 'another_widget.field', 3)
        self.assertEqual(widget.another_widget.field, 3)

        setattr(widget, 'unknown_field', 4)
        self.assertEqual(widget.unknown_field, 4)

        with self.assertRaises(AttributeError):
            setattr(widget, 'another_widget.another_widget.field', 5)

        with self.assertRaises(AttributeError):
            setattr(widget, 'unknown_field2.field', 6)
