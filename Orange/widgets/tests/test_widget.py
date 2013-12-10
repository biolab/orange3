from unittest import TestCase
from PyQt4.QtGui import QApplication
from Orange.widgets.gui import CONTROLLED_ATTRIBUTES, ATTRIBUTE_CONTROLLERS
from Orange.widgets.widget import OWWidget


class Dummy:
    b = None


class MyWidget(OWWidget):
    def __init__(self, depth=1):
        super().__init__()

        self.field = 42
        self.non_widget = Dummy()
        if depth:
            self.widget = MyWidget(depth=depth-1)
        else:
            self.widget = None


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

        setattr(widget, 'widget.field', 3)
        self.assertEqual(widget.widget.field, 3)

        setattr(widget, 'unknown_field', 4)
        self.assertEqual(widget.unknown_field, 4)

        with self.assertRaises(AttributeError):
            setattr(widget, 'widget.widget.field', 5)

        with self.assertRaises(AttributeError):
            setattr(widget, 'unknown_field2.field', 6)

    def test_notify_controller_on_attribute_change(self):
        widget = MyWidget()
        delattr(widget.widget, CONTROLLED_ATTRIBUTES)
        calls = []

        def callback(*args, **kwargs):
            calls.append((args, kwargs))

        getattr(widget, CONTROLLED_ATTRIBUTES)['field'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['non_widget.b'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['widget.field'] = callback

        widget.field = 5
        #widget.non_widget.b = 5
        widget.widget.field = 5

        setattr(widget, 'field', 7)
        setattr(widget, 'non_widget.b', 7)
        setattr(widget, 'widget.field', 7)

        self.assertEqual(len(calls), 6)



