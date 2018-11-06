import unittest

from AnyQt.QtWidgets import QWidget

from Orange.canvas.gui import utils
from ..test import QAppTestCase


class TestUpdatesDisabled(QAppTestCase):
    def test_context_manager(self):
        widget = QWidget()
        self.assertTrue(widget.updatesEnabled())
        with utils.updates_disabled(widget):
            self.assertFalse(widget.updatesEnabled())
        self.assertTrue(widget.updatesEnabled())

        # Disabling twice still does not reenable updates when exiting the
        # inner context, but disables afterwards
        with utils.updates_disabled(widget):
            self.assertFalse(widget.updatesEnabled())
            with utils.updates_disabled(widget):
                self.assertFalse(widget.updatesEnabled())
            self.assertFalse(widget.updatesEnabled())
        self.assertTrue(widget.updatesEnabled())

        # Updates are reenabled even when exception occurs within the context
        with self.assertRaises(ValueError):
            with utils.updates_disabled(widget):
                self.assertFalse(widget.updatesEnabled())
                raise ValueError("foo")
        self.assertTrue(widget.updatesEnabled())

    def test_decorator(self):
        test_self = self

        class OWSomeWidget(QWidget):
            def __init__(self):
                super().__init__()
                self.child = QWidget()

            @utils.updates_disabled('child')
            def method(self, x=1):
                test_self.assertFalse(self.child.updatesEnabled())
                x = 1 / x
                with utils.updates_disabled(self.child):
                    test_self.assertFalse(self.child.updatesEnabled())
                test_self.assertFalse(self.child.updatesEnabled())

        widget = OWSomeWidget()
        self.assertTrue(widget.child.updatesEnabled())
        widget.method()
        self.assertTrue(widget.child.updatesEnabled())
        self.assertRaises(ZeroDivisionError, widget.method, 0)
        self.assertTrue(widget.child.updatesEnabled())


if __name__ == "__main__":
    unittest.main()
