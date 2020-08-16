"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import os
import unittest

from AnyQt.QtTest import QTest
from AnyQt.QtCore import Qt, QTimer

from Orange.widgets.data.utils.pythoneditor.tests.base import SimpleWidget
from Orange.widgets.tests.base import WidgetTest


class _BaseTest(WidgetTest):
    """Base class for tests
    """

    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart

    def tearDown(self):
        self.qpart.terminate()


def _rm():
    try:
        os.remove('print.pdf')
    except (OSError, ValueError):
        pass


def _exists():
    return os.path.isfile('print.pdf')


class Print(_BaseTest):
    @unittest.skip("Does not work")
    def test_1(self):
        self._rm()
        self.assertFalse(self._exists())
        self.qpart.show()
        def acceptDialog():
            QTest.keyClick(self.app.focusWidget(), Qt.Key_Enter, Qt.NoModifier)
        QTimer.singleShot(1000, acceptDialog)
        QTest.keyClick(self.qpart, Qt.Key_P, Qt.ControlModifier)

        self.assertTrue(self._exists())
        self._rm()


if __name__ == '__main__':
    unittest.main()
