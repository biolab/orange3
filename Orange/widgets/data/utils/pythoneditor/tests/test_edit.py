"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import unittest

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QKeySequence
from AnyQt.QtTest import QTest

from Orange.widgets.data.utils.pythoneditor.tests import base
from Orange.widgets.data.utils.pythoneditor.tests.base import SimpleWidget
from Orange.widgets.tests.base import WidgetTest


class Test(WidgetTest):
    """Base class for tests
    """
    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart

    def tearDown(self):
        self.qpart.terminate()

    def test_overwrite_edit(self):
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClicks(self.qpart, "stu")
        self.assertEqual(self.qpart.text, 'stuabcd')
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClicks(self.qpart, "xy")
        self.assertEqual(self.qpart.text, 'stuxycd')
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClicks(self.qpart, "z")
        self.assertEqual(self.qpart.text, 'stuxyzcd')

    def test_overwrite_backspace(self):
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        for _ in range(3):
            QTest.keyClick(self.qpart, Qt.Key_Right)
        for _ in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Backspace)
        self.assertEqual(self.qpart.text, 'a  d')

    @base.in_main_loop
    def test_overwrite_undo(self):
        self.qpart.show()
        self.qpart.text = 'abcd'
        QTest.keyClick(self.qpart, Qt.Key_Insert)
        QTest.keyClick(self.qpart, Qt.Key_Right)
        QTest.keyClick(self.qpart, Qt.Key_X)
        QTest.keyClick(self.qpart, Qt.Key_X)
        self.assertEqual(self.qpart.text, 'axxd')
        # Ctrl+Z doesn't work. Wtf???
        self.qpart.document().undo()
        self.qpart.document().undo()
        self.assertEqual(self.qpart.text, 'abcd')

    def test_home1(self):
        """ Test the operation of the home key. """

        self.qpart.show()
        self.qpart.text = '  xx'
        # Move to the end of this string.
        self.qpart.cursorPosition = (100, 100)
        # Press home the first time. This should move to the beginning of the
        # indent: line 0, column 4.
        self.assertEqual(self.qpart.cursorPosition, (0, 4))

    def column(self):
        """ Return the column at which the cursor is located."""
        return self.qpart.cursorPosition[1]

    def test_home2(self):
        """ Test the operation of the home key. """

        self.qpart.show()
        self.qpart.text = '\n\n    ' + 'x'*10000
        # Move to the end of this string.
        self.qpart.cursorPosition = (100, 100)
        # Press home. We should either move to the line beginning or indent. Use
        # a QKeySequence because there's no home key on some Macs, so use
        # whatever means home on that platform.
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        # There's no way I can find of determine what the line beginning should
        # be. So, just press home again if we're not at the indent.
        if self.column() != 4:
            # Press home again to move to the beginning of the indent.
            base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        # We're at the indent.
        self.assertEqual(self.column(), 4)

        # Move to the beginning of the line.
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        self.assertEqual(self.column(), 0)

        # Move back to the beginning of the indent.
        base.keySequenceClicks(self.qpart, QKeySequence.MoveToStartOfLine)
        self.assertEqual(self.column(), 4)


if __name__ == '__main__':
    unittest.main()
