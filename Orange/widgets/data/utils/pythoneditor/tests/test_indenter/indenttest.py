"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code

from AnyQt.QtCore import Qt
from AnyQt.QtTest import QTest

from Orange.widgets.data.utils.pythoneditor.tests.base import EditorTest

# pylint: disable=protected-access


class IndentTest(EditorTest):
    """Base class for tests
    """

    def setUp(self):
        super().setUp()
        if hasattr(self, 'INDENT_WIDTH'):
            self.qpart.indentWidth = self.INDENT_WIDTH

    def setOrigin(self, text):
        self.qpart.text = '\n'.join(text)

    def verifyExpected(self, text):
        lines = self.qpart.text.split('\n')
        self.assertEqual(text, lines)

    def setCursorPosition(self, line, col):
        self.qpart.cursorPosition = line, col

    def enter(self):
        QTest.keyClick(self.qpart, Qt.Key_Enter)

    def tab(self):
        QTest.keyClick(self.qpart, Qt.Key_Tab)

    def type(self, text):
        QTest.keyClicks(self.qpart, text)

    def writeCursorPosition(self):
        line, col = self.qpart.cursorPosition
        text = '(%d,%d)' % (line, col)
        self.type(text)

    def writeln(self):
        self.qpart.textCursor().insertText('\n')

    def alignLine(self, index):
        self.qpart._indenter.autoIndentBlock(self.qpart.document().findBlockByNumber(index), '')

    def alignAll(self):
        QTest.keyClick(self.qpart, Qt.Key_A, Qt.ControlModifier)
        self.qpart.autoIndentLineAction.trigger()
