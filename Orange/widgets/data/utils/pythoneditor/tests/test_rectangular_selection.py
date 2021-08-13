"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import unittest

# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=unused-variable

from AnyQt.QtCore import Qt
from AnyQt.QtTest import QTest
from AnyQt.QtGui import QKeySequence

from Orange.widgets.data.utils.pythoneditor.tests import base
from Orange.widgets.data.utils.pythoneditor.tests.base import SimpleWidget
from Orange.widgets.tests.base import WidgetTest


class _Test(WidgetTest):
    """Base class for tests
    """

    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart

    def tearDown(self):
        self.qpart.hide()
        self.qpart.terminate()

    def test_real_to_visible(self):
        self.qpart.text = 'abcdfg'
        self.assertEqual(0, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 0))
        self.assertEqual(2, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 2))
        self.assertEqual(6, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 6))

        self.qpart.text = '\tab\tcde\t'
        self.assertEqual(0, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 0))
        self.assertEqual(4, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 1))
        self.assertEqual(5, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 2))
        self.assertEqual(8, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 4))
        self.assertEqual(12, self.qpart._rectangularSelection._realToVisibleColumn(self.qpart.text, 8))

    def test_visible_to_real(self):
        self.qpart.text = 'abcdfg'
        self.assertEqual(0, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 0))
        self.assertEqual(2, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 2))
        self.assertEqual(6, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 6))

        self.qpart.text = '\tab\tcde\t'
        self.assertEqual(0, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 0))
        self.assertEqual(1, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 4))
        self.assertEqual(2, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 5))
        self.assertEqual(4, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 8))
        self.assertEqual(8, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 12))

        self.assertEqual(None, self.qpart._rectangularSelection._visibleToRealColumn(self.qpart.text, 13))

    def test_basic(self):
        self.qpart.show()
        for key in [Qt.Key_Delete, Qt.Key_Backspace]:
            self.qpart.text = 'abcd\nef\nghkl\nmnop'
            QTest.keyClick(self.qpart, Qt.Key_Right)
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
            QTest.keyClick(self.qpart, key)
            self.assertEqual(self.qpart.text, 'ad\ne\ngl\nmnop')

    def test_reset_by_move(self):
        self.qpart.show()
        self.qpart.text = 'abcd\nef\nghkl\nmnop'
        QTest.keyClick(self.qpart, Qt.Key_Right)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Left)
        QTest.keyClick(self.qpart, Qt.Key_Backspace)
        self.assertEqual(self.qpart.text, 'abcd\nef\ngkl\nmnop')

    def test_reset_by_edit(self):
        self.qpart.show()
        self.qpart.text = 'abcd\nef\nghkl\nmnop'
        QTest.keyClick(self.qpart, Qt.Key_Right)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClicks(self.qpart, 'x')
        QTest.keyClick(self.qpart, Qt.Key_Backspace)
        self.assertEqual(self.qpart.text, 'abcd\nef\nghkl\nmnop')

    def test_with_tabs(self):
        self.qpart.show()
        self.qpart.text = 'abcdefghhhhh\n\tklm\n\t\txyz'
        self.qpart.cursorPosition = (0, 6)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Delete)

        # 3 variants, Qt behavior differs on different systems
        self.assertIn(self.qpart.text, ('abcdefhh\n\tkl\n\t\tz',
                                        'abcdefh\n\tkl\n\t\t',
                                        'abcdefhhh\n\tkl\n\t\tyz'))

    def test_delete(self):
        self.qpart.show()
        self.qpart.text = 'this is long\nshort\nthis is long'
        self.qpart.cursorPosition = (0, 8)
        for i in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        for i in range(4):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)

        QTest.keyClick(self.qpart, Qt.Key_Delete)
        self.assertEqual(self.qpart.text, 'this is \nshort\nthis is ')

    def test_copy_paste(self):
        self.qpart.indentUseTabs = True
        self.qpart.show()
        self.qpart.text = 'xx 123 yy\n' + \
                          'xx 456 yy\n' + \
                          'xx 789 yy\n' + \
                          '\n' + \
                          'asdfghijlmn\n' + \
                          'x\t\n' + \
                          '\n' + \
                          '\t\t\n' + \
                          'end\n'
        self.qpart.cursorPosition = 0, 3
        for i in range(3):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        for i in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)

        QTest.keyClick(self.qpart, Qt.Key_C, Qt.ControlModifier)

        self.qpart.cursorPosition = 4, 10
        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)

        self.assertEqual(self.qpart.text,
                         'xx 123 yy\nxx 456 yy\nxx 789 yy\n\nasdfghijlm123n\nx\t      456\n\t\t  789\n\t\t\nend\n')

    def test_copy_paste_utf8(self):
        self.qpart.show()
        self.qpart.text = 'фыва'
        for i in range(3):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_C, Qt.ControlModifier)

        QTest.keyClick(self.qpart, Qt.Key_Right)
        QTest.keyClick(self.qpart, Qt.Key_Space)
        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)

        self.assertEqual(self.qpart.text,
                         'фыва фыв')

    def test_paste_replace_selection(self):
        self.qpart.show()
        self.qpart.text = 'asdf'

        for i in range(4):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_C, Qt.ControlModifier)

        QTest.keyClick(self.qpart, Qt.Key_End)
        QTest.keyClick(self.qpart, Qt.Key_Left, Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)

        self.assertEqual(self.qpart.text,
                         'asdasdf')

    def test_paste_replace_rectangular_selection(self):
        self.qpart.show()
        self.qpart.text = 'asdf'

        for i in range(4):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_C, Qt.ControlModifier)

        QTest.keyClick(self.qpart, Qt.Key_Left)
        QTest.keyClick(self.qpart, Qt.Key_Left, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)

        self.assertEqual(self.qpart.text,
                         'asasdff')

    def test_paste_new_lines(self):
        self.qpart.show()
        self.qpart.text = 'a\nb\nc\nd'

        for i in range(4):
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_C, Qt.ControlModifier)

        self.qpart.text = 'x\ny'
        self.qpart.cursorPosition = (1, 1)

        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)

        self.assertEqual(self.qpart.text,
                         'x\nya\n b\n c\n d')

    def test_cut(self):
        self.qpart.show()
        self.qpart.text = 'asdf'

        for i in range(4):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        QTest.keyClick(self.qpart, Qt.Key_X, Qt.ControlModifier)
        self.assertEqual(self.qpart.text, '')

        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)
        self.assertEqual(self.qpart.text, 'asdf')

    def test_cut_paste(self):
        # Cursor must be moved to top-left after cut, and original text is restored after paste

        self.qpart.show()
        self.qpart.text = 'abcd\nefgh\nklmn'

        QTest.keyClick(self.qpart, Qt.Key_Right)
        for i in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Right, Qt.AltModifier | Qt.ShiftModifier)
        for i in range(2):
            QTest.keyClick(self.qpart, Qt.Key_Down, Qt.AltModifier | Qt.ShiftModifier)

        QTest.keyClick(self.qpart, Qt.Key_X, Qt.ControlModifier)
        self.assertEqual(self.qpart.cursorPosition, (0, 1))

        QTest.keyClick(self.qpart, Qt.Key_V, Qt.ControlModifier)
        self.assertEqual(self.qpart.text, 'abcd\nefgh\nklmn')

    def test_warning(self):
        self.qpart.show()
        self.qpart.text = 'a\n' * 3000
        warning = [None]
        def _saveWarning(text):
            warning[0] = text
        self.qpart.userWarning.connect(_saveWarning)

        base.keySequenceClicks(self.qpart, QKeySequence.SelectEndOfDocument, Qt.AltModifier)

        self.assertEqual(warning[0], 'Rectangular selection area is too big')


if __name__ == '__main__':
    unittest.main()
