"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import unittest

from Orange.widgets.data.utils.pythoneditor.tests.base import SimpleWidget
from Orange.widgets.tests.base import WidgetTest

# pylint: disable=protected-access

class _BaseTest(WidgetTest):
    """Base class for tests
    """

    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart

    def tearDown(self):
        self.qpart.terminate()


class Selection(_BaseTest):

    def test_resetSelection(self):
        # Reset selection
        self.qpart.text = 'asdf fdsa'
        self.qpart.absSelectedPosition = 1, 3
        self.assertTrue(self.qpart.textCursor().hasSelection())
        self.qpart.resetSelection()
        self.assertFalse(self.qpart.textCursor().hasSelection())

    def test_setSelection(self):
        self.qpart.text = 'asdf fdsa'

        self.qpart.selectedPosition = ((0, 3), (0, 7))

        self.assertEqual(self.qpart.selectedText, "f fd")
        self.assertEqual(self.qpart.selectedPosition, ((0, 3), (0, 7)))

    def test_selected_multiline_text(self):
        self.qpart.text = "a\nb"
        self.qpart.selectedPosition = ((0, 0), (1, 1))
        self.assertEqual(self.qpart.selectedText, "a\nb")


class ReplaceText(_BaseTest):
    def test_replaceText1(self):
        # Basic case
        self.qpart.text = '123456789'
        self.qpart.replaceText(3, 4, 'xyz')
        self.assertEqual(self.qpart.text, '123xyz89')

    def test_replaceText2(self):
        # Replace uses (line, col) position
        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText((1, 4), 3, 'Z')
        self.assertEqual(self.qpart.text, '12345\n6789Zbcde')

    def test_replaceText3(self):
        # Edge cases
        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText((0, 0), 3, 'Z')
        self.assertEqual(self.qpart.text, 'Z45\n67890\nabcde')

        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText((2, 4), 1, 'Z')
        self.assertEqual(self.qpart.text, '12345\n67890\nabcdZ')

        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText((0, 0), 0, 'Z')
        self.assertEqual(self.qpart.text, 'Z12345\n67890\nabcde')

        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText((2, 5), 0, 'Z')
        self.assertEqual(self.qpart.text, '12345\n67890\nabcdeZ')

    def test_replaceText4(self):
        # Replace nothing with something
        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.replaceText(2, 0, 'XYZ')
        self.assertEqual(self.qpart.text, '12XYZ345\n67890\nabcde')

    def test_replaceText5(self):
        # Make sure exceptions are raised for invalid params
        self.qpart.text = '12345\n67890\nabcde'
        self.assertRaises(IndexError, self.qpart.replaceText, -1, 1, 'Z')
        self.assertRaises(IndexError, self.qpart.replaceText, len(self.qpart.text) + 1, 0, 'Z')
        self.assertRaises(IndexError, self.qpart.replaceText, len(self.qpart.text), 1, 'Z')
        self.assertRaises(IndexError, self.qpart.replaceText, (0, 7), 1, 'Z')
        self.assertRaises(IndexError, self.qpart.replaceText, (7, 0), 1, 'Z')


class InsertText(_BaseTest):
    def test_1(self):
        # Basic case
        self.qpart.text = '123456789'
        self.qpart.insertText(3, 'xyz')
        self.assertEqual(self.qpart.text, '123xyz456789')

    def test_2(self):
        # (line, col) position
        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.insertText((1, 4), 'Z')
        self.assertEqual(self.qpart.text, '12345\n6789Z0\nabcde')

    def test_3(self):
        # Edge cases
        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.insertText((0, 0), 'Z')
        self.assertEqual(self.qpart.text, 'Z12345\n67890\nabcde')

        self.qpart.text = '12345\n67890\nabcde'
        self.qpart.insertText((2, 5), 'Z')
        self.assertEqual(self.qpart.text, '12345\n67890\nabcdeZ')


class IsCodeOrComment(_BaseTest):
    def test_1(self):
        # Basic case
        self.qpart.text = 'a + b # comment'
        self.assertEqual([self.qpart.isCode(0, i) for i in range(len(self.qpart.text))],
                         [True, True, True, True, True, True, False, False, False, False,
                          False, False, False, False, False])
        self.assertEqual([self.qpart.isComment(0, i) for i in range(len(self.qpart.text))],
                         [False, False, False, False, False, False, True, True, True, True,
                          True, True, True, True, True])

    def test_2(self):
        self.qpart.text = '#'

        self.assertFalse(self.qpart.isCode(0, 0))
        self.assertTrue(self.qpart.isComment(0, 0))


class ToggleCommentTest(_BaseTest):
    def test_single_line(self):
        self.qpart.text = 'a = 2'
        self.qpart._onToggleCommentLine()
        self.assertEqual('# a = 2\n', self.qpart.text)
        self.qpart._onToggleCommentLine()
        self.assertEqual('# a = 2\n', self.qpart.text)
        self.qpart._selectLines(0, 0)
        self.qpart._onToggleCommentLine()
        self.assertEqual('a = 2\n', self.qpart.text)

    def test_two_lines(self):
        self.qpart.text = 'a = 2\nb = 3'
        self.qpart._selectLines(0, 1)
        self.qpart._onToggleCommentLine()
        self.assertEqual('# a = 2\n# b = 3\n', self.qpart.text)
        self.qpart.undo()
        self.assertEqual('a = 2\nb = 3', self.qpart.text)


class Signals(_BaseTest):
    def test_indent_width_changed(self):
        newValue = [None]

        def setNeVal(val):
            newValue[0] = val

        self.qpart.indentWidthChanged.connect(setNeVal)

        self.qpart.indentWidth = 7
        self.assertEqual(newValue[0], 7)

    def test_use_tabs_changed(self):
        newValue = [None]

        def setNeVal(val):
            newValue[0] = val

        self.qpart.indentUseTabsChanged.connect(setNeVal)

        self.qpart.indentUseTabs = True
        self.assertEqual(newValue[0], True)

    def test_eol_changed(self):
        newValue = [None]

        def setNeVal(val):
            newValue[0] = val

        self.qpart.eolChanged.connect(setNeVal)

        self.qpart.eol = '\r\n'
        self.assertEqual(newValue[0], '\r\n')


class Lines(_BaseTest):
    def setUp(self):
        super().setUp()
        self.qpart.text = 'abcd\nefgh\nklmn\nopqr'

    def test_accessByIndex(self):
        self.assertEqual(self.qpart.lines[0], 'abcd')
        self.assertEqual(self.qpart.lines[1], 'efgh')
        self.assertEqual(self.qpart.lines[-1], 'opqr')

    def test_modifyByIndex(self):
        self.qpart.lines[2] = 'new text'
        self.assertEqual(self.qpart.text, 'abcd\nefgh\nnew text\nopqr')

    def test_getSlice(self):
        self.assertEqual(self.qpart.lines[0], 'abcd')
        self.assertEqual(self.qpart.lines[1], 'efgh')
        self.assertEqual(self.qpart.lines[3], 'opqr')
        self.assertEqual(self.qpart.lines[-4], 'abcd')
        self.assertEqual(self.qpart.lines[1:4], ['efgh', 'klmn', 'opqr'])
        self.assertEqual(self.qpart.lines[1:7],
                         ['efgh', 'klmn', 'opqr'])  # Python list behaves this way
        self.assertEqual(self.qpart.lines[0:0], [])
        self.assertEqual(self.qpart.lines[0:1], ['abcd'])
        self.assertEqual(self.qpart.lines[:2], ['abcd', 'efgh'])
        self.assertEqual(self.qpart.lines[0:-2], ['abcd', 'efgh'])
        self.assertEqual(self.qpart.lines[-2:], ['klmn', 'opqr'])
        self.assertEqual(self.qpart.lines[-4:-2], ['abcd', 'efgh'])

        with self.assertRaises(IndexError):
            self.qpart.lines[4]  # pylint: disable=pointless-statement
        with self.assertRaises(IndexError):
            self.qpart.lines[-5]  # pylint: disable=pointless-statement

    def test_setSlice_1(self):
        self.qpart.lines[0] = 'xyz'
        self.assertEqual(self.qpart.text, 'xyz\nefgh\nklmn\nopqr')

    def test_setSlice_2(self):
        self.qpart.lines[1] = 'xyz'
        self.assertEqual(self.qpart.text, 'abcd\nxyz\nklmn\nopqr')

    def test_setSlice_3(self):
        self.qpart.lines[-4] = 'xyz'
        self.assertEqual(self.qpart.text, 'xyz\nefgh\nklmn\nopqr')

    def test_setSlice_4(self):
        self.qpart.lines[0:4] = ['st', 'uv', 'wx', 'z']
        self.assertEqual(self.qpart.text, 'st\nuv\nwx\nz')

    def test_setSlice_5(self):
        self.qpart.lines[0:47] = ['st', 'uv', 'wx', 'z']
        self.assertEqual(self.qpart.text, 'st\nuv\nwx\nz')

    def test_setSlice_6(self):
        self.qpart.lines[1:3] = ['st', 'uv']
        self.assertEqual(self.qpart.text, 'abcd\nst\nuv\nopqr')

    def test_setSlice_61(self):
        with self.assertRaises(ValueError):
            self.qpart.lines[1:3] = ['st', 'uv', 'wx', 'z']

    def test_setSlice_7(self):
        self.qpart.lines[-3:3] = ['st', 'uv']
        self.assertEqual(self.qpart.text, 'abcd\nst\nuv\nopqr')

    def test_setSlice_8(self):
        self.qpart.lines[-3:-1] = ['st', 'uv']
        self.assertEqual(self.qpart.text, 'abcd\nst\nuv\nopqr')

    def test_setSlice_9(self):
        with self.assertRaises(IndexError):
            self.qpart.lines[4] = 'st'
        with self.assertRaises(IndexError):
            self.qpart.lines[-5] = 'st'


class LinesWin(Lines):
    def setUp(self):
        super().setUp()
        self.qpart.eol = '\r\n'


if __name__ == '__main__':
    unittest.main()
