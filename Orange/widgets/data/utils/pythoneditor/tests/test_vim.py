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
from AnyQt.QtTest import QTest

from Orange.widgets.data.utils.pythoneditor.tests.base import SimpleWidget
from Orange.widgets.data.utils.pythoneditor.vim import _globalClipboard
from Orange.widgets.tests.base import WidgetTest

# pylint: disable=too-many-lines


class _Test(WidgetTest):
    """Base class for tests
    """

    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart
        self.qpart.lines = ['The quick brown fox',
                            'jumps over the',
                            'lazy dog',
                            'back']
        self.qpart.vimModeIndicationChanged.connect(self._onVimModeChanged)

        self.qpart.vimModeEnabled = True
        self.vimMode = 'normal'

    def tearDown(self):
        self.qpart.hide()
        self.qpart.terminate()

    def _onVimModeChanged(self, _, mode):
        self.vimMode = mode

    def click(self, keys):
        if isinstance(keys, str):
            for key in keys:
                if key.isupper() or key in '$%^<>':
                    QTest.keyClick(self.qpart, key, Qt.ShiftModifier)
                else:
                    QTest.keyClicks(self.qpart, key)
        else:
            QTest.keyClick(self.qpart, keys)


class Modes(_Test):
    def test_01(self):
        """Switch modes insert/normal
        """
        self.assertEqual(self.vimMode, 'normal')
        self.click("i123")
        self.assertEqual(self.vimMode, 'insert')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')
        self.click("i4")
        self.assertEqual(self.vimMode, 'insert')
        self.assertEqual(self.qpart.lines[0],
                         '1234The quick brown fox')

    def test_02(self):
        """Append with A
        """
        self.qpart.cursorPosition = (2, 0)
        self.click("A")
        self.assertEqual(self.vimMode, 'insert')
        self.click("XY")

        self.assertEqual(self.qpart.lines[2],
                         'lazy dogXY')

    def test_03(self):
        """Append with a
        """
        self.qpart.cursorPosition = (2, 0)
        self.click("a")
        self.assertEqual(self.vimMode, 'insert')
        self.click("XY")

        self.assertEqual(self.qpart.lines[2],
                         'lXYazy dog')

    def test_04(self):
        """Mode line shows composite command start
        """
        self.assertEqual(self.vimMode, 'normal')
        self.click('d')
        self.assertEqual(self.vimMode, 'd')
        self.click('w')
        self.assertEqual(self.vimMode, 'normal')

    def test_05(self):
        """ Replace mode
        """
        self.assertEqual(self.vimMode, 'normal')
        self.click('R')
        self.assertEqual(self.vimMode, 'replace')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0],
                         'asdfquick brown fox')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')

        self.click('R')
        self.assertEqual(self.vimMode, 'replace')
        self.click(Qt.Key_Insert)
        self.assertEqual(self.vimMode, 'insert')

    def test_05a(self):
        """ Replace mode - at end of line
        """
        self.click('$')
        self.click('R')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0],
                         'The quick brown foxasdf')

    def test_06(self):
        """ Visual mode
        """
        self.assertEqual(self.vimMode, 'normal')

        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click(Qt.Key_Escape)
        self.assertEqual(self.vimMode, 'normal')

        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click('i')
        self.assertEqual(self.vimMode, 'insert')

    def test_07(self):
        """ Switch to visual on selection
        """
        QTest.keyClick(self.qpart, Qt.Key_Right, Qt.ShiftModifier)
        self.assertEqual(self.vimMode, 'visual')

    def test_08(self):
        """ From VISUAL to VISUAL LINES
        """
        self.click('v')
        self.click('kkk')
        self.click('V')
        self.assertEqual(self.qpart.selectedText,
                         'The quick brown fox')
        self.assertEqual(self.vimMode, 'visual lines')

    def test_09(self):
        """ From VISUAL LINES to VISUAL
        """
        self.click('V')
        self.click('v')
        self.assertEqual(self.qpart.selectedText,
                         'The quick brown fox')
        self.assertEqual(self.vimMode, 'visual')

    def test_10(self):
        """ Insert mode with I
        """
        self.qpart.lines[1] = '   indented line'
        self.click('j8lI')
        self.click('Z')
        self.assertEqual(self.qpart.lines[1],
                         '   Zindented line')


class Move(_Test):
    def test_01(self):
        """Move hjkl
        """
        self.click("ll")
        self.assertEqual(self.qpart.cursorPosition, (0, 2))

        self.click("jjj")
        self.assertEqual(self.qpart.cursorPosition, (3, 2))

        self.click("h")
        self.assertEqual(self.qpart.cursorPosition, (3, 1))

        self.click("k")
        # (2, 1) on monospace, (2, 2) on non-monospace font
        self.assertIn(self.qpart.cursorPosition, ((2, 1), (2, 2)))

    def test_02(self):
        """w
        """
        self.qpart.lines[0] = 'word, comma, word'
        self.qpart.cursorPosition = (0, 0)
        for column in (4, 6, 11, 13, 17, 0):
            self.click('w')
            self.assertEqual(self.qpart.cursorPosition[1], column)

        self.assertEqual(self.qpart.cursorPosition, (1, 0))

    def test_03(self):
        """e
        """
        self.qpart.lines[0] = '  word, comma, word'
        self.qpart.cursorPosition = (0, 0)
        for column in (6, 7, 13, 14, 19, 5):
            self.click('e')
            self.assertEqual(self.qpart.cursorPosition[1], column)

        self.assertEqual(self.qpart.cursorPosition, (1, 5))

    def test_04(self):
        """$
        """
        self.click('$')
        self.assertEqual(self.qpart.cursorPosition, (0, 19))
        self.click('$')
        self.assertEqual(self.qpart.cursorPosition, (0, 19))

    def test_05(self):
        """0
        """
        self.qpart.cursorPosition = (0, 10)
        self.click('0')
        self.assertEqual(self.qpart.cursorPosition, (0, 0))

    def test_06(self):
        """G
        """
        self.qpart.cursorPosition = (0, 10)
        self.click('G')
        self.assertEqual(self.qpart.cursorPosition, (3, 0))

    def test_07(self):
        """gg
        """
        self.qpart.cursorPosition = (2, 10)
        self.click('gg')
        self.assertEqual(self.qpart.cursorPosition, (00, 0))

    def test_08(self):
        """ b word back
        """
        self.qpart.cursorPosition = (0, 19)
        self.click('b')
        self.assertEqual(self.qpart.cursorPosition, (0, 16))

        self.click('b')
        self.assertEqual(self.qpart.cursorPosition, (0, 10))

    def test_09(self):
        """ % to jump to next braket
        """
        self.qpart.lines[0] = '(asdf fdsa) xxx'
        self.qpart.cursorPosition = (0, 0)
        self.click('%')
        self.assertEqual(self.qpart.cursorPosition,
                         (0, 10))

    def test_10(self):
        """ ^ to jump to the first non-space char
        """
        self.qpart.lines[0] = '    indented line'
        self.qpart.cursorPosition = (0, 14)
        self.click('^')
        self.assertEqual(self.qpart.cursorPosition, (0, 4))

    def test_11(self):
        """ f to search forward
        """
        self.click('fv')
        self.assertEqual(self.qpart.cursorPosition,
                         (1, 7))

    def test_12(self):
        """ F to search backward
        """
        self.qpart.cursorPosition = (2, 0)
        self.click('Fv')
        self.assertEqual(self.qpart.cursorPosition,
                         (1, 7))

    def test_13(self):
        """ t to search forward
        """
        self.click('tv')
        self.assertEqual(self.qpart.cursorPosition,
                         (1, 6))

    def test_14(self):
        """ T to search backward
        """
        self.qpart.cursorPosition = (2, 0)
        self.click('Tv')
        self.assertEqual(self.qpart.cursorPosition,
                         (1, 8))

    def test_15(self):
        """ f in a composite command
        """
        self.click('dff')
        self.assertEqual(self.qpart.lines[0],
                         'ox')

    def test_16(self):
        """ E
        """
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (0, 0)
        for pos in (5, 6, 8, 9):
            self.click('e')
            self.assertEqual(self.qpart.cursorPosition[1],
                             pos)
        self.qpart.cursorPosition = (0, 0)
        for pos in (10, 22, 34, 45, 5):
            self.click('E')
            self.assertEqual(self.qpart.cursorPosition[1],
                             pos)

    def test_17(self):
        """ W
        """
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (0, 0)
        for pos in ((0, 12), (0, 24), (0, 35), (1, 0), (1, 6)):
            self.click('W')
            self.assertEqual(self.qpart.cursorPosition,
                             pos)

    def test_18(self):
        """ B
        """
        self.qpart.lines[0] = 'asdfk.xx.z  asdfk.xx.z  asdfk.xx.z asdfk.xx.z'
        self.qpart.cursorPosition = (1, 8)
        for pos in ((1, 6), (1, 0), (0, 35), (0, 24), (0, 12)):
            self.click('B')
            self.assertEqual(self.qpart.cursorPosition,
                             pos)

    def test_19(self):
        """ Enter, Return
        """
        self.qpart.lines[1] = '   indented line'
        self.qpart.lines[2] = '     more indented line'
        self.click(Qt.Key_Enter)
        self.assertEqual(self.qpart.cursorPosition, (1, 3))
        self.click(Qt.Key_Return)
        self.assertEqual(self.qpart.cursorPosition, (2, 5))


class Del(_Test):
    def test_01a(self):
        """Delete with x
        """
        self.qpart.cursorPosition = (0, 4)
        self.click("xxxxx")

        self.assertEqual(self.qpart.lines[0],
                         'The  brown fox')
        self.assertEqual(_globalClipboard.value, 'k')

    def test_01b(self):
        """Delete with x. Use count
        """
        self.qpart.cursorPosition = (0, 4)
        self.click("5x")

        self.assertEqual(self.qpart.lines[0],
                         'The  brown fox')
        self.assertEqual(_globalClipboard.value, 'quick')

    def test_02(self):
        """Composite delete with d. Left and right
        """
        self.qpart.cursorPosition = (1, 1)
        self.click("dl")
        self.assertEqual(self.qpart.lines[1],
                         'jmps over the')

        self.click("dh")
        self.assertEqual(self.qpart.lines[1],
                         'mps over the')

    def test_03(self):
        """Composite delete with d. Down
        """
        self.qpart.cursorPosition = (0, 2)
        self.click('dj')
        self.assertEqual(self.qpart.lines[:],
                         ['lazy dog',
                          'back'])
        self.assertEqual(self.qpart.cursorPosition[1], 0)

        # nothing deleted, if having only one line
        self.qpart.cursorPosition = (1, 1)
        self.click('dj')
        self.assertEqual(self.qpart.lines[:],
                         ['lazy dog',
                          'back'])


        self.click('k')
        self.click('dj')
        self.assertEqual(self.qpart.lines[:],
                         [''])
        self.assertEqual(_globalClipboard.value,
                         ['lazy dog',
                          'back'])

    def test_04(self):
        """Composite delete with d. Up
        """
        self.qpart.cursorPosition = (0, 2)
        self.click('dk')
        self.assertEqual(len(self.qpart.lines), 4)

        self.qpart.cursorPosition = (2, 1)
        self.click('dk')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'back'])
        self.assertEqual(_globalClipboard.value,
                         ['jumps over the',
                          'lazy dog'])

        self.assertEqual(self.qpart.cursorPosition[1], 0)

    def test_05(self):
        """Delete Count times
        """
        self.click('3dw')
        self.assertEqual(self.qpart.lines[0], 'fox')
        self.assertEqual(_globalClipboard.value,
                         'The quick brown ')

    def test_06(self):
        """Delete line
        dd
        """
        self.qpart.cursorPosition = (1, 0)
        self.click('dd')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'lazy dog',
                          'back'])

    def test_07(self):
        """Delete until end of file
        G
        """
        self.qpart.cursorPosition = (2, 0)
        self.click('dG')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the'])

    def test_08(self):
        """Delete until start of file
        gg
        """
        self.qpart.cursorPosition = (1, 0)
        self.click('dgg')
        self.assertEqual(self.qpart.lines[:],
                         ['lazy dog',
                          'back'])

    def test_09(self):
        """Delete with X
        """
        self.click("llX")

        self.assertEqual(self.qpart.lines[0],
                         'Te quick brown fox')

    def test_10(self):
        """Delete with D
        """
        self.click("jll")
        self.click("2D")

        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'ju',
                          'back'])


class Edit(_Test):
    def test_01(self):
        """Undo
        """
        oldText = self.qpart.text
        self.click('ddu')
        modifiedText = self.qpart.text  # pylint: disable=unused-variable
        self.assertEqual(self.qpart.text, oldText)
        # NOTE this part of test doesn't work. Don't know why.
        # self.click('U')
        # self.assertEqual(self.qpart.text, modifiedText)

    def test_02(self):
        """Change with C
        """
        self.click("lllCpig")

        self.assertEqual(self.qpart.lines[0],
                         'Thepig')

    def test_03(self):
        """ Substitute with s
        """
        self.click('j4sz')
        self.assertEqual(self.qpart.lines[1],
                         'zs over the')

    def test_04(self):
        """Replace char with r
        """
        self.qpart.cursorPosition = (0, 4)
        self.click('rZ')
        self.assertEqual(self.qpart.lines[0],
                         'The Zuick brown fox')

        self.click('rW')
        self.assertEqual(self.qpart.lines[0],
                         'The Wuick brown fox')

    def test_05(self):
        """Change 2 words with c
        """
        self.click('c2e')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[0],
                         'asdf brown fox')

    def test_06(self):
        """Open new line with o
        """
        self.qpart.lines = ['    indented line',
                            '    next indented line']
        self.click('o')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[:],
                         ['    indented line',
                          '    asdf',
                          '    next indented line'])

    def test_07(self):
        """Open new line with O

        Check indentation
        """
        self.qpart.lines = ['    indented line',
                            '    next indented line']
        self.click('j')
        self.click('O')
        self.click('asdf')
        self.assertEqual(self.qpart.lines[:],
                         ['    indented line',
                          '    asdf',
                          '    next indented line'])

    def test_08(self):
        """ Substitute with S
        """
        self.qpart.lines = ['    indented line',
                            '    next indented line']
        self.click('ljS')
        self.click('xyz')
        self.assertEqual(self.qpart.lines[:],
                         ['    indented line',
                          '    xyz'])

    def test_09(self):
        """ % to jump to next braket
        """
        self.qpart.lines[0] = '(asdf fdsa) xxx'
        self.qpart.cursorPosition = (0, 0)
        self.click('d%')
        self.assertEqual(self.qpart.lines[0],
                         ' xxx')

    def test_10(self):
        """ J join lines
        """
        self.click('2J')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox jumps over the lazy dog',
                          'back'])


class Indent(_Test):
    def test_01(self):
        """ Increase indent with >j, decrease with <j
        """
        self.click('>2j')
        self.assertEqual(self.qpart.lines[:],
                         ['    The quick brown fox',
                          '    jumps over the',
                          '    lazy dog',
                          'back'])

        self.click('<j')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the',
                          '    lazy dog',
                          'back'])

    def test_02(self):
        """ Increase indent with >>, decrease with <<
        """
        self.click('>>')
        self.click('>>')
        self.assertEqual(self.qpart.lines[0],
                         '        The quick brown fox')

        self.click('<<')
        self.assertEqual(self.qpart.lines[0],
                         '    The quick brown fox')

    def test_03(self):
        """ Autoindent with =j
        """
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('=j')
        self.assertEqual(self.qpart.lines[:],
                         ['    The quick brown fox',
                          '    jumps over the',
                          '    lazy dog',
                          'back'])

    def test_04(self):
        """ Autoindent with ==
        """
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('==')
        self.assertEqual(self.qpart.lines[:],
                         ['    The quick brown fox',
                          '    jumps over the',
                          'lazy dog',
                          'back'])

    def test_11(self):
        """ Increase indent with >, decrease with < in visual mode
        """
        self.click('v2>')
        self.assertEqual(self.qpart.lines[:2],
                         ['        The quick brown fox',
                          'jumps over the'])

        self.click('v<')
        self.assertEqual(self.qpart.lines[:2],
                         ['    The quick brown fox',
                          'jumps over the'])

    def test_12(self):
        """ Autoindent with = in visual mode
        """
        self.click('i    ')
        self.click(Qt.Key_Escape)
        self.click('j')
        self.click('Vj=')
        self.assertEqual(self.qpart.lines[:],
                         ['    The quick brown fox',
                          '    jumps over the',
                          '    lazy dog',
                          'back'])


class CopyPaste(_Test):
    def test_02(self):
        """Paste text with p
        """
        self.qpart.cursorPosition = (0, 4)
        self.click("5x")
        self.assertEqual(self.qpart.lines[0],
                         'The  brown fox')

        self.click("p")
        self.assertEqual(self.qpart.lines[0],
                         'The  quickbrown fox')

    def test_03(self):
        """Paste lines with p
        """
        self.qpart.cursorPosition = (1, 2)
        self.click("2dd")
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'back'])

        self.click("kkk")
        self.click("p")
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the',
                          'lazy dog',
                          'back'])

    def test_04(self):
        """Paste lines with P
        """
        self.qpart.cursorPosition = (1, 2)
        self.click("2dd")
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'back'])

        self.click("P")
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the',
                          'lazy dog',
                          'back'])

    def test_05(self):
        """ Yank line with yy
        """
        self.click('y2y')
        self.click('jll')
        self.click('p')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the',
                          'The quick brown fox',
                          'jumps over the',
                          'lazy dog',
                          'back'])

    def test_06(self):
        """ Yank until the end of line
        """
        self.click('2wYo')
        self.click(Qt.Key_Escape)
        self.click('P')
        self.assertEqual(self.qpart.lines[1],
                         'brown fox')

    def test_08(self):
        """ Composite yank with y, paste with P
        """
        self.click('y2w')
        self.click('P')
        self.assertEqual(self.qpart.lines[0],
                         'The quick The quick brown fox')




class Visual(_Test):
    def test_01(self):
        """ x
        """
        self.click('v')
        self.assertEqual(self.vimMode, 'visual')
        self.click('2w')
        self.assertEqual(self.qpart.selectedText, 'The quick ')
        self.click('x')
        self.assertEqual(self.qpart.lines[0],
                         'brown fox')
        self.assertEqual(self.vimMode, 'normal')

    def test_02(self):
        """Append with a
        """
        self.click("vllA")
        self.click("asdf ")
        self.assertEqual(self.qpart.lines[0],
                         'The asdf quick brown fox')

    def test_03(self):
        """Replace with r
        """
        self.qpart.cursorPosition = (0, 16)
        self.click("v8l")
        self.click("rz")
        self.assertEqual(self.qpart.lines[0:2],
                         ['The quick brown zzz',
                          'zzzzz over the'])

    def test_04(self):
        """Replace selected lines with R
        """
        self.click("vjl")
        self.click("R")
        self.click("Z")
        self.assertEqual(self.qpart.lines[:],
                         ['Z',
                          'lazy dog',
                          'back'])

    def test_05(self):
        """Reset selection with u
        """
        self.qpart.cursorPosition = (1, 3)
        self.click('vjl')
        self.click('u')
        self.assertEqual(self.qpart.selectedPosition, ((1, 3), (1, 3)))

    def test_06(self):
        """Yank with y and paste with p
        """
        self.qpart.cursorPosition = (0, 4)
        self.click("ve")
        #print self.qpart.selectedText
        self.click("y")
        self.click(Qt.Key_Escape)
        self.qpart.cursorPosition = (0, 16)
        self.click("ve")
        self.click("p")
        self.assertEqual(self.qpart.lines[0],
                         'The quick brown quick')

    def test_07(self):
        """ Replace word when pasting
        """
        self.click("vey")  # copy word
        self.click('ww')  # move
        self.click('vep')  # replace word
        self.assertEqual(self.qpart.lines[0],
                         'The quick The fox')

    def test_08(self):
        """Change with c
        """
        self.click("w")
        self.click("vec")
        self.click("slow")
        self.assertEqual(self.qpart.lines[0],
                         'The slow brown fox')

    def test_09(self):
        """ Delete lines with X and D
        """
        self.click('jvlX')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'lazy dog',
                          'back'])

        self.click('u')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the',
                          'lazy dog',
                          'back'])

        self.click('vjD')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'back'])

    def test_10(self):
        """ Check if f works
        """
        self.click('vfo')
        self.assertEqual(self.qpart.selectedText,
                         'The quick bro')

    def test_11(self):
        """ J join lines
        """
        self.click('jvjJ')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          'jumps over the lazy dog',
                          'back'])


class VisualLines(_Test):
    def test_01(self):
        """ x Delete
        """
        self.click('V')
        self.assertEqual(self.vimMode, 'visual lines')
        self.click('x')
        self.click('p')
        self.assertEqual(self.qpart.lines[:],
                         ['jumps over the',
                          'The quick brown fox',
                          'lazy dog',
                          'back'])
        self.assertEqual(self.vimMode, 'normal')

    def test_02(self):
        """ Replace text when pasting
        """
        self.click('Vy')
        self.click('j')
        self.click('Vp')
        self.assertEqual(self.qpart.lines[0:3],
                         ['The quick brown fox',
                          'The quick brown fox',
                          'lazy dog',])

    def test_06(self):
        """Yank with y and paste with p
        """
        self.qpart.cursorPosition = (0, 4)
        self.click("V")
        self.click("y")
        self.click(Qt.Key_Escape)
        self.qpart.cursorPosition = (0, 16)
        self.click("p")
        self.assertEqual(self.qpart.lines[0:3],
                         ['The quick brown fox',
                          'The quick brown fox',
                          'jumps over the'])

    def test_07(self):
        """Change with c
        """
        self.click("Vc")
        self.click("slow")
        self.assertEqual(self.qpart.lines[0],
                         'slow')


class Repeat(_Test):
    def test_01(self):
        """ Repeat o
        """
        self.click('o')
        self.click(Qt.Key_Escape)
        self.click('j2.')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          '',
                          'jumps over the',
                          '',
                          '',
                          'lazy dog',
                          'back'])

    def test_02(self):
        """ Repeat o. Use count from previous command
        """
        self.click('2o')
        self.click(Qt.Key_Escape)
        self.click('j.')
        self.assertEqual(self.qpart.lines[:],
                         ['The quick brown fox',
                          '',
                          '',
                          'jumps over the',
                          '',
                          '',
                          'lazy dog',
                          'back'])

    def test_03(self):
        """ Repeat O
        """
        self.click('O')
        self.click(Qt.Key_Escape)
        self.click('2j2.')
        self.assertEqual(self.qpart.lines[:],
                         ['',
                          'The quick brown fox',
                          '',
                          '',
                          'jumps over the',
                          'lazy dog',
                          'back'])

    def test_04(self):
        """ Repeat p
        """
        self.click('ylp.')
        self.assertEqual(self.qpart.lines[0],
                         'TTThe quick brown fox')

    def test_05(self):
        """ Repeat p
        """
        self.click('x...')
        self.assertEqual(self.qpart.lines[0],
                         'quick brown fox')

    def test_06(self):
        """ Repeat D
        """
        self.click('Dj.')
        self.assertEqual(self.qpart.lines[:],
                         ['',
                          '',
                          'lazy dog',
                          'back'])

    def test_07(self):
        """ Repeat dw
        """
        self.click('dw')
        self.click('j0.')
        self.assertEqual(self.qpart.lines[:],
                         ['quick brown fox',
                          'over the',
                          'lazy dog',
                          'back'])

    def test_08(self):
        """ Repeat Visual x
        """
        self.qpart.lines.append('one more')
        self.click('Vjx')
        self.click('.')
        self.assertEqual(self.qpart.lines[:],
                         ['one more'])

    def test_09(self):
        """ Repeat visual X
        """
        self.qpart.lines.append('one more')
        self.click('vjX')
        self.click('.')
        self.assertEqual(self.qpart.lines[:],
                         ['one more'])

    def test_10(self):
        """ Repeat Visual >
        """
        self.qpart.lines.append('one more')
        self.click('Vj>')
        self.click('3j')
        self.click('.')
        self.assertEqual(self.qpart.lines[:],
                         ['    The quick brown fox',
                          '    jumps over the',
                          'lazy dog',
                          '    back',
                          '    one more'])

if __name__ == '__main__':
    unittest.main()
