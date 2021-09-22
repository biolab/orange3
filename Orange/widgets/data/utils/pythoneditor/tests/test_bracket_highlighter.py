"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import unittest

from Orange.widgets.data.utils.pythoneditor.brackethighlighter import BracketHighlighter
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

    def _verify(self, actual, expected):
        converted = []
        for item in actual:
            if item.format.foreground().color() == BracketHighlighter.MATCHED_COLOR:
                matched = True
            elif item.format.foreground().color() == BracketHighlighter.UNMATCHED_COLOR:
                matched = False
            else:
                self.fail("Invalid color")
            start = item.cursor.selectionStart()
            end = item.cursor.selectionEnd()
            converted.append((start, end, matched))

        self.assertEqual(converted, expected)

    def test_1(self):
        self.qpart.lines = \
            ['func(param,',
             '     "text ( param"))']

        firstBlock = self.qpart.document().firstBlock()
        secondBlock = firstBlock.next()

        bh = BracketHighlighter()

        self._verify(bh.extraSelections(self.qpart, firstBlock, 1),
                     [])

        self._verify(bh.extraSelections(self.qpart, firstBlock, 4),
                     [(4, 5, True), (31, 32, True)])
        self._verify(bh.extraSelections(self.qpart, firstBlock, 5),
                     [(4, 5, True), (31, 32, True)])
        self._verify(bh.extraSelections(self.qpart, secondBlock, 11),
                     [])
        self._verify(bh.extraSelections(self.qpart, secondBlock, 19),
                     [(31, 32, True), (4, 5, True)])
        self._verify(bh.extraSelections(self.qpart, secondBlock, 20),
                     [(32, 33, False)])
        self._verify(bh.extraSelections(self.qpart, secondBlock, 21),
                     [(32, 33, False)])


if __name__ == '__main__':
    unittest.main()
