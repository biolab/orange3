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


class Test(WidgetTest):
    """Base class for tests
    """

    def setUp(self):
        self.widget = self.create_widget(SimpleWidget)
        self.qpart = self.widget.qpart

    def tearDown(self):
        self.qpart.terminate()

    def _ws_test(self,
                 text,
                 expectedResult,
                 drawAny=None,
                 drawIncorrect=None,
                 useTab=None,
                 indentWidth=None):
        if drawAny is None:
            drawAny = [True, False]
        if drawIncorrect is None:
            drawIncorrect = [True, False]
        if useTab is None:
            useTab = [True, False]
        if indentWidth is None:
            indentWidth = [1, 2, 3, 4, 8]
        for drawAnyVal in drawAny:
            self.qpart.drawAnyWhitespace = drawAnyVal

            for drawIncorrectVal in drawIncorrect:
                self.qpart.drawIncorrectIndentation = drawIncorrectVal

                for useTabVal in useTab:
                    self.qpart.indentUseTabs = useTabVal

                    for indentWidthVal in indentWidth:
                        self.qpart.indentWidth = indentWidthVal
                        try:
                            self._verify(text, expectedResult)
                        except:
                            print("Failed params:\n\tany {}\n\tincorrect {}\n\ttabs {}\n\twidth {}"
                                  .format(self.qpart.drawAnyWhitespace,
                                          self.qpart.drawIncorrectIndentation,
                                          self.qpart.indentUseTabs,
                                          self.qpart.indentWidth))
                            raise

    def _verify(self, text, expectedResult):
        res = self.qpart._chooseVisibleWhitespace(text)  # pylint: disable=protected-access
        for index, value in enumerate(expectedResult):
            if value == '1':
                if not res[index]:
                    self.fail("Item {} is not True:\n\t{}".format(index, res))
            elif value == '0':
                if res[index]:
                    self.fail("Item {} is not False:\n\t{}".format(index, res))
            else:
                assert value == ' '

    def test_1(self):
        # Trailing
        self._ws_test('   m xyz\t ',
                      '   0 00011',
                      drawIncorrect=[True])

    def test_2(self):
        # Tabs in space mode
        self._ws_test('\txyz\t',
                      '10001',
                      drawIncorrect=[True], useTab=[False])

    def test_3(self):
        # Spaces in tab mode
        self._ws_test('    2   3     5',
                      '111100000000000',
                      drawIncorrect=[True], drawAny=[False], indentWidth=[3], useTab=[True])

    def test_4(self):
        # Draw any
        self._ws_test(' 1 1  2   3     5\t',
                      '100011011101111101',
                      drawAny=[True],
                      indentWidth=[2, 3, 4, 8])


if __name__ == '__main__':
    unittest.main()
