"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import unittest

import os.path
import sys

from Orange.widgets.data.utils.pythoneditor.tests.test_indenter.indenttest import IndentTest

sys.path.append(os.path.abspath(os.path.join(__file__, '..')))


class Test(IndentTest):
    LANGUAGE = 'Python'
    INDENT_WIDTH = 2

    def test_dedentReturn(self):
        origin = [
            "def some_function():",
            "  return"]
        expected = [
            "def some_function():",
            "  return",
            "pass"]

        self.setOrigin(origin)

        self.setCursorPosition(1, 11)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_dedentContinue(self):
        origin = [
            "while True:",
            "  continue"]
        expected = [
            "while True:",
            "  continue",
            "pass"]

        self.setOrigin(origin)

        self.setCursorPosition(1, 11)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_keepIndent2(self):
        origin = [
            "class my_class():",
            "  def my_fun():",
            '    print "Foo"',
            "    print 3"]
        expected = [
            "class my_class():",
            "  def my_fun():",
            '    print "Foo"',
            "    print 3",
            "    pass"]

        self.setOrigin(origin)

        self.setCursorPosition(3, 12)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_keepIndent4(self):
        origin = [
            "def some_function():"]
        expected = [
            "def some_function():",
            "  pass",
            "",
            "pass"]

        self.setOrigin(origin)

        self.setCursorPosition(0, 22)
        self.enter()
        self.type("pass")
        self.enter()
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_dedentRaise(self):
        origin = [
            "try:",
            "  raise"]
        expected = [
            "try:",
            "  raise",
            "except:"]

        self.setOrigin(origin)

        self.setCursorPosition(1, 9)
        self.enter()
        self.type("except:")
        self.verifyExpected(expected)

    def test_indentColon1(self):
        origin = [
            "def some_function(param, param2):"]
        expected = [
            "def some_function(param, param2):",
            "  pass"]

        self.setOrigin(origin)

        self.setCursorPosition(0, 34)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_indentColon2(self):
        origin = [
            "def some_function(1,",
            "                  2):"
        ]
        expected = [
            "def some_function(1,",
            "                  2):",
            "  pass"
        ]

        self.setOrigin(origin)

        self.setCursorPosition(1, 21)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_indentColon3(self):
        """Do not indent colon if hanging indentation used
        """
        origin = [
            "     a = {1:"
        ]
        expected = [
            "     a = {1:",
            "          x"
        ]

        self.setOrigin(origin)

        self.setCursorPosition(0, 12)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)

    def test_dedentPass(self):
        origin = [
            "def some_function():",
            "  pass"]
        expected = [
            "def some_function():",
            "  pass",
            "pass"]

        self.setOrigin(origin)

        self.setCursorPosition(1, 8)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_dedentBreak(self):
        origin = [
            "def some_function():",
            "  return"]
        expected = [
            "def some_function():",
            "  return",
            "pass"]

        self.setOrigin(origin)

        self.setCursorPosition(1, 11)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_keepIndent3(self):
        origin = [
            "while True:",
            "  returnFunc()",
            "  myVar = 3"]
        expected = [
            "while True:",
            "  returnFunc()",
            "  myVar = 3",
            "  pass"]

        self.setOrigin(origin)

        self.setCursorPosition(2, 12)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_keepIndent1(self):
        origin = [
            "def some_function(param, param2):",
            "  a = 5",
            "  b = 7"]
        expected = [
            "def some_function(param, param2):",
            "  a = 5",
            "  b = 7",
            "  pass"]

        self.setOrigin(origin)

        self.setCursorPosition(2, 8)
        self.enter()
        self.type("pass")
        self.verifyExpected(expected)

    def test_autoIndentAfterEmpty(self):
        origin = [
            "while True:",
            "   returnFunc()",
            "",
            "   myVar = 3"]
        expected = [
            "while True:",
            "   returnFunc()",
            "",
            "   x",
            "   myVar = 3"]

        self.setOrigin(origin)

        self.setCursorPosition(2, 0)
        self.enter()
        self.tab()
        self.type("x")
        self.verifyExpected(expected)

    def test_hangingIndentation(self):
        origin = [
            "     return func (something,",
        ]
        expected = [
            "     return func (something,",
            "                  x",
        ]

        self.setOrigin(origin)

        self.setCursorPosition(0, 28)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)

    def test_hangingIndentation2(self):
        origin = [
            "     return func (",
            "         something,",
        ]
        expected = [
            "     return func (",
            "         something,",
            "         x",
        ]

        self.setOrigin(origin)

        self.setCursorPosition(1, 19)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)

    def test_hangingIndentation3(self):
        origin = [
            "     a = func (",
            "         something)",
        ]
        expected = [
            "     a = func (",
            "         something)",
            "     x",
        ]

        self.setOrigin(origin)

        self.setCursorPosition(1, 19)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)

    def test_hangingIndentation4(self):
        origin = [
            "     return func(a,",
            "                 another_func(1,",
            "                              2),",
        ]
        expected = [
            "     return func(a,",
            "                 another_func(1,",
            "                              2),",
            "                 x"
        ]

        self.setOrigin(origin)

        self.setCursorPosition(2, 33)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)

    def test_hangingIndentation5(self):
        origin = [
            "     return func(another_func(1,",
            "                              2),",
        ]
        expected = [
            "     return func(another_func(1,",
            "                              2),",
            "                 x"
        ]

        self.setOrigin(origin)

        self.setCursorPosition(2, 33)
        self.enter()
        self.type("x")
        self.verifyExpected(expected)


if __name__ == '__main__':
    unittest.main()
