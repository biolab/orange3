"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
from AnyQt.QtGui import QKeySequence
from AnyQt.QtTest import QTest
from AnyQt.QtCore import Qt

from orangewidget.utils import enum_as_int

from Orange.widgets.data.utils.pythoneditor.editor import PythonEditor
from Orange.widgets.data.utils.pythoneditor.vim import key_code
from Orange.widgets.tests.base import GuiTest


class EditorTest(GuiTest):
    def setUp(self) -> None:
        super().setUp()
        self.qpart = PythonEditor()

    def tearDown(self) -> None:
        self.qpart.terminate()
        del self.qpart
        super().tearDown()


def keySequenceClicks(widget_, keySequence, extraModifiers=Qt.NoModifier):
    """Use QTest.keyClick to send a QKeySequence to a widget."""
    # pylint: disable=line-too-long
    # This is based on a simplified version of http://stackoverflow.com/questions/14034209/convert-string-representation-of-keycode-to-qtkey-or-any-int-and-back. I added code to handle the case in which the resulting key contains a modifier (for example, Shift+Home). When I execute QTest.keyClick(widget, keyWithModifier), I get the error "ASSERT: "false" in file .\qasciikey.cpp, line 495". To fix this, the following code splits the key into a key and its modifier.
    # Bitmask for all modifier keys.
    modifierMask = enum_as_int(Qt.KeyboardModifierMask)
    ks = QKeySequence(keySequence)
    # For now, we don't handle a QKeySequence("Ctrl") or any other modified by itself.
    assert ks.count() > 0
    for _, key in enumerate(ks):
        key = key_code(key)
        modifiers = Qt.KeyboardModifiers((key & modifierMask) | enum_as_int(extraModifiers))
        key = key & ~modifierMask
        QTest.keyClick(widget_, Qt.Key(key), modifiers, 10)
