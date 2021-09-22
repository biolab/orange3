"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""  # pylint: disable=duplicate-code
import time

from AnyQt.QtCore import QTimer
from AnyQt.QtGui import QKeySequence
from AnyQt.QtTest import QTest
from AnyQt.QtCore import Qt, QCoreApplication

from Orange.widgets import widget
from Orange.widgets.data.utils.pythoneditor.editor import PythonEditor


def _processPendingEvents(app):
    """Process pending application events.
    Timeout is used, because on Windows hasPendingEvents() always returns True
    """
    t = time.time()
    while app.hasPendingEvents() and (time.time() - t < 0.1):
        app.processEvents()


def in_main_loop(func, *_):
    """Decorator executes test method in the QApplication main loop.
    QAction shortcuts doesn't work, if main loop is not running.
    Do not use for tests, which doesn't use main loop, because it slows down execution.
    """
    def wrapper(*args):
        app = QCoreApplication.instance()
        self = args[0]

        def execWithArgs():
            self.qpart.show()
            QTest.qWaitForWindowExposed(self.qpart)
            _processPendingEvents(app)

            try:
                func(*args)
            finally:
                _processPendingEvents(app)
                app.quit()

        QTimer.singleShot(0, execWithArgs)

        app.exec_()

    wrapper.__name__ = func.__name__  # for unittest test runner
    return wrapper

class SimpleWidget(widget.OWWidget):
    name = "Simple widget"

    def __init__(self):
        super().__init__()
        self.qpart = PythonEditor(self)
        self.mainArea.layout().addWidget(self.qpart)


def keySequenceClicks(widget_, keySequence, extraModifiers=Qt.NoModifier):
    """Use QTest.keyClick to send a QKeySequence to a widget."""
    # pylint: disable=line-too-long
    # This is based on a simplified version of http://stackoverflow.com/questions/14034209/convert-string-representation-of-keycode-to-qtkey-or-any-int-and-back. I added code to handle the case in which the resulting key contains a modifier (for example, Shift+Home). When I execute QTest.keyClick(widget, keyWithModifier), I get the error "ASSERT: "false" in file .\qasciikey.cpp, line 495". To fix this, the following code splits the key into a key and its modifier.
    # Bitmask for all modifier keys.
    modifierMask = int(Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier |
                       Qt.MetaModifier |  Qt.KeypadModifier)
    ks = QKeySequence(keySequence)
    # For now, we don't handle a QKeySequence("Ctrl") or any other modified by itself.
    assert ks.count() > 0
    for _, key in enumerate(ks):
        modifiers = Qt.KeyboardModifiers((key & modifierMask) | extraModifiers)
        key = key & ~modifierMask
        QTest.keyClick(widget_, key, modifiers, 10)
