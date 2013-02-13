"""
"""
import sys
import traceback

from functools import wraps
from PyQt4.QtGui import (
    QWidget, QPlainTextEdit, QVBoxLayout, QTextCursor, QTextCharFormat,
    QFont, QSizePolicy
)

from PyQt4.QtCore import Qt, QObject, QEvent, QCoreApplication, QThread, QSize
from PyQt4.QtCore import pyqtSignal as Signal


class TerminalView(QPlainTextEdit):
    def __init__(self, *args, **kwargs):
        QPlainTextEdit.__init__(self, *args, **kwargs)
        self.setFrameStyle(QPlainTextEdit.NoFrame)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        font = self.font()
        if hasattr(QFont, "Monospace"):
            # Why is this not available on Debian squeeze
            font.setStyleHint(QFont.Monospace)
        else:
            font.setStyleHint(QFont.Courier)

        font.setFamily("Monaco")
        font.setPointSize(12)
        self.setFont(font)

    def sizeHint(self):
        metrics = self.fontMetrics()
        width = metrics.boundingRect("_" * 81).width()
        height = metrics.lineSpacing()
        scroll_width = self.verticalScrollBar().width()
        size = QSize(width + scroll_width, height * 25)
        return size


class OutputView(QWidget):
    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)

        self.__lines = 5000

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.__text = TerminalView()

        self.__currentCharFormat = self.__text.currentCharFormat()

        self.layout().addWidget(self.__text)

    def setMaximumLines(self, lines):
        """
        Set the maximum number of lines to keep displayed.
        """
        if self.__lines != lines:
            self.__lines = lines
            self.__text.setMaximumBlockCount(lines)

    def maximumLines(self):
        """
        Return the maximum number of lines in the display.
        """
        return self.__lines

    def clear(self):
        """
        Clear the displayed text.
        """
        self.__text.clear()

    def setCurrentCharFormat(self, charformat):
        """Set the QTextCharFormat to be used when writing.
        """
        if self.__currentCharFormat != charformat:
            self.__currentCharFormat = charformat

    def currentCharFormat(self):
        return self.__currentCharFormat

    def toPlainText(self):
        """
        Return the full contents of the output view.
        """
        return self.__text.toPlainText()

    # A file like interface.
    def write(self, string):
        self.__text.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        self.__text.setCurrentCharFormat(self.__currentCharFormat)

        self.__text.insertPlainText(string)

    def writelines(self, lines):
        self.write("".join(lines))

    def flush(self):
        QCoreApplication.flush()

    def writeWithFormat(self, string, charformat):
        self.__text.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        self.__text.setCurrentCharFormat(charformat)
        self.__text.insertPlainText(string)

    def writelinesWithFormat(self, lines, charformat):
        self.writeWithFormat("".join(lines), charformat)

    def formated(self, color=None, background=None, weight=None,
                 italic=None, underline=None, font=None):
        """
        Return a formated file like object proxy.
        """
        charformat = update_char_format(
            self.currentCharFormat(), color, background, weight,
            italic, underline, font
        )
        return formater(self, charformat)


def update_char_format(baseformat, color=None, background=None, weight=None,
                       italic=None, underline=None, font=None):
    """
    Return a copy of `baseformat` :class:`QTextCharFormat` with
    updated color, weight, background and font properties.

    """
    charformat = QTextCharFormat(baseformat)

    if color is not None:
        charformat.setForeground(color)

    if background is not None:
        charformat.setBackground(background)

    if font is not None:
        charformat.setFont(font)
    else:
        font = update_font(baseformat.font(), weight, italic, underline)
        charformat.setFont(font)

    return charformat


def update_font(basefont, weight=None, italic=None, underline=None,
                pixelSize=None, pointSize=None):
    """
    Return a copy of `basefont` :class:`QFont` with updated properties.
    """
    font = QFont(basefont)

    if weight is not None:
        font.setWeight(weight)

    if italic is not None:
        font.setItalic(italic)

    if underline is not None:
        font.setUnderline(underline)

    if pixelSize is not None:
        font.setPixelSize(pixelSize)

    if pointSize is not None:
        font.setPointSize(pointSize)

    return font


class formater(object):
    def __init__(self, outputview, charformat):
        self.outputview = outputview
        self.charformat = charformat

    def write(self, string):
        self.outputview.writeWithFormat(string, self.charformat)

    def writelines(self, lines):
        self.outputview.writelines(lines, self.charformat)

    def flush(self):
        self.outputview.flush()

    def formated(self, color=None, background=None, weight=None,
                 italic=None, underline=None, font=None):
        charformat = update_char_format(self.charformat, color, background,
                                        weight, italic, underline, font)
        return formater(self.outputview, charformat)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.outputview = None
        self.charformat = None


class QueuedCallEvent(QEvent):
    QueuedCall = QEvent.registerEventType()

    def __init__(self, function, args, kwargs):
        QEvent.__init__(self, QueuedCallEvent.QueuedCall)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._exc_info = None
        self._state = 0

    def call(self):
        try:
            self._result = self.function(*self.args, **self.kwargs)
            self._state = 1
        except Exception as ex:
            self._exc_info = (type(ex), ex.args, None)
            raise

    def result(self):
        if self._state == 1:
            return self._result
        elif self._exc_info:
            raise self._exc_info[0](self._exc_info[1])
        else:
            # Should this block, add timeout?
            raise RuntimeError("Result not yet ready")

    def isready(self):
        return self._state == 1 or self._exc_info


def queued(method):
    """
    Run method from the event queue.
    """
    @wraps(method)
    def delay_method_call(self, *args, **kwargs):
        event = QueuedCallEvent(method.__get__(self), args, kwargs)
        QCoreApplication.postEvent(self, event)

    return delay_method_call


def queued_blocking(method):
    """
    Run method from the event queue and wait until the event is processed.
    Return the call's return value.

    """
    @wraps(method)
    def delay_method_call(self, *args, **kwargs):
        event = QueuedCallEvent(method, (self,) + args, kwargs)
        QCoreApplication.postEvent(self, event)
        QCoreApplication.sendPostedEvents()
        return event.result()

    return delay_method_call


class TextStream(QObject):
    stream = Signal(str)
    flushed = Signal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    @queued
    def write(self, string):
        self.stream.emit(string)

    @queued
    def writelines(self, lines):
        self.stream.emit("".join(lines))

    @queued_blocking
    def flush(self):
        self.flushed.emit()

    def customEvent(self, event):
        if event.type() == QueuedCallEvent.QueuedCall:
            event.call()
            event.accept()


class ExceptHook(QObject):
    handledException = Signal()

    def __init__(self, parent=None, stream=None):
        QObject.__init__(self, parent)
        self.stream = stream

    def __call__(self, exc_type, exc_value, tb):
        text = traceback.format_exception(exc_type, exc_value, tb)
        separator = "-" * 80 + "\n"
        if QThread.currentThread() != QCoreApplication.instance().thread():
            header = exc_type.__name__ + " (in non GUI thread)"
        else:
            header = exc_type.__name__

        header_fmt = "%%%is\n"
        if tb:
            header += (header_fmt % (80 - len(header))) % text[0].strip()
            del text[0]
        else:
            header

        if self.stream is None:
            stream = sys.stderr
        else:
            stream = self.stream

        stream.writelines([separator, header] + text)

        self.handledException.emit()
