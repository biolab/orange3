"""
A splash screen widget with support for positioning of the message text.

"""

from AnyQt.QtWidgets import QSplashScreen, QWidget, QApplication
from AnyQt.QtGui import (
    QPixmap, QPainter, QTextDocument, QTextBlockFormat, QTextCursor
)
from AnyQt.QtCore import Qt

from .utils import is_transparency_supported

if hasattr(Qt, "mightBeRichText"):
    mightBeRichText = Qt.mightBeRichText
else:
    def mightBeRichText(text):
        return False


class SplashScreen(QSplashScreen):
    """
    Splash screen widget.

    Parameters
    ----------
    parent : :class:`QWidget`
        Parent widget

    pixmap : :class:`QPixmap`
        Splash window pixmap.

    textRect : :class:`QRect`
        Bounding rectangle of the shown message on the widget.

    textFormat : Qt.TextFormat
        How message text format should be interpreted.
    """
    def __init__(self, parent=None, pixmap=None, textRect=None,
                 textFormat=Qt.PlainText, **kwargs):
        QSplashScreen.__init__(self, parent, **kwargs)
        self.__textRect = textRect
        self.__message = ""
        self.__color = Qt.black
        self.__alignment = Qt.AlignLeft
        self.__textFormat = textFormat

        if pixmap is None:
            pixmap = QPixmap()

        self.setPixmap(pixmap)

        self.setAutoFillBackground(False)
        # Also set FramelesWindowHint (if not already set)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def setTextRect(self, rect):
        """
        Set the rectangle (:class:`QRect`) in which to show the message text.
        """
        if self.__textRect != rect:
            self.__textRect = rect
            self.update()

    def textRect(self):
        """
        Return the text message rectangle.
        """
        return self.__textRect

    def textFormat(self):
        return self.__textFormat

    def setTextFormat(self, format):
        if format != self.__textFormat:
            self.__textFormat = format
            self.update()

    def showEvent(self, event):
        QSplashScreen.showEvent(self, event)
        # Raise to top on show.
        self.raise_()

    def drawContents(self, painter):
        """
        Reimplementation of drawContents to limit the drawing
        inside `textRext`.

        """
        painter.setPen(self.__color)
        painter.setFont(self.font())

        if self.__textRect:
            rect = self.__textRect
        else:
            rect = self.rect().adjusted(5, 5, -5, -5)

        tformat = self.__textFormat

        if tformat == Qt.AutoText:
            if mightBeRichText(self.__message):
                tformat = Qt.RichText
            else:
                tformat = Qt.PlainText

        if tformat == Qt.RichText:
            doc = QTextDocument()
            doc.setHtml(self.__message)
            doc.setTextWidth(rect.width())
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.Document)
            fmt = QTextBlockFormat()
            fmt.setAlignment(self.__alignment)
            cursor.mergeBlockFormat(fmt)
            painter.save()
            painter.translate(rect.topLeft())
            doc.drawContents(painter)
            painter.restore()
        else:
            painter.drawText(rect, self.__alignment, self.__message)

    def showMessage(self, message, alignment=Qt.AlignLeft, color=Qt.black):
        """
        Show the `message` with `color` and `alignment`.
        """
        # Need to store all this arguments for drawContents (no access
        # methods)
        self.__alignment = alignment
        self.__color = color
        self.__message = message
        QSplashScreen.showMessage(self, message, alignment, color)
        QApplication.instance().processEvents()

    # Reimplemented to allow graceful fall back if the windowing system
    # does not support transparency.
    def setPixmap(self, pixmap):
        self.setAttribute(Qt.WA_TranslucentBackground,
                          pixmap.hasAlpha() and \
                          is_transparency_supported())

        self.__pixmap = pixmap

        QSplashScreen.setPixmap(self, pixmap)
        if pixmap.hasAlpha() and not is_transparency_supported():
            self.setMask(pixmap.createHeuristicMask())

    def repaint(self):
        QWidget.repaint(self)
        QApplication.flush()

    def event(self, event):
        if event.type() == event.Paint:
            pixmap = self.__pixmap
            painter = QPainter(self)
            if not pixmap.isNull():
                painter.drawPixmap(0, 0, pixmap)
            self.drawContents(painter)
            return True
        return QSplashScreen.event(self, event)
