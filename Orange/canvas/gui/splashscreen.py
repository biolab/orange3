"""
Orange Canvas Splash Screen
"""

from PyQt4.QtGui import (
    QSplashScreen,  QWidget, QPixmap, QPainter, QTextDocument,
    QTextBlockFormat, QTextCursor, QApplication
)
from PyQt4.QtCore import Qt, qVersion

from .utils import is_transparency_supported


class SplashScreen(QSplashScreen):
    def __init__(self, parent=None, pixmap=None, textRect=None, **kwargs):
        QSplashScreen.__init__(self, parent, **kwargs)
        self.__textRect = textRect
        self.__message = ""
        self.__color = Qt.black
        self.__alignment = Qt.AlignLeft

        if pixmap is None:
            pixmap = QPixmap()

        self.setPixmap(pixmap)

        self.setAutoFillBackground(False)
        # Also set FramelesWindowHint (if not already set)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def setTextRect(self, rect):
        """Set the rectangle in which to show the message text.
        """
        if self.__textRect != rect:
            self.__textRect = rect
            self.update()

    def textRect(self):
        """Return the text rect.
        """
        return self.__textRect

    def showEvent(self, event):
        QSplashScreen.showEvent(self, event)
        # Raise to top on show.
        self.raise_()

    def drawContents(self, painter):
        """Reimplementation of drawContents to limit the drawing
        inside `textRext`.

        """
        painter.setPen(self.__color)
        painter.setFont(self.font())

        if self.__textRect:
            rect = self.__textRect
        else:
            rect = self.rect().adjusted(5, 5, -5, -5)
        if Qt.mightBeRichText(self.__message):
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
        # Need to store all this arguments for drawContents (no access
        # methods)
        self.__alignment = alignment
        self.__color = color
        self.__message = message
        QSplashScreen.showMessage(self, message, alignment, color)
        QApplication.instance().processEvents()

    if qVersion() < "4.8":
        # in 4.7 the splash screen does not support transparency
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
