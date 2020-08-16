"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Line numbers and bookmarks areas
"""

from PyQt5.QtCore import QPoint, Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtGui import QPainter, QPalette, QPixmap, QTextBlock

import qutepart
from qutepart.bookmarks import Bookmarks
from qutepart.margins import MarginBase



# Dynamic mixin at runtime:
# http://stackoverflow.com/questions/8544983/dynamically-mixin-a-base-class-to-an-instance-in-python
def extend_instance(obj, cls):
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls), {})



class LineNumberArea(QWidget):
    """Line number area widget
    """
    _LEFT_MARGIN = 5
    _RIGHT_MARGIN = 3

    def __init__(self, parent):
        QWidget.__init__(self, parent)

        extend_instance(self, MarginBase)
        MarginBase.__init__(self, parent, "line_numbers", 0)

        self.__width = self.__calculateWidth()

        self._qpart.blockCountChanged.connect(self.__updateWidth)

    def __updateWidth(self, newBlockCount=None):
        newWidth = self.__calculateWidth()
        if newWidth != self.__width:
            self.__width = newWidth
            self._qpart.updateViewport()

    def paintEvent(self, event):
        """QWidget.paintEvent() implementation
        """
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.palette().color(QPalette.Window))
        painter.setPen(Qt.black)

        block = self._qpart.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self._qpart.blockBoundingGeometry(block).translated(self._qpart.contentOffset()).top())
        bottom = top + int(self._qpart.blockBoundingRect(block).height())
        singleBlockHeight = self._qpart.cursorRect().height()

        boundingRect = self._qpart.blockBoundingRect(block)
        availableWidth = self.__width - self._RIGHT_MARGIN - self._LEFT_MARGIN
        availableHeight = self._qpart.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.drawText(self._LEFT_MARGIN, top,
                                 availableWidth, availableHeight,
                                 Qt.AlignRight, number)
                if boundingRect.height() >= singleBlockHeight * 2:  # wrapped block
                    painter.fillRect(1, top + singleBlockHeight,
                                     self.__width - 2, boundingRect.height() - singleBlockHeight - 2,
                                     Qt.darkGreen)

            block = block.next()
            boundingRect = self._qpart.blockBoundingRect(block)
            top = bottom
            bottom = top + int(boundingRect.height())
            blockNumber += 1

    def __calculateWidth(self):
        digits = len(str(max(1, self._qpart.blockCount())))
        return self._LEFT_MARGIN + self._qpart.fontMetrics().width('9') * digits + self._RIGHT_MARGIN

    def width(self):
        """Desired width. Includes text and margins
        """
        return self.__width

    def setFont(self, font):
        QWidget.setFont(self, font)
        self.__updateWidth()


class MarkArea(QWidget):

    _MARGIN = 1

    def __init__(self, qpart):
        QWidget.__init__(self, qpart)

        extend_instance(self, MarginBase)
        MarginBase.__init__(self, qpart, "mark_area", 1)

        qpart.blockCountChanged.connect(self.update)

        self.setMouseTracking(True)

        self._bookmarkPixmap = self._loadIcon('emblem-favorite')
        self._lintPixmaps = {qpart.LINT_ERROR: self._loadIcon('emblem-error'),
                             qpart.LINT_WARNING: self._loadIcon('emblem-warning'),
                             qpart.LINT_NOTE: self._loadIcon('emblem-information')}

        self._bookmarks = Bookmarks(qpart, self)

    def _loadIcon(self, fileName):
        icon = qutepart.getIcon(fileName)
        size = self._qpart.cursorRect().height() - 6
        pixmap = icon.pixmap(size, size)  # This also works with Qt.AA_UseHighDpiPixmaps
        return pixmap

    def sizeHint(self, ):
        """QWidget.sizeHint() implementation
        """
        return QSize(self.width(), 0)

    def paintEvent(self, event):
        """QWidget.paintEvent() implementation
        Draw markers
        """
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.palette().color(QPalette.Window))

        block = self._qpart.firstVisibleBlock()
        blockBoundingGeometry = self._qpart.blockBoundingGeometry(block).translated(self._qpart.contentOffset())
        top = blockBoundingGeometry.top()
        bottom = top + blockBoundingGeometry.height()

        for block in qutepart.iterateBlocksFrom(block):
            height = self._qpart.blockBoundingGeometry(block).height()
            if top > event.rect().bottom():
                break
            if block.isVisible() and \
               bottom >= event.rect().top():
                if block.blockNumber() in self._qpart.lintMarks:
                    msgType, msgText = self._qpart.lintMarks[block.blockNumber()]
                    pixMap = self._lintPixmaps[msgType]
                    yPos = top + ((height - pixMap.height()) / 2)  # centered
                    painter.drawPixmap(0, yPos, pixMap)

                if self.isBlockMarked(block):
                    yPos = top + ((height - self._bookmarkPixmap.height()) / 2)  # centered
                    painter.drawPixmap(0, yPos, self._bookmarkPixmap)

            top += height

    def width(self):
        """Desired width. Includes text and margins
        """
        return self._MARGIN + self._bookmarkPixmap.width() + self._MARGIN

    def mouseMoveEvent(self, event):
        blockNumber = self._qpart.cursorForPosition(event.pos()).blockNumber()
        if blockNumber in self._qpart._lintMarks:
            msgType, msgText = self._qpart._lintMarks[blockNumber]
            QToolTip.showText(event.globalPos(), msgText)
        else:
            QToolTip.hideText()

        return QWidget.mouseMoveEvent(self, event)

    def clearBookmarks(self, startBlock, endBlock):
        """Clears the bookmarks
        """
        self._bookmarks.clear(startBlock, endBlock)

    def clear(self):
        self._bookmarks.removeActions()
        MarginBase.clear(self)

