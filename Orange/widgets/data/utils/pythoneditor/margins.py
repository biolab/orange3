"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Base class for margins
"""


from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QTextBlock


class MarginBase:
    """Base class which each margin should derive from
    """

    # The parent class derives from QWidget and mixes MarginBase in at
    # run-time. Thus the signal declaration and emmitting works here too.
    blockClicked = pyqtSignal(QTextBlock)

    def __init__(self, parent, name, bit_count):
        """qpart: reference to the editor
           name: margin identifier
           bit_count: number of bits to be used by the margin
        """
        self._qpart = parent
        self._name = name
        self._bit_count = bit_count
        self._bitRange = None
        self.__allocateBits()

        self._countCache = (-1, -1)
        self._qpart.updateRequest.connect(self.__updateRequest)

    def __allocateBits(self):
        """Allocates the bit range depending on the required bit count
        """
        if self._bit_count < 0:
            raise Exception( "A margin cannot request negative number of bits" )
        if self._bit_count == 0:
            return

        # Build a list of occupied ranges
        margins = self._qpart.getMargins()

        occupiedRanges = []
        for margin in margins:
            bitRange = margin.getBitRange()
            if bitRange is not None:
                # pick the right position
                added = False
                for index in range( len( occupiedRanges ) ):
                    r = occupiedRanges[ index ]
                    if bitRange[ 1 ] < r[ 0 ]:
                        occupiedRanges.insert(index, bitRange)
                        added = True
                        break
                if not added:
                    occupiedRanges.append(bitRange)

        vacant = 0
        for r in occupiedRanges:
            if r[ 0 ] - vacant >= self._bit_count:
                self._bitRange = (vacant, vacant + self._bit_count - 1)
                return
            vacant = r[ 1 ] + 1
        # Not allocated, i.e. grab the tail bits
        self._bitRange = (vacant, vacant + self._bit_count - 1)

    def __updateRequest(self, rect, dy):
        """Repaint line number area if necessary
        """
        if dy:
            self.scroll(0, dy)
        elif self._countCache[0] != self._qpart.blockCount() or \
             self._countCache[1] != self._qpart.textCursor().block().lineCount():

            # if block height not added to rect, last line number sometimes is not drawn
            blockHeight = self._qpart.blockBoundingRect(self._qpart.firstVisibleBlock()).height()

            self.update(0, rect.y(), self.width(), rect.height() + blockHeight)
            self._countCache = (self._qpart.blockCount(), self._qpart.textCursor().block().lineCount())

        if rect.contains(self._qpart.viewport().rect()):
            self._qpart.updateViewportMargins()

    def getName(self):
        """Provides the margin identifier
        """
        return self._name

    def getBitRange(self):
        """None or inclusive bits used pair,
           e.g. (2,4) => 3 bits used 2nd, 3rd and 4th
        """
        return self._bitRange

    def setBlockValue(self, block, value):
        """Sets the required value to the block without damaging the other bits
        """
        if self._bit_count == 0:
            raise Exception( "The margin '" + self._name +
                             "' did not allocate any bits for the values")
        if value < 0:
            raise Exception( "The margin '" + self._name +
                             "' must be a positive integer"  )

        if value >= 2 ** self._bit_count:
            raise Exception( "The margin '" + self._name +
                             "' value exceeds the allocated bit range" )

        newMarginValue = value << self._bitRange[ 0 ]
        currentUserState = block.userState()

        if currentUserState in [ 0, -1 ]:
            block.setUserState(newMarginValue)
        else:
            marginMask = 2 ** self._bit_count - 1
            otherMarginsValue = currentUserState & ~marginMask
            block.setUserState(newMarginValue | otherMarginsValue)

    def getBlockValue(self, block):
        """Provides the previously set block value respecting the bits range.
           0 value and not marked block are treated the same way and 0 is
           provided.
        """
        if self._bit_count == 0:
            raise Exception( "The margin '" + self._name +
                             "' did not allocate any bits for the values")
        val = block.userState()
        if val in [ 0, -1 ]:
            return 0

        # Shift the value to the right
        val >>= self._bitRange[ 0 ]

        # Apply the mask to the value
        mask = 2 ** self._bit_count - 1
        val &= mask
        return val

    def hide(self):
        """Override the QWidget::hide() method to properly recalculate the
           editor viewport.
        """
        if not self.isHidden():
            QWidget.hide(self)
            self._qpart.updateViewport()

    def show(self):
        """Override the QWidget::show() method to properly recalculate the
           editor viewport.
        """
        if self.isHidden():
            QWidget.show(self)
            self._qpart.updateViewport()

    def setVisible(self, val):
        """Override the QWidget::setVisible(bool) method to properly
           recalculate the editor viewport.
        """
        if val != self.isVisible():
            if val:
                QWidget.setVisible(self, True)
            else:
                QWidget.setVisible(self, False)
            self._qpart.updateViewport()

    def mousePressEvent(self, mouseEvent):
        cursor = self._qpart.cursorForPosition(QPoint(0, mouseEvent.y()))
        block = cursor.block()
        blockRect = self._qpart.blockBoundingGeometry(block).translated(self._qpart.contentOffset())
        if blockRect.bottom() >= mouseEvent.y():  # clicked not lower, then end of text
            self.blockClicked.emit(block)

    # Convenience methods

    def clear(self):
        """Convenience method to reset all the block values to 0
        """
        if self._bit_count == 0:
            return

        block = self._qpart.document().begin()
        while block.isValid():
            if self.getBlockValue(block):
                self.setBlockValue(block, 0)
            block = block.next()

    # Methods for 1-bit margins
    def isBlockMarked(self, block):
        return self.getBlockValue(block) != 0
    def toggleBlockMark(self, block):
        self.setBlockValue(block, 0 if self.isBlockMarked(block) else 1)

