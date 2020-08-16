"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""QSyntaxHighlighter implementation
Uses syntax module for doing the job
"""

import time

from PyQt5.QtCore import QObject, QTimer, pyqtSlot
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QTextBlockUserData, QTextLayout

import qutepart.syntax


def _cmpFormatRanges(a, b):
    """PyQt does not define proper comparison for QTextLayout.FormatRange
    Define it to check correctly, if formats has changed.
    It is important for the performance
    """
    if a.format == b.format and \
       a.start == b.start and \
       a.length == b.length:
        return 0
    else:
        return cmp(id(a), id(b))


def _formatRangeListsEqual(a, b):
    if len(a) != len(b):
        return False

    for a_item, b_item in zip(a, b):
        if a_item != b_item:
            return False

    return True


class _TextBlockUserData(QTextBlockUserData):
    def __init__(self, data):
        QTextBlockUserData.__init__(self)
        self.data = data


class GlobalTimer:
    """All parsing and highlighting is done in main loop thread.
    If parsing is being done for long time, main loop gets blocked.
    Therefore SyntaxHighlighter controls, how long parsign is going, and, if too long,
    schedules timer and releases main loop.
    One global timer is used by all Qutepart instances, because main loop time usage
    must not depend on opened files count
    """

    def __init__(self):
        self._timer = QTimer(QApplication.instance())
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._onTimer)

        self._scheduledCallbacks = []

    def isActive(self):
        return self._timer.isActive()

    def scheduleCallback(self, callback):
        if not callback in self._scheduledCallbacks:
            self._scheduledCallbacks.append(callback)
            self._timer.start()

    def unScheduleCallback(self, callback):
        if callback in self._scheduledCallbacks:
            self._scheduledCallbacks.remove(callback)

        if not self._scheduledCallbacks:
            self._timer.stop()

    def isCallbackScheduled(self, callback):
        return callback in self._scheduledCallbacks

    def _onTimer(self):
        if self._scheduledCallbacks:
            callback = self._scheduledCallbacks.pop()
            callback()
        if self._scheduledCallbacks:
            self._timer.start()


"""Global var, because main loop time usage shall not depend on Qutepart instances count

Pyside crashes, if this variable is a class field
"""
_gLastChangeTime = -777.


class SyntaxHighlighter(QObject):

    # when initially parsing text, it is better, if highlighted text is drawn without flickering
    _MAX_PARSING_TIME_BIG_CHANGE_SEC = 0.4
    # when user is typing text - response shall be quick
    _MAX_PARSING_TIME_SMALL_CHANGE_SEC = 0.02

    _globalTimer = GlobalTimer()

    def __init__(self, syntax, textEdit):
        QObject.__init__(self, textEdit.document())

        self._syntax = syntax
        self._textEdit = textEdit
        self._document = textEdit.document()

        # can't store references to block, Qt crashes if block removed
        self._pendingBlockNumber = None
        self._pendingAtLeastUntilBlockNumber = None

        self._document.contentsChange.connect(self._onContentsChange)

        charsAdded = self._document.lastBlock().position() + self._document.lastBlock().length()
        self._onContentsChange(0, 0, charsAdded, zeroTimeout=self._wasChangedJustBefore())

    def terminate(self):
        try:
            self._document.contentsChange.disconnect(self._onContentsChange)
        except TypeError:
            pass

        self._globalTimer.unScheduleCallback(self._onContinueHighlighting)
        block = self._document.firstBlock()
        while block.isValid():
            block.layout().setAdditionalFormats([])
            block.setUserData(None)
            self._document.markContentsDirty(block.position(), block.length())
            block = block.next()
        self._globalTimer.unScheduleCallback(self._onContinueHighlighting)

    def syntax(self):
        """Return own syntax
        """
        return self._syntax

    def isInProgress(self):
        """Highlighting is in progress
        """
        return self._globalTimer.isCallbackScheduled(self._onContinueHighlighting)

    def isCode(self, block, column):
        """Check if character at column is a a code
        """
        dataObject = block.userData()
        data = dataObject.data if dataObject is not None else None
        return self._syntax.isCode(data, column)

    def isComment(self, block, column):
        """Check if character at column is a comment
        """
        dataObject = block.userData()
        data = dataObject.data if dataObject is not None else None
        return self._syntax.isComment(data, column)

    def isBlockComment(self, block, column):
        """Check if character at column is a block comment
        """
        dataObject = block.userData()
        data = dataObject.data if dataObject is not None else None
        return self._syntax.isBlockComment(data, column)

    def isHereDoc(self, block, column):
        """Check if character at column is a here document
        """
        dataObject = block.userData()
        data = dataObject.data if dataObject is not None else None
        return self._syntax.isHereDoc(data, column)

    @staticmethod
    def _lineData(block):
        dataObject = block.userData()
        if dataObject is not None:
            return dataObject.data
        else:
            return None

    def _wasChangedJustBefore(self):
        """Check if ANY Qutepart instance was changed just before"""
        return time.time() <= _gLastChangeTime + 1

    @pyqtSlot(int, int, int)
    def _onContentsChange(self, from_, charsRemoved, charsAdded, zeroTimeout=False):
        global _gLastChangeTime
        firstBlock = self._document.findBlock(from_)
        untilBlock = self._document.findBlock(from_ + charsAdded)

        if self._globalTimer.isCallbackScheduled(self._onContinueHighlighting):  # have not finished task.
            """ Intersect ranges. Might produce a lot of extra highlighting work
            More complicated algorithm might be invented later
            """
            if self._pendingBlockNumber < firstBlock.blockNumber():
                firstBlock = self._document.findBlockByNumber(self._pendingBlockNumber)
            if self._pendingAtLeastUntilBlockNumber > untilBlock.blockNumber():
                untilBlockNumber = min(self._pendingAtLeastUntilBlockNumber,
                                       self._document.blockCount() - 1)
                untilBlock = self._document.findBlockByNumber(untilBlockNumber)
            self._globalTimer.unScheduleCallback(self._onContinueHighlighting)

        if zeroTimeout:
            timeout = 0  # no parsing, only schedule
        elif charsAdded > 20 and \
             (not self._wasChangedJustBefore()):
            """Use big timeout, if change is really big and previous big change was long time ago"""
            timeout = self._MAX_PARSING_TIME_BIG_CHANGE_SEC
        else:
            timeout = self._MAX_PARSING_TIME_SMALL_CHANGE_SEC

        _gLastChangeTime = time.time()

        self._highlighBlocks(firstBlock, untilBlock, timeout)

    def _onContinueHighlighting(self):
        self._highlighBlocks(self._document.findBlockByNumber(self._pendingBlockNumber),
                             self._document.findBlockByNumber(self._pendingAtLeastUntilBlockNumber),
                             self._MAX_PARSING_TIME_SMALL_CHANGE_SEC)

    def _highlighBlocks(self, fromBlock, atLeastUntilBlock, timeout):
        endTime = time.time() + timeout

        block = fromBlock
        lineData = self._lineData(block.previous())

        while block.isValid() and block != atLeastUntilBlock:
            if time.time() >= endTime:  # time is over, schedule parsing later and release event loop
                self._pendingBlockNumber = block.blockNumber()
                self._pendingAtLeastUntilBlockNumber = atLeastUntilBlock.blockNumber()
                self._globalTimer.scheduleCallback(self._onContinueHighlighting)
                return

            contextStack = lineData[0] if lineData is not None else None
            if block.length() < 4096:
                lineData, highlightedSegments = self._syntax.highlightBlock(block.text(), contextStack)
            else:
                """Parser freezes for a long time, if line is too long
                invalid parsing results are still better, than freeze
                """
                lineData, highlightedSegments = None, []
            if lineData is not None:
                block.setUserData(_TextBlockUserData(lineData))
            else:
                block.setUserData(None)

            self._applyHighlightedSegments(block, highlightedSegments)
            block = block.next()

        # reached atLeastUntilBlock, now parse next only while data changed
        prevLineData = self._lineData(block)
        while block.isValid():
            if time.time() >= endTime:  # time is over, schedule parsing later and release event loop
                self._pendingBlockNumber = block.blockNumber()
                self._pendingAtLeastUntilBlockNumber = atLeastUntilBlock.blockNumber()
                self._globalTimer.scheduleCallback(self._onContinueHighlighting)
                return
            contextStack = lineData[0] if lineData is not None else None
            lineData, highlightedSegments = self._syntax.highlightBlock(block.text(), contextStack)
            if lineData is not None:
                block.setUserData(_TextBlockUserData(lineData))
            else:
                block.setUserData(None)

            self._applyHighlightedSegments(block, highlightedSegments)
            if prevLineData == lineData:
                break

            block = block.next()
            prevLineData = self._lineData(block)

        # sucessfully finished, reset pending tasks
        self._pendingBlockNumber = None
        self._pendingAtLeastUntilBlockNumber = None

        """Emit sizeChanged when highlighting finished, because document size might change.
        See andreikop/enki issue #191
        """
        documentLayout = self._textEdit.document().documentLayout()
        documentLayout.documentSizeChanged.emit(documentLayout.documentSize())

    def _applyHighlightedSegments(self, block, highlightedSegments):
        ranges = []
        currentPos = 0

        for length, format in highlightedSegments:
            if format is not None:  # might be in incorrect syntax file
                range = QTextLayout.FormatRange()
                range.format = format
                range.start = currentPos
                range.length = length
                ranges.append(range)
            currentPos += length

        if not _formatRangeListsEqual(block.layout().additionalFormats(), ranges):
            block.layout().setAdditionalFormats(ranges)
            self._document.markContentsDirty(block.position(), block.length())
