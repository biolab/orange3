"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import time

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QTextCursor, QColor
from AnyQt.QtWidgets import QTextEdit, QApplication

# Bracket highlighter.
# Calculates list of QTextEdit.ExtraSelection


class _TimeoutException(UserWarning):
    """Operation timeout happened
    """


class BracketHighlighter:
    """Bracket highliter.
    Calculates list of QTextEdit.ExtraSelection

    Currently, this class might be just a set of functions.
    Probably, it will contain instance specific selection colors later
    """
    MATCHED_COLOR = QColor('#0b0')
    UNMATCHED_COLOR = QColor('#a22')

    _MAX_SEARCH_TIME_SEC = 0.02

    _START_BRACKETS = '({['
    _END_BRACKETS = ')}]'
    _ALL_BRACKETS = _START_BRACKETS + _END_BRACKETS
    _OPOSITE_BRACKET = dict(zip(_START_BRACKETS + _END_BRACKETS, _END_BRACKETS + _START_BRACKETS))

    # instance variable. None or ((block, columnIndex), (block, columnIndex))
    currentMatchedBrackets = None

    def _iterateDocumentCharsForward(self, block, startColumnIndex):
        """Traverse document forward. Yield (block, columnIndex, char)
        Raise _TimeoutException if time is over
        """
        # Chars in the start line
        endTime = time.time() + self._MAX_SEARCH_TIME_SEC
        for columnIndex, char in list(enumerate(block.text()))[startColumnIndex:]:
            yield block, columnIndex, char
        block = block.next()

        # Next lines
        while block.isValid():
            for columnIndex, char in enumerate(block.text()):
                yield block, columnIndex, char

            if time.time() > endTime:
                raise _TimeoutException('Time is over')

            block = block.next()

    def _iterateDocumentCharsBackward(self, block, startColumnIndex):
        """Traverse document forward. Yield (block, columnIndex, char)
        Raise _TimeoutException if time is over
        """
        # Chars in the start line
        endTime = time.time() + self._MAX_SEARCH_TIME_SEC
        for columnIndex, char in reversed(list(enumerate(block.text()[:startColumnIndex]))):
            yield block, columnIndex, char
        block = block.previous()

        # Next lines
        while block.isValid():
            for columnIndex, char in reversed(list(enumerate(block.text()))):
                yield block, columnIndex, char

            if time.time() > endTime:
                raise _TimeoutException('Time is over')

            block = block.previous()

    def _findMatchingBracket(self, bracket, qpart, block, columnIndex):
        """Find matching bracket for the bracket.
        Return (block, columnIndex) or (None, None)
        Raise _TimeoutException, if time is over
        """
        if bracket in self._START_BRACKETS:
            charsGenerator = self._iterateDocumentCharsForward(block, columnIndex + 1)
        else:
            charsGenerator = self._iterateDocumentCharsBackward(block, columnIndex)

        depth = 1
        oposite = self._OPOSITE_BRACKET[bracket]
        for b, c_index, char in charsGenerator:
            if qpart.isCode(b, c_index):
                if char == oposite:
                    depth -= 1
                    if depth == 0:
                        return b, c_index
                elif char == bracket:
                    depth += 1
        return None, None

    def _makeMatchSelection(self, block, columnIndex, matched):
        """Make matched or unmatched QTextEdit.ExtraSelection
        """
        selection = QTextEdit.ExtraSelection()
        darkMode = QApplication.instance().property('darkMode')

        if matched:
            fgColor = self.MATCHED_COLOR
        else:
            fgColor = self.UNMATCHED_COLOR

        selection.format.setForeground(fgColor)
        # repaint hack
        selection.format.setBackground(Qt.white if not darkMode else QColor('#111111'))
        selection.cursor = QTextCursor(block)
        selection.cursor.setPosition(block.position() + columnIndex)
        selection.cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)

        return selection

    def _highlightBracket(self, bracket, qpart, block, columnIndex):
        """Highlight bracket and matching bracket
        Return tuple of QTextEdit.ExtraSelection's
        """
        try:
            matchedBlock, matchedColumnIndex = self._findMatchingBracket(bracket, qpart,
                                                                         block, columnIndex)
        except _TimeoutException:  # not found, time is over
            return[] # highlight nothing

        if matchedBlock is not None:
            self.currentMatchedBrackets = ((block, columnIndex), (matchedBlock, matchedColumnIndex))
            return [self._makeMatchSelection(block, columnIndex, True),
                    self._makeMatchSelection(matchedBlock, matchedColumnIndex, True)]
        else:
            self.currentMatchedBrackets = None
            return [self._makeMatchSelection(block, columnIndex, False)]

    def extraSelections(self, qpart, block, columnIndex):
        """List of QTextEdit.ExtraSelection's, which highlighte brackets
        """
        blockText = block.text()

        if columnIndex < len(blockText) and \
             blockText[columnIndex] in self._ALL_BRACKETS and \
             qpart.isCode(block, columnIndex):
            return self._highlightBracket(blockText[columnIndex], qpart, block, columnIndex)
        elif columnIndex > 0 and \
           blockText[columnIndex - 1] in self._ALL_BRACKETS and \
           qpart.isCode(block, columnIndex - 1):
            return self._highlightBracket(blockText[columnIndex - 1], qpart, block, columnIndex - 1)
        else:
            self.currentMatchedBrackets = None
            return []
