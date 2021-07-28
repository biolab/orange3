"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import sys

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QColor, QTextCursor

# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches

# This magic code sets variables like _a and _A in the global scope
# pylint: disable=undefined-variable
thismodule = sys.modules[__name__]
for charCode in range(ord('a'), ord('z') + 1):
    shortName = chr(charCode)
    longName = 'Key_' + shortName.upper()
    qtCode = getattr(Qt, longName)
    setattr(thismodule, '_' + shortName, qtCode)
    setattr(thismodule, '_' + shortName.upper(), Qt.ShiftModifier + qtCode)

_0 = Qt.Key_0
_Dollar = Qt.ShiftModifier + Qt.Key_Dollar
_Percent = Qt.ShiftModifier + Qt.Key_Percent
_Caret = Qt.ShiftModifier + Qt.Key_AsciiCircum
_Esc = Qt.Key_Escape
_Insert = Qt.Key_Insert
_Down = Qt.Key_Down
_Up = Qt.Key_Up
_Left = Qt.Key_Left
_Right = Qt.Key_Right
_Space = Qt.Key_Space
_BackSpace = Qt.Key_Backspace
_Equal = Qt.Key_Equal
_Less = Qt.ShiftModifier + Qt.Key_Less
_Greater = Qt.ShiftModifier + Qt.Key_Greater
_Home = Qt.Key_Home
_End = Qt.Key_End
_PageDown = Qt.Key_PageDown
_PageUp = Qt.Key_PageUp
_Period = Qt.Key_Period
_Enter = Qt.Key_Enter
_Return = Qt.Key_Return


def code(ev):
    modifiers = ev.modifiers()
    modifiers &= ~Qt.KeypadModifier  # ignore keypad modifier to handle both main and numpad numbers
    return int(modifiers) + ev.key()


def isChar(ev):
    """ Check if an event may be a typed character
    """
    text = ev.text()
    if len(text) != 1:
        return False

    if ev.modifiers() not in (Qt.ShiftModifier, Qt.KeypadModifier, Qt.NoModifier):
        return False

    asciiCode = ord(text)
    if asciiCode <= 31 or asciiCode == 0x7f:  # control characters
        return False

    if text == ' ' and ev.modifiers() == Qt.ShiftModifier:
        return False  # Shift+Space is a shortcut, not a text

    return True


NORMAL = 'normal'
INSERT = 'insert'
REPLACE_CHAR = 'replace character'

MODE_COLORS = {NORMAL: QColor('#33cc33'),
               INSERT: QColor('#ff9900'),
               REPLACE_CHAR: QColor('#ff3300')}


class _GlobalClipboard:
    def __init__(self):
        self.value = ''


_globalClipboard = _GlobalClipboard()


class Vim(QObject):
    """Vim mode implementation.
    Listens events and does actions
    """
    modeIndicationChanged = pyqtSignal(QColor, str)

    def __init__(self, qpart):
        QObject.__init__(self)
        self._qpart = qpart
        self._mode = Normal(self, qpart)

        self._qpart.selectionChanged.connect(self._onSelectionChanged)
        self._qpart.document().modificationChanged.connect(self._onModificationChanged)

        self._processingKeyPress = False

        self.updateIndication()

        self.lastEditCmdFunc = None

    def terminate(self):
        self._qpart.selectionChanged.disconnect(self._onSelectionChanged)
        try:
            self._qpart.document().modificationChanged.disconnect(self._onModificationChanged)
        except TypeError:
            pass

    def indication(self):
        return self._mode.color, self._mode.text()

    def updateIndication(self):
        self.modeIndicationChanged.emit(*self.indication())

    def keyPressEvent(self, ev):
        """Check the event. Return True if processed and False otherwise
        """
        if ev.key() in (Qt.Key_Shift, Qt.Key_Control,
                        Qt.Key_Meta, Qt.Key_Alt,
                        Qt.Key_AltGr, Qt.Key_CapsLock,
                        Qt.Key_NumLock, Qt.Key_ScrollLock):
            return False  # ignore modifier pressing. Will process key pressing later

        self._processingKeyPress = True
        try:
            ret = self._mode.keyPressEvent(ev)
        finally:
            self._processingKeyPress = False
        return ret

    def inInsertMode(self):
        return isinstance(self._mode, Insert)

    def mode(self):
        return self._mode

    def setMode(self, mode):
        self._mode = mode

        self._qpart._updateVimExtraSelections()

        self.updateIndication()

    def extraSelections(self):
        """ In normal mode - QTextEdit.ExtraSelection which highlightes the cursor
        """
        if not isinstance(self._mode, Normal):
            return []

        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor('#ffcc22'))
        selection.format.setForeground(QColor('#000000'))
        selection.cursor = self._qpart.textCursor()
        selection.cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)

        return [selection]

    def _onSelectionChanged(self):
        if not self._processingKeyPress:
            if self._qpart.selectedText:
                if not isinstance(self._mode, (Visual, VisualLines)):
                    self.setMode(Visual(self, self._qpart))
            else:
                self.setMode(Normal(self, self._qpart))

    def _onModificationChanged(self, modified):
        if not modified and isinstance(self._mode, Insert):
            self.setMode(Normal(self, self._qpart))


class Mode:
    # pylint: disable=no-self-use
    color = None

    def __init__(self, vim, qpart):
        self._vim = vim
        self._qpart = qpart

    def text(self):
        return None

    def keyPressEvent(self, ev):
        pass

    def switchMode(self, modeClass, *args):
        mode = modeClass(self._vim, self._qpart, *args)
        self._vim.setMode(mode)

    def switchModeAndProcess(self, text, modeClass, *args):
        mode = modeClass(self._vim, self._qpart, *args)
        self._vim.setMode(mode)
        return mode.keyPressEvent(text)


class Insert(Mode):
    color = QColor('#ff9900')

    def text(self):
        return 'insert'

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.switchMode(Normal)
            return True

        return False


class ReplaceChar(Mode):
    color = QColor('#ee7777')

    def text(self):
        return 'replace char'

    def keyPressEvent(self, ev):
        if isChar(ev):  # a char
            self._qpart.setOverwriteMode(False)
            line, col = self._qpart.cursorPosition
            if col > 0:
                # return the cursor back after replacement
                self._qpart.cursorPosition = (line, col - 1)
            self.switchMode(Normal)
            return True
        else:
            self._qpart.setOverwriteMode(False)
            self.switchMode(Normal)
            return False


class Replace(Mode):
    color = QColor('#ee7777')

    def text(self):
        return 'replace'

    def keyPressEvent(self, ev):
        if ev.key() == _Insert:
            self._qpart.setOverwriteMode(False)
            self.switchMode(Insert)
            return True
        elif ev.key() == _Esc:
            self._qpart.setOverwriteMode(False)
            self.switchMode(Normal)
            return True
        else:
            return False


class BaseCommandMode(Mode):
    """ Base class for Normal and Visual modes
    """

    def __init__(self, *args):
        Mode.__init__(self, *args)
        self._reset()

    def keyPressEvent(self, ev):
        self._typedText += ev.text()
        try:
            self._processCharCoroutine.send(ev)
        except StopIteration as ex:
            retVal = ex.value
            self._reset()
        else:
            retVal = True

        self._vim.updateIndication()

        return retVal

    def text(self):
        return self._typedText or self.name

    def _reset(self):
        self._processCharCoroutine = self._processChar()
        next(self._processCharCoroutine)  # run until the first yield
        self._typedText = ''

    _MOTIONS = (_0, _Home,
                _Dollar, _End,
                _Percent, _Caret,
                _b, _B,
                _e, _E,
                _G,
                _j, _Down,
                _l, _Right, _Space,
                _k, _Up,
                _h, _Left, _BackSpace,
                _w, _W,
                'gg',
                _f, _F, _t, _T,
                _PageDown, _PageUp,
                _Enter, _Return,
                )

    @staticmethod
    def moveToFirstNonSpace(cursor, moveMode):
        text = cursor.block().text()
        spaceLen = len(text) - len(text.lstrip())
        cursor.setPosition(cursor.block().position() + spaceLen, moveMode)

    def _moveCursor(self, motion, count, searchChar=None, select=False):
        """ Move cursor.
        Used by Normal and Visual mode
        """
        cursor = self._qpart.textCursor()

        effectiveCount = count or 1

        moveMode = QTextCursor.KeepAnchor if select else QTextCursor.MoveAnchor

        moveOperation = {_b: QTextCursor.WordLeft,
                         _j: QTextCursor.Down,
                         _Down: QTextCursor.Down,
                         _k: QTextCursor.Up,
                         _Up: QTextCursor.Up,
                         _h: QTextCursor.Left,
                         _Left: QTextCursor.Left,
                         _BackSpace: QTextCursor.Left,
                         _l: QTextCursor.Right,
                         _Right: QTextCursor.Right,
                         _Space: QTextCursor.Right,
                         _w: QTextCursor.WordRight,
                         _Dollar: QTextCursor.EndOfBlock,
                         _End: QTextCursor.EndOfBlock,
                         _0: QTextCursor.StartOfBlock,
                         _Home: QTextCursor.StartOfBlock,
                         'gg': QTextCursor.Start,
                         _G: QTextCursor.End
                         }

        if motion == _G:
            if count == 0:  # default - go to the end
                cursor.movePosition(QTextCursor.End, moveMode)
            else:  # if count is set - move to line
                block = self._qpart.document().findBlockByNumber(count - 1)
                if not block.isValid():
                    return
                cursor.setPosition(block.position(), moveMode)
            self.moveToFirstNonSpace(cursor, moveMode)
        elif motion in moveOperation:
            for _ in range(effectiveCount):
                cursor.movePosition(moveOperation[motion], moveMode)
        elif motion in (_e, _E):
            for _ in range(effectiveCount):
                # skip spaces
                text = cursor.block().text()
                pos = cursor.positionInBlock()
                for char in text[pos:]:
                    if char.isspace():
                        cursor.movePosition(QTextCursor.NextCharacter, moveMode)
                    else:
                        break

                if cursor.positionInBlock() == len(text):  # at the end of line
                    # move to the next line
                    cursor.movePosition(QTextCursor.NextCharacter, moveMode)

                    # now move to the end of word
                if motion == _e:
                    cursor.movePosition(QTextCursor.EndOfWord, moveMode)
                else:
                    text = cursor.block().text()
                    pos = cursor.positionInBlock()
                    for char in text[pos:]:
                        if not char.isspace():
                            cursor.movePosition(QTextCursor.NextCharacter, moveMode)
                        else:
                            break
        elif motion == _B:
            cursor.movePosition(QTextCursor.WordLeft, moveMode)
            while cursor.positionInBlock() != 0 and \
                    (not cursor.block().text()[cursor.positionInBlock() - 1].isspace()):
                cursor.movePosition(QTextCursor.WordLeft, moveMode)
        elif motion == _W:
            cursor.movePosition(QTextCursor.WordRight, moveMode)
            while cursor.positionInBlock() != 0 and \
                    (not cursor.block().text()[cursor.positionInBlock() - 1].isspace()):
                cursor.movePosition(QTextCursor.WordRight, moveMode)
        elif motion == _Percent:
            # Percent move is done only once
            if self._qpart._bracketHighlighter.currentMatchedBrackets is not None:
                ((startBlock, startCol), (endBlock, endCol)) = \
                    self._qpart._bracketHighlighter.currentMatchedBrackets
                startPos = startBlock.position() + startCol
                endPos = endBlock.position() + endCol
                if select and \
                        (endPos > startPos):
                    endPos += 1  # to select the bracket, not only chars before it
                cursor.setPosition(endPos, moveMode)
        elif motion == _Caret:
            # Caret move is done only once
            self.moveToFirstNonSpace(cursor, moveMode)
        elif motion in (_f, _F, _t, _T):
            if motion in (_f, _t):
                iterator = self._iterateDocumentCharsForward(cursor.block(), cursor.columnNumber())
                stepForward = QTextCursor.Right
                stepBack = QTextCursor.Left
            else:
                iterator = self._iterateDocumentCharsBackward(cursor.block(), cursor.columnNumber())
                stepForward = QTextCursor.Left
                stepBack = QTextCursor.Right

            for block, columnIndex, char in iterator:
                if char == searchChar:
                    cursor.setPosition(block.position() + columnIndex, moveMode)
                    if motion in (_t, _T):
                        cursor.movePosition(stepBack, moveMode)
                    if select:
                        cursor.movePosition(stepForward, moveMode)
                    break
        elif motion in (_PageDown, _PageUp):
            cursorHeight = self._qpart.cursorRect().height()
            qpartHeight = self._qpart.height()
            visibleLineCount = qpartHeight / cursorHeight
            direction = QTextCursor.Down if motion == _PageDown else QTextCursor.Up
            for _ in range(int(visibleLineCount)):
                cursor.movePosition(direction, moveMode)
        elif motion in (_Enter, _Return):
            if cursor.block().next().isValid():  # not the last line
                for _ in range(effectiveCount):
                    cursor.movePosition(QTextCursor.NextBlock, moveMode)
                    self.moveToFirstNonSpace(cursor, moveMode)
        else:
            assert 0, 'Not expected motion ' + str(motion)

        self._qpart.setTextCursor(cursor)

    @staticmethod
    def _iterateDocumentCharsForward(block, startColumnIndex):
        """Traverse document forward. Yield (block, columnIndex, char)
        Raise _TimeoutException if time is over
        """
        # Chars in the start line
        for columnIndex, char in list(enumerate(block.text()))[startColumnIndex:]:
            yield block, columnIndex, char
        block = block.next()

        # Next lines
        while block.isValid():
            for columnIndex, char in enumerate(block.text()):
                yield block, columnIndex, char

            block = block.next()

    @staticmethod
    def _iterateDocumentCharsBackward(block, startColumnIndex):
        """Traverse document forward. Yield (block, columnIndex, char)
        Raise _TimeoutException if time is over
        """
        # Chars in the start line
        for columnIndex, char in reversed(list(enumerate(block.text()[:startColumnIndex]))):
            yield block, columnIndex, char
        block = block.previous()

        # Next lines
        while block.isValid():
            for columnIndex, char in reversed(list(enumerate(block.text()))):
                yield block, columnIndex, char

            block = block.previous()

    def _resetSelection(self, moveToTop=False):
        """ Reset selection.
        If moveToTop is True - move cursor to the top position
        """
        ancor, pos = self._qpart.selectedPosition
        dst = min(ancor, pos) if moveToTop else pos
        self._qpart.cursorPosition = dst

    def _expandSelection(self):
        cursor = self._qpart.textCursor()
        anchor = cursor.anchor()
        pos = cursor.position()

        if pos >= anchor:
            anchorSide = QTextCursor.StartOfBlock
            cursorSide = QTextCursor.EndOfBlock
        else:
            anchorSide = QTextCursor.EndOfBlock
            cursorSide = QTextCursor.StartOfBlock

        cursor.setPosition(anchor)
        cursor.movePosition(anchorSide)
        cursor.setPosition(pos, QTextCursor.KeepAnchor)
        cursor.movePosition(cursorSide, QTextCursor.KeepAnchor)

        self._qpart.setTextCursor(cursor)


class BaseVisual(BaseCommandMode):
    color = QColor('#6699ff')
    _selectLines = NotImplementedError()

    def _processChar(self):
        ev = yield None

        # Get count
        typedCount = 0

        if ev.key() != _0:
            char = ev.text()
            while char.isdigit():
                digit = int(char)
                typedCount = (typedCount * 10) + digit
                ev = yield
                char = ev.text()

        count = typedCount if typedCount else 1

        # Now get the action
        action = code(ev)
        if action in self._SIMPLE_COMMANDS:
            cmdFunc = self._SIMPLE_COMMANDS[action]
            for _ in range(count):
                cmdFunc(self, action)
            if action not in (_v, _V):  # if not switched to another visual mode
                self._resetSelection(moveToTop=True)
            if self._vim.mode() is self:  # if the command didn't switch the mode
                self.switchMode(Normal)

            return True
        elif action == _Esc:
            self._resetSelection()
            self.switchMode(Normal)
            return True
        elif action == _g:
            ev = yield
            if code(ev) == _g:
                self._moveCursor('gg', 1, select=True)
                if self._selectLines:
                    self._expandSelection()
            return True
        elif action in (_f, _F, _t, _T):
            ev = yield
            if not isChar(ev):
                return True

            searchChar = ev.text()
            self._moveCursor(action, typedCount, searchChar=searchChar, select=True)
            return True
        elif action == _z:
            ev = yield
            if code(ev) == _z:
                self._qpart.centerCursor()
            return True
        elif action in self._MOTIONS:
            if self._selectLines and action in (_k, _Up, _j, _Down):
                # There is a bug in visual mode:
                # If a line is wrapped, cursor moves up, but stays on same line.
                # Then selection is expanded and cursor returns to previous position.
                # So user can't move the cursor up. So, in Visual mode we move cursor up until it
                # moved to previous line. The same bug when moving down
                cursorLine = self._qpart.cursorPosition[0]
                if (action in (_k, _Up) and cursorLine > 0) or \
                        (action in (_j, _Down) and (cursorLine + 1) < len(self._qpart.lines)):
                    while self._qpart.cursorPosition[0] == cursorLine:
                        self._moveCursor(action, typedCount, select=True)
            else:
                self._moveCursor(action, typedCount, select=True)

            if self._selectLines:
                self._expandSelection()
            return True
        elif action == _r:
            ev = yield
            newChar = ev.text()
            if newChar:
                newChars = [newChar if char != '\n' else '\n' \
                            for char in self._qpart.selectedText
                            ]
                newText = ''.join(newChars)
                self._qpart.selectedText = newText
            self.switchMode(Normal)
            return True
        elif isChar(ev):
            return True  # ignore unknown character
        else:
            return False  # but do not ignore not-a-character keys

        assert 0  # must StopIteration on if

    def _selectedLinesRange(self):
        """ Selected lines range for line manipulation methods
        """
        (startLine, _), (endLine, _) = self._qpart.selectedPosition
        start = min(startLine, endLine)
        end = max(startLine, endLine)
        return start, end

    def _selectRangeForRepeat(self, repeatLineCount):
        start = self._qpart.cursorPosition[0]
        self._qpart.selectedPosition = ((start, 0),
                                        (start + repeatLineCount - 1, 0))
        cursor = self._qpart.textCursor()
        # expand until the end of line
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        self._qpart.setTextCursor(cursor)

    def _saveLastEditLinesCmd(self, cmd, lineCount):
        self._vim.lastEditCmdFunc = lambda: self._SIMPLE_COMMANDS[cmd](self, cmd, lineCount)

    #
    # Simple commands
    #

    def cmdDelete(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)

        cursor = self._qpart.textCursor()
        if cursor.selectedText():
            if self._selectLines:
                start, end = self._selectedLinesRange()
                self._saveLastEditLinesCmd(cmd, end - start + 1)
                _globalClipboard.value = self._qpart.lines[start:end + 1]
                del self._qpart.lines[start:end + 1]
            else:
                _globalClipboard.value = cursor.selectedText()
                cursor.removeSelectedText()

    def cmdDeleteLines(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)

        start, end = self._selectedLinesRange()
        self._saveLastEditLinesCmd(cmd, end - start + 1)

        _globalClipboard.value = self._qpart.lines[start:end + 1]
        del self._qpart.lines[start:end + 1]

    def cmdInsertMode(self, cmd):
        self.switchMode(Insert)

    def cmdJoinLines(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)

        start, end = self._selectedLinesRange()
        count = end - start

        if not count:  # nothing to join
            return

        self._saveLastEditLinesCmd(cmd, end - start + 1)

        cursor = QTextCursor(self._qpart.document().findBlockByNumber(start))
        with self._qpart:
            for _ in range(count):
                cursor.movePosition(QTextCursor.EndOfBlock)
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
                self.moveToFirstNonSpace(cursor, QTextCursor.KeepAnchor)
                nonEmptyBlock = cursor.block().length() > 1
                cursor.removeSelectedText()
                if nonEmptyBlock:
                    cursor.insertText(' ')

        self._qpart.setTextCursor(cursor)

    def cmdAppendAfterChar(self, cmd):
        cursor = self._qpart.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.Right)
        self._qpart.setTextCursor(cursor)
        self.switchMode(Insert)

    def cmdReplaceSelectedLines(self, cmd):
        start, end = self._selectedLinesRange()
        _globalClipboard.value = self._qpart.lines[start:end + 1]

        lastLineLen = len(self._qpart.lines[end])
        self._qpart.selectedPosition = ((start, 0), (end, lastLineLen))
        self._qpart.selectedText = ''

        self.switchMode(Insert)

    def cmdResetSelection(self, cmd):
        self._qpart.cursorPosition = self._qpart.selectedPosition[0]

    def cmdInternalPaste(self, cmd):
        if not _globalClipboard.value:
            return

        with self._qpart:
            cursor = self._qpart.textCursor()

            if self._selectLines:
                start, end = self._selectedLinesRange()
                del self._qpart.lines[start:end + 1]
            else:
                cursor.removeSelectedText()

            if isinstance(_globalClipboard.value, str):
                self._qpart.textCursor().insertText(_globalClipboard.value)
            elif isinstance(_globalClipboard.value, list):
                currentLineIndex = self._qpart.cursorPosition[0]
                text = '\n'.join(_globalClipboard.value)
                index = currentLineIndex if self._selectLines else currentLineIndex + 1
                self._qpart.lines.insert(index, text)

    def cmdVisualMode(self, cmd):
        if not self._selectLines:
            self._resetSelection()
            return  # already in visual mode

        self.switchMode(Visual)

    def cmdVisualLinesMode(self, cmd):
        if self._selectLines:
            self._resetSelection()
            return  # already in visual lines mode

        self.switchMode(VisualLines)

    def cmdYank(self, cmd):
        if self._selectLines:
            start, end = self._selectedLinesRange()
            _globalClipboard.value = self._qpart.lines[start:end + 1]
        else:
            _globalClipboard.value = self._qpart.selectedText

        self._qpart.copy()

    def cmdChange(self, cmd):
        cursor = self._qpart.textCursor()
        if cursor.selectedText():
            if self._selectLines:
                _globalClipboard.value = cursor.selectedText().splitlines()
            else:
                _globalClipboard.value = cursor.selectedText()
            cursor.removeSelectedText()
        self.switchMode(Insert)

    def cmdUnIndent(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)
        else:
            start, end = self._selectedLinesRange()
            self._saveLastEditLinesCmd(cmd, end - start + 1)

        self._qpart._indenter.onChangeSelectedBlocksIndent(increase=False, withSpace=False)

        if repeatLineCount:
            self._resetSelection(moveToTop=True)

    def cmdIndent(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)
        else:
            start, end = self._selectedLinesRange()
            self._saveLastEditLinesCmd(cmd, end - start + 1)

        self._qpart._indenter.onChangeSelectedBlocksIndent(increase=True, withSpace=False)

        if repeatLineCount:
            self._resetSelection(moveToTop=True)

    def cmdAutoIndent(self, cmd, repeatLineCount=None):
        if repeatLineCount is not None:
            self._selectRangeForRepeat(repeatLineCount)
        else:
            start, end = self._selectedLinesRange()
            self._saveLastEditLinesCmd(cmd, end - start + 1)

        self._qpart._indenter.onAutoIndentTriggered()

        if repeatLineCount:
            self._resetSelection(moveToTop=True)

    _SIMPLE_COMMANDS = {
        _A: cmdAppendAfterChar,
        _c: cmdChange,
        _C: cmdReplaceSelectedLines,
        _d: cmdDelete,
        _D: cmdDeleteLines,
        _i: cmdInsertMode,
        _J: cmdJoinLines,
        _R: cmdReplaceSelectedLines,
        _p: cmdInternalPaste,
        _u: cmdResetSelection,
        _x: cmdDelete,
        _s: cmdChange,
        _S: cmdReplaceSelectedLines,
        _v: cmdVisualMode,
        _V: cmdVisualLinesMode,
        _X: cmdDeleteLines,
        _y: cmdYank,
        _Less: cmdUnIndent,
        _Greater: cmdIndent,
        _Equal: cmdAutoIndent,
    }


class Visual(BaseVisual):
    name = 'visual'

    _selectLines = False


class VisualLines(BaseVisual):
    name = 'visual lines'

    _selectLines = True

    def __init__(self, *args):
        BaseVisual.__init__(self, *args)
        self._expandSelection()


class Normal(BaseCommandMode):
    color = QColor('#33cc33')
    name = 'normal'

    def _processChar(self):
        ev = yield None
        # Get action count
        typedCount = 0

        if ev.key() != _0:
            char = ev.text()
            while char.isdigit():
                digit = int(char)
                typedCount = (typedCount * 10) + digit
                ev = yield
                char = ev.text()

        effectiveCount = typedCount or 1

        # Now get the action
        action = code(ev)

        if action in self._SIMPLE_COMMANDS:
            cmdFunc = self._SIMPLE_COMMANDS[action]
            cmdFunc(self, action, effectiveCount)
            return True
        elif action == _g:
            ev = yield
            if code(ev) == _g:
                self._moveCursor('gg', 1)

            return True
        elif action in (_f, _F, _t, _T):
            ev = yield
            if not isChar(ev):
                return True

            searchChar = ev.text()
            self._moveCursor(action, effectiveCount, searchChar=searchChar, select=False)
            return True
        elif action == _Period:  # repeat command
            if self._vim.lastEditCmdFunc is not None:
                if typedCount:
                    self._vim.lastEditCmdFunc(typedCount)
                else:
                    self._vim.lastEditCmdFunc()
            return True
        elif action in self._MOTIONS:
            self._moveCursor(action, typedCount, select=False)
            return True
        elif action in self._COMPOSITE_COMMANDS:
            moveCount = 0
            ev = yield

            if ev.key() != _0:  # 0 is a command, not a count
                char = ev.text()
                while char.isdigit():
                    digit = int(char)
                    moveCount = (moveCount * 10) + digit
                    ev = yield
                    char = ev.text()

            if moveCount == 0:
                moveCount = 1

            count = effectiveCount * moveCount

            # Get motion for a composite command
            motion = code(ev)
            searchChar = None

            if motion == _g:
                ev = yield
                if code(ev) == _g:
                    motion = 'gg'
                else:
                    return True
            elif motion in (_f, _F, _t, _T):
                ev = yield
                if not isChar(ev):
                    return True

                searchChar = ev.text()

            if (action != _z and motion in self._MOTIONS) or \
                    (action, motion) in ((_d, _d),
                                         (_y, _y),
                                         (_Less, _Less),
                                         (_Greater, _Greater),
                                         (_Equal, _Equal),
                                         (_z, _z)):
                cmdFunc = self._COMPOSITE_COMMANDS[action]
                cmdFunc(self, action, motion, searchChar, count)

            return True
        elif isChar(ev):
            return True  # ignore unknown character
        else:
            return False  # but do not ignore not-a-character keys

        assert 0  # must StopIteration on if

    def _repeat(self, count, func):
        """ Repeat action 1 or more times.
        If more than one - do it as 1 undoble action
        """
        if count != 1:
            with self._qpart:
                for _ in range(count):
                    func()
        else:
            func()

    def _saveLastEditSimpleCmd(self, cmd, count):
        def doCmd(count=count):
            self._SIMPLE_COMMANDS[cmd](self, cmd, count)

        self._vim.lastEditCmdFunc = doCmd

    def _saveLastEditCompositeCmd(self, cmd, motion, searchChar, count):
        def doCmd(count=count):
            self._COMPOSITE_COMMANDS[cmd](self, cmd, motion, searchChar, count)

        self._vim.lastEditCmdFunc = doCmd

    #
    # Simple commands
    #

    def cmdInsertMode(self, cmd, count):
        self.switchMode(Insert)

    def cmdInsertAtLineStartMode(self, cmd, count):
        cursor = self._qpart.textCursor()
        text = cursor.block().text()
        spaceLen = len(text) - len(text.lstrip())
        cursor.setPosition(cursor.block().position() + spaceLen)
        self._qpart.setTextCursor(cursor)

        self.switchMode(Insert)

    def cmdJoinLines(self, cmd, count):
        cursor = self._qpart.textCursor()
        if not cursor.block().next().isValid():  # last block
            return

        with self._qpart:
            for _ in range(count):
                cursor.movePosition(QTextCursor.EndOfBlock)
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
                self.moveToFirstNonSpace(cursor, QTextCursor.KeepAnchor)
                nonEmptyBlock = cursor.block().length() > 1
                cursor.removeSelectedText()
                if nonEmptyBlock:
                    cursor.insertText(' ')

                if not cursor.block().next().isValid():  # last block
                    break

        self._qpart.setTextCursor(cursor)

    def cmdReplaceMode(self, cmd, count):
        self.switchMode(Replace)
        self._qpart.setOverwriteMode(True)

    def cmdReplaceCharMode(self, cmd, count):
        self.switchMode(ReplaceChar)
        self._qpart.setOverwriteMode(True)

    def cmdAppendAfterLine(self, cmd, count):
        cursor = self._qpart.textCursor()
        cursor.movePosition(QTextCursor.EndOfBlock)
        self._qpart.setTextCursor(cursor)
        self.switchMode(Insert)

    def cmdAppendAfterChar(self, cmd, count):
        cursor = self._qpart.textCursor()
        cursor.movePosition(QTextCursor.Right)
        self._qpart.setTextCursor(cursor)
        self.switchMode(Insert)

    def cmdUndo(self, cmd, count):
        for _ in range(count):
            self._qpart.undo()

    def cmdRedo(self, cmd, count):
        for _ in range(count):
            self._qpart.redo()

    def cmdNewLineBelow(self, cmd, count):
        cursor = self._qpart.textCursor()
        cursor.movePosition(QTextCursor.EndOfBlock)
        self._qpart.setTextCursor(cursor)
        self._repeat(count, self._qpart._insertNewBlock)

        self._saveLastEditSimpleCmd(cmd, count)

        self.switchMode(Insert)

    def cmdNewLineAbove(self, cmd, count):
        cursor = self._qpart.textCursor()

        def insert():
            cursor.movePosition(QTextCursor.StartOfBlock)
            self._qpart.setTextCursor(cursor)
            self._qpart._insertNewBlock()
            cursor.movePosition(QTextCursor.Up)
            self._qpart._indenter.autoIndentBlock(cursor.block())

        self._repeat(count, insert)
        self._qpart.setTextCursor(cursor)

        self._saveLastEditSimpleCmd(cmd, count)

        self.switchMode(Insert)

    def cmdInternalPaste(self, cmd, count):
        if not _globalClipboard.value:
            return

        if isinstance(_globalClipboard.value, str):
            cursor = self._qpart.textCursor()
            if cmd == _p:
                cursor.movePosition(QTextCursor.Right)
                self._qpart.setTextCursor(cursor)

            self._repeat(count,
                         lambda: cursor.insertText(_globalClipboard.value))
            cursor.movePosition(QTextCursor.Left)
            self._qpart.setTextCursor(cursor)

        elif isinstance(_globalClipboard.value, list):
            index = self._qpart.cursorPosition[0]
            if cmd == _p:
                index += 1

            self._repeat(count,
                         lambda: self._qpart.lines.insert(index, '\n'.join(_globalClipboard.value)))

        self._saveLastEditSimpleCmd(cmd, count)

    def cmdSubstitute(self, cmd, count):
        """ s
        """
        cursor = self._qpart.textCursor()
        for _ in range(count):
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor)

        if cursor.selectedText():
            _globalClipboard.value = cursor.selectedText()
            cursor.removeSelectedText()

        self._saveLastEditSimpleCmd(cmd, count)
        self.switchMode(Insert)

    def cmdSubstituteLines(self, cmd, count):
        """ S
        """
        lineIndex = self._qpart.cursorPosition[0]
        availableCount = len(self._qpart.lines) - lineIndex
        effectiveCount = min(availableCount, count)

        _globalClipboard.value = self._qpart.lines[lineIndex:lineIndex + effectiveCount]
        with self._qpart:
            del self._qpart.lines[lineIndex:lineIndex + effectiveCount]
            self._qpart.lines.insert(lineIndex, '')
            self._qpart.cursorPosition = (lineIndex, 0)
            self._qpart._indenter.autoIndentBlock(self._qpart.textCursor().block())

        self._saveLastEditSimpleCmd(cmd, count)
        self.switchMode(Insert)

    def cmdVisualMode(self, cmd, count):
        cursor = self._qpart.textCursor()
        cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
        self._qpart.setTextCursor(cursor)
        self.switchMode(Visual)

    def cmdVisualLinesMode(self, cmd, count):
        self.switchMode(VisualLines)

    def cmdDelete(self, cmd, count):
        """ x
        """
        cursor = self._qpart.textCursor()
        direction = QTextCursor.Left if cmd == _X else QTextCursor.Right
        for _ in range(count):
            cursor.movePosition(direction, QTextCursor.KeepAnchor)

        if cursor.selectedText():
            _globalClipboard.value = cursor.selectedText()
            cursor.removeSelectedText()

        self._saveLastEditSimpleCmd(cmd, count)

    def cmdDeleteUntilEndOfBlock(self, cmd, count):
        """ C and D
        """
        cursor = self._qpart.textCursor()
        for _ in range(count - 1):
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        _globalClipboard.value = cursor.selectedText()
        cursor.removeSelectedText()
        if cmd == _C:
            self.switchMode(Insert)

        self._saveLastEditSimpleCmd(cmd, count)

    def cmdYankUntilEndOfLine(self, cmd, count):
        oldCursor = self._qpart.textCursor()
        cursor = self._qpart.textCursor()
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        _globalClipboard.value = cursor.selectedText()
        self._qpart.setTextCursor(cursor)
        self._qpart.copy()
        self._qpart.setTextCursor(oldCursor)

    _SIMPLE_COMMANDS = {_A: cmdAppendAfterLine,
                        _a: cmdAppendAfterChar,
                        _C: cmdDeleteUntilEndOfBlock,
                        _D: cmdDeleteUntilEndOfBlock,
                        _i: cmdInsertMode,
                        _I: cmdInsertAtLineStartMode,
                        _J: cmdJoinLines,
                        _r: cmdReplaceCharMode,
                        _R: cmdReplaceMode,
                        _v: cmdVisualMode,
                        _V: cmdVisualLinesMode,
                        _o: cmdNewLineBelow,
                        _O: cmdNewLineAbove,
                        _p: cmdInternalPaste,
                        _P: cmdInternalPaste,
                        _s: cmdSubstitute,
                        _S: cmdSubstituteLines,
                        _u: cmdUndo,
                        _U: cmdRedo,
                        _x: cmdDelete,
                        _X: cmdDelete,
                        _Y: cmdYankUntilEndOfLine,
                        }

    #
    # Composite commands
    #

    def cmdCompositeDelete(self, cmd, motion, searchChar, count):
        if motion in (_j, _Down):
            lineIndex = self._qpart.cursorPosition[0]
            availableCount = len(self._qpart.lines) - lineIndex
            if availableCount < 2:  # last line
                return

            effectiveCount = min(availableCount, count)

            _globalClipboard.value = self._qpart.lines[lineIndex:lineIndex + effectiveCount + 1]
            del self._qpart.lines[lineIndex:lineIndex + effectiveCount + 1]
        elif motion in (_k, _Up):
            lineIndex = self._qpart.cursorPosition[0]
            if lineIndex == 0:  # first line
                return

            effectiveCount = min(lineIndex, count)

            _globalClipboard.value = self._qpart.lines[lineIndex - effectiveCount:lineIndex + 1]
            del self._qpart.lines[lineIndex - effectiveCount:lineIndex + 1]
        elif motion == _d:  # delete whole line
            lineIndex = self._qpart.cursorPosition[0]
            availableCount = len(self._qpart.lines) - lineIndex

            effectiveCount = min(availableCount, count)

            _globalClipboard.value = self._qpart.lines[lineIndex:lineIndex + effectiveCount]
            del self._qpart.lines[lineIndex:lineIndex + effectiveCount]
        elif motion == _G:
            currentLineIndex = self._qpart.cursorPosition[0]
            _globalClipboard.value = self._qpart.lines[currentLineIndex:]
            del self._qpart.lines[currentLineIndex:]
        elif motion == 'gg':
            currentLineIndex = self._qpart.cursorPosition[0]
            _globalClipboard.value = self._qpart.lines[:currentLineIndex + 1]
            del self._qpart.lines[:currentLineIndex + 1]
        else:
            self._moveCursor(motion, count, select=True, searchChar=searchChar)

            selText = self._qpart.textCursor().selectedText()
            if selText:
                _globalClipboard.value = selText
                self._qpart.textCursor().removeSelectedText()

        self._saveLastEditCompositeCmd(cmd, motion, searchChar, count)

    def cmdCompositeChange(self, cmd, motion, searchChar, count):
        # TODO deletion and next insertion should be undo-ble as 1 action
        self.cmdCompositeDelete(cmd, motion, searchChar, count)
        self.switchMode(Insert)

    def cmdCompositeYank(self, cmd, motion, searchChar, count):
        oldCursor = self._qpart.textCursor()
        if motion == _y:
            cursor = self._qpart.textCursor()
            cursor.movePosition(QTextCursor.StartOfBlock)
            for _ in range(count - 1):
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            self._qpart.setTextCursor(cursor)
            _globalClipboard.value = [self._qpart.selectedText]
        else:
            self._moveCursor(motion, count, select=True, searchChar=searchChar)
            _globalClipboard.value = self._qpart.selectedText

        self._qpart.copy()
        self._qpart.setTextCursor(oldCursor)

    def cmdCompositeUnIndent(self, cmd, motion, searchChar, count):
        if motion == _Less:
            pass  # current line is already selected
        else:
            self._moveCursor(motion, count, select=True, searchChar=searchChar)
            self._expandSelection()

        self._qpart._indenter.onChangeSelectedBlocksIndent(increase=False, withSpace=False)
        self._resetSelection(moveToTop=True)

        self._saveLastEditCompositeCmd(cmd, motion, searchChar, count)

    def cmdCompositeIndent(self, cmd, motion, searchChar, count):
        if motion == _Greater:
            pass  # current line is already selected
        else:
            self._moveCursor(motion, count, select=True, searchChar=searchChar)
            self._expandSelection()

        self._qpart._indenter.onChangeSelectedBlocksIndent(increase=True, withSpace=False)
        self._resetSelection(moveToTop=True)

        self._saveLastEditCompositeCmd(cmd, motion, searchChar, count)

    def cmdCompositeAutoIndent(self, cmd, motion, searchChar, count):
        if motion == _Equal:
            pass  # current line is already selected
        else:
            self._moveCursor(motion, count, select=True, searchChar=searchChar)
            self._expandSelection()

        self._qpart._indenter.onAutoIndentTriggered()
        self._resetSelection(moveToTop=True)

        self._saveLastEditCompositeCmd(cmd, motion, searchChar, count)

    def cmdCompositeScrollView(self, cmd, motion, searchChar, count):
        if motion == _z:
            self._qpart.centerCursor()

    _COMPOSITE_COMMANDS = {_c: cmdCompositeChange,
                           _d: cmdCompositeDelete,
                           _y: cmdCompositeYank,
                           _Less: cmdCompositeUnIndent,
                           _Greater: cmdCompositeIndent,
                           _Equal: cmdCompositeAutoIndent,
                           _z: cmdCompositeScrollView,
                           }
