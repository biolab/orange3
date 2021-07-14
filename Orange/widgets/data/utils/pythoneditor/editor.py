"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import re
import sys

from AnyQt.QtCore import Signal, Qt, QRect, QPoint
from AnyQt.QtGui import QColor, QPainter, QPalette, QTextCursor, QKeySequence, QTextBlock, \
    QTextFormat, QBrush, QPen, QTextCharFormat
from AnyQt.QtWidgets import QPlainTextEdit, QWidget, QTextEdit, QAction, QApplication

from pygments.token import Token
from qtconsole.pygments_highlighter import PygmentsHighlighter, PygmentsBlockUserData

from Orange.widgets.data.utils.pythoneditor.completer import Completer
from Orange.widgets.data.utils.pythoneditor.brackethighlighter import BracketHighlighter
from Orange.widgets.data.utils.pythoneditor.indenter import Indenter
from Orange.widgets.data.utils.pythoneditor.lines import Lines
from Orange.widgets.data.utils.pythoneditor.rectangularselection import RectangularSelection
from Orange.widgets.data.utils.pythoneditor.vim import Vim, isChar


# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods


def setPositionInBlock(cursor, positionInBlock, anchor=QTextCursor.MoveAnchor):
    return cursor.setPosition(cursor.block().position() + positionInBlock, anchor)


def iterateBlocksFrom(block):
    """Generator, which iterates QTextBlocks from block until the End of a document
    """
    while block.isValid():
        yield block
        block = block.next()


def iterateBlocksBackFrom(block):
    """Generator, which iterates QTextBlocks from block until the Start of a document
    """
    while block.isValid():
        yield block
        block = block.previous()


class PythonEditor(QPlainTextEdit):
    userWarning = Signal(str)
    languageChanged = Signal(str)
    indentWidthChanged = Signal(int)
    indentUseTabsChanged = Signal(bool)
    eolChanged = Signal(str)
    vimModeIndicationChanged = Signal(QColor, str)
    vimModeEnabledChanged = Signal(bool)

    LINT_ERROR = 'e'
    LINT_WARNING = 'w'
    LINT_NOTE = 'n'

    _DEFAULT_EOL = '\n'

    _DEFAULT_COMPLETION_THRESHOLD = 3
    _DEFAULT_COMPLETION_ENABLED = True

    def __init__(self, *args):
        QPlainTextEdit.__init__(self, *args)

        self.setAttribute(Qt.WA_KeyCompression, False)  # vim can't process compressed keys

        self._lastKeyPressProcessedByParent = False
        # toPlainText() takes a lot of time on long texts, therefore it is cached
        self._cachedText = None

        self._fontBackup = self.font()

        self._eol = self._DEFAULT_EOL
        self._indenter = Indenter(self)
        self._lineLengthEdge = None
        self._lineLengthEdgeColor = QColor(255, 0, 0, 128)
        self._atomicModificationDepth = 0

        self.drawIncorrectIndentation = True
        self.drawAnyWhitespace = False
        self._drawIndentations = True
        self._drawSolidEdge = False
        self._solidEdgeLine = EdgeLine(self)
        self._solidEdgeLine.setVisible(False)

        self._rectangularSelection = RectangularSelection(self)

        """Sometimes color themes will be supported.
        Now black on white is hardcoded in the highlighters.
        Hardcode same palette for not highlighted text
        """
        palette = self.palette()
        # don't clear syntax highlighting when highlighting text
        palette.setBrush(QPalette.HighlightedText, QBrush(Qt.NoBrush))
        if QApplication.instance().property('darkMode'):
            palette.setColor(QPalette.Base, QColor('#111111'))
            palette.setColor(QPalette.Text, QColor('#ffffff'))
            palette.setColor(QPalette.Highlight, QColor('#444444'))
            self._currentLineColor = QColor('#111111')
        else:
            palette.setColor(QPalette.Base, QColor('#ffffff'))
            palette.setColor(QPalette.Text, QColor('#000000'))
            self._currentLineColor = QColor('#ffffff')
        self.setPalette(palette)

        self._bracketHighlighter = BracketHighlighter()

        self._lines = Lines(self)

        self.completionThreshold = self._DEFAULT_COMPLETION_THRESHOLD
        self.completionEnabled = self._DEFAULT_COMPLETION_ENABLED
        self._completer = Completer(self)
        self.auto_invoke_completions = False
        self.dot_invoke_completions = False

        doc = self.document()
        highlighter = PygmentsHighlighter(doc)
        doc.highlighter = highlighter

        self._vim = None

        self._initActions()

        self._line_number_margin = LineNumberArea(self)
        self._marginWidth = -1

        self._nonVimExtraSelections = []
        # we draw bracket highlighting, current line and extra selections by user
        self._userExtraSelections = []
        self._userExtraSelectionFormat = QTextCharFormat()
        self._userExtraSelectionFormat.setBackground(QBrush(QColor('#ffee00')))

        self._lintMarks = {}

        self.cursorPositionChanged.connect(self._updateExtraSelections)
        self.textChanged.connect(self._dropUserExtraSelections)
        self.textChanged.connect(self._resetCachedText)
        self.textChanged.connect(self._clearLintMarks)

        self._updateExtraSelections()

    def _initActions(self):
        """Init shortcuts for text editing
        """

        def createAction(text, shortcut, slot, iconFileName=None):
            """Create QAction with given parameters and add to the widget
            """
            action = QAction(text, self)
            # if iconFileName is not None:
            #     action.setIcon(getIcon(iconFileName))

            keySeq = shortcut if isinstance(shortcut, QKeySequence) else QKeySequence(shortcut)
            action.setShortcut(keySeq)
            action.setShortcutContext(Qt.WidgetShortcut)
            action.triggered.connect(slot)

            self.addAction(action)

            return action

        # custom Orange actions
        self.commentLine = createAction('Toggle comment line', 'Ctrl+/', self._onToggleCommentLine)

        # scrolling
        self.scrollUpAction = createAction('Scroll up', 'Ctrl+Up',
                                           lambda: self._onShortcutScroll(down=False),
                                           'go-up')
        self.scrollDownAction = createAction('Scroll down', 'Ctrl+Down',
                                             lambda: self._onShortcutScroll(down=True),
                                             'go-down')
        self.selectAndScrollUpAction = createAction('Select and scroll Up', 'Ctrl+Shift+Up',
                                                    lambda: self._onShortcutSelectAndScroll(
                                                        down=False))
        self.selectAndScrollDownAction = createAction('Select and scroll Down', 'Ctrl+Shift+Down',
                                                      lambda: self._onShortcutSelectAndScroll(
                                                          down=True))

        # indentation
        self.increaseIndentAction = createAction('Increase indentation', 'Tab',
                                                 self._onShortcutIndent,
                                                 'format-indent-more')
        self.decreaseIndentAction = \
            createAction('Decrease indentation', 'Shift+Tab',
                         lambda: self._indenter.onChangeSelectedBlocksIndent(
                             increase=False),
                         'format-indent-less')
        self.autoIndentLineAction = \
            createAction('Autoindent line', 'Ctrl+I',
                         self._indenter.onAutoIndentTriggered)
        self.indentWithSpaceAction = \
            createAction('Indent with 1 space', 'Ctrl+Shift+Space',
                         lambda: self._indenter.onChangeSelectedBlocksIndent(
                             increase=True,
                             withSpace=True))
        self.unIndentWithSpaceAction = \
            createAction('Unindent with 1 space', 'Ctrl+Shift+Backspace',
                         lambda: self._indenter.onChangeSelectedBlocksIndent(
                             increase=False,
                             withSpace=True))

        # editing
        self.undoAction = createAction('Undo', QKeySequence.Undo,
                                       self.undo, 'edit-undo')
        self.redoAction = createAction('Redo', QKeySequence.Redo,
                                       self.redo, 'edit-redo')

        self.moveLineUpAction = createAction('Move line up', 'Alt+Up',
                                             lambda: self._onShortcutMoveLine(down=False),
                                             'go-up')
        self.moveLineDownAction = createAction('Move line down', 'Alt+Down',
                                               lambda: self._onShortcutMoveLine(down=True),
                                               'go-down')
        self.deleteLineAction = createAction('Delete line', 'Alt+Del',
                                             self._onShortcutDeleteLine, 'edit-delete')
        self.cutLineAction = createAction('Cut line', 'Alt+X',
                                          self._onShortcutCutLine, 'edit-cut')
        self.copyLineAction = createAction('Copy line', 'Alt+C',
                                           self._onShortcutCopyLine, 'edit-copy')
        self.pasteLineAction = createAction('Paste line', 'Alt+V',
                                            self._onShortcutPasteLine, 'edit-paste')
        self.duplicateLineAction = createAction('Duplicate line', 'Alt+D',
                                                self._onShortcutDuplicateLine)

    def _onToggleCommentLine(self):
        cursor: QTextCursor = self.textCursor()
        cursor.beginEditBlock()

        startBlock = self.document().findBlock(cursor.selectionStart())
        endBlock = self.document().findBlock(cursor.selectionEnd())

        def lineIndentationLength(text):
            return len(text) - len(text.lstrip())

        def isHashCommentSelected(lines):
            return all(not line.strip() or line.lstrip().startswith('#') for line in lines)

        blocks = []
        lines = []

        block = startBlock
        line = block.text()
        if block != endBlock or line.strip():
            blocks += [block]
            lines += [line]
            while block != endBlock:
                block = block.next()
                line = block.text()
                if line.strip():
                    blocks += [block]
                    lines += [line]

        if isHashCommentSelected(lines):
            # remove the hash comment
            for block, text in zip(blocks, lines):
                cursor = QTextCursor(block)
                cursor.setPosition(block.position() + lineIndentationLength(text))
                for _ in range(lineIndentationLength(text[lineIndentationLength(text) + 1:]) + 1):
                    cursor.deleteChar()
        else:
            # add a hash comment
            for block, text in zip(blocks, lines):
                cursor = QTextCursor(block)
                cursor.setPosition(block.position() + lineIndentationLength(text))
                cursor.insertText('# ')

        if endBlock == self.document().lastBlock():
            if endBlock.text().strip():
                cursor = QTextCursor(endBlock)
                cursor.movePosition(QTextCursor.End)
                self.setTextCursor(cursor)
                self._insertNewBlock()
                cursorBlock = endBlock.next()
            else:
                cursorBlock = endBlock
        else:
            cursorBlock = endBlock.next()
        cursor = QTextCursor(cursorBlock)
        cursor.movePosition(QTextCursor.EndOfBlock)
        self.setTextCursor(cursor)
        cursor.endEditBlock()

    def _onShortcutIndent(self):
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._indenter.onChangeSelectedBlocksIndent(increase=True)
        elif cursor.positionInBlock() == cursor.block().length() - 1 and \
                cursor.block().text().strip():
            self._onCompletion()
        else:
            self._indenter.onShortcutIndentAfterCursor()

    def _onShortcutScroll(self, down):
        """Ctrl+Up/Down pressed, scroll viewport
        """
        value = self.verticalScrollBar().value()
        if down:
            value += 1
        else:
            value -= 1
        self.verticalScrollBar().setValue(value)

    def _onShortcutSelectAndScroll(self, down):
        """Ctrl+Shift+Up/Down pressed.
        Select line and scroll viewport
        """
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Down if down else QTextCursor.Up, QTextCursor.KeepAnchor)
        self.setTextCursor(cursor)
        self._onShortcutScroll(down)

    def _onShortcutHome(self, select):
        """Home pressed. Run a state machine:

            1. Not at the line beginning. Move to the beginning of the line or
               the beginning of the indent, whichever is closest to the current
               cursor position.
            2. At the line beginning. Move to the beginning of the indent.
            3. At the beginning of the indent. Go to the beginning of the block.
            4. At the beginning of the block. Go to the beginning of the indent.
        """
        # Gather info for cursor state and movement.
        cursor = self.textCursor()
        text = cursor.block().text()
        indent = len(text) - len(text.lstrip())
        anchor = QTextCursor.KeepAnchor if select else QTextCursor.MoveAnchor

        # Determine current state and move based on that.
        if cursor.positionInBlock() == indent:
            # We're at the beginning of the indent. Go to the beginning of the
            # block.
            cursor.movePosition(QTextCursor.StartOfBlock, anchor)
        elif cursor.atBlockStart():
            # We're at the beginning of the block. Go to the beginning of the
            # indent.
            setPositionInBlock(cursor, indent, anchor)
        else:
            # Neither of the above. There's no way I can find to directly
            # determine if we're at the beginning of a line. So, try moving and
            # see if the cursor location changes.
            pos = cursor.positionInBlock()
            cursor.movePosition(QTextCursor.StartOfLine, anchor)
            # If we didn't move, we were already at the beginning of the line.
            # So, move to the indent.
            if pos == cursor.positionInBlock():
                setPositionInBlock(cursor, indent, anchor)
            # If we did move, check to see if the indent was closer to the
            # cursor than the beginning of the indent. If so, move to the
            # indent.
            elif cursor.positionInBlock() < indent:
                setPositionInBlock(cursor, indent, anchor)

        self.setTextCursor(cursor)

    def _selectLines(self, startBlockNumber, endBlockNumber):
        """Select whole lines
        """
        startBlock = self.document().findBlockByNumber(startBlockNumber)
        endBlock = self.document().findBlockByNumber(endBlockNumber)
        cursor = QTextCursor(startBlock)
        cursor.setPosition(endBlock.position(), QTextCursor.KeepAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        self.setTextCursor(cursor)

    def _selectedBlocks(self):
        """Return selected blocks and tuple (startBlock, endBlock)
        """
        cursor = self.textCursor()
        return self.document().findBlock(cursor.selectionStart()), \
               self.document().findBlock(cursor.selectionEnd())

    def _selectedBlockNumbers(self):
        """Return selected block numbers and tuple (startBlockNumber, endBlockNumber)
        """
        startBlock, endBlock = self._selectedBlocks()
        return startBlock.blockNumber(), endBlock.blockNumber()

    def _onShortcutMoveLine(self, down):
        """Move line up or down
        Actually, not a selected text, but next or previous block is moved
        TODO keep bookmarks when moving
        """
        startBlock, endBlock = self._selectedBlocks()

        startBlockNumber = startBlock.blockNumber()
        endBlockNumber = endBlock.blockNumber()

        def _moveBlock(block, newNumber):
            text = block.text()
            with self:
                del self.lines[block.blockNumber()]
                self.lines.insert(newNumber, text)

        if down:  # move next block up
            blockToMove = endBlock.next()
            if not blockToMove.isValid():
                return

            _moveBlock(blockToMove, startBlockNumber)

            # self._selectLines(startBlockNumber + 1, endBlockNumber + 1)
        else:  # move previous block down
            blockToMove = startBlock.previous()
            if not blockToMove.isValid():
                return

            _moveBlock(blockToMove, endBlockNumber)

            # self._selectLines(startBlockNumber - 1, endBlockNumber - 1)

    def _selectedLinesSlice(self):
        """Get slice of selected lines
        """
        startBlockNumber, endBlockNumber = self._selectedBlockNumbers()
        return slice(startBlockNumber, endBlockNumber + 1, 1)

    def _onShortcutDeleteLine(self):
        """Delete line(s) under cursor
        """
        del self.lines[self._selectedLinesSlice()]

    def _onShortcutCopyLine(self):
        """Copy selected lines to the clipboard
        """
        lines = self.lines[self._selectedLinesSlice()]
        text = self._eol.join(lines)
        QApplication.clipboard().setText(text)

    def _onShortcutPasteLine(self):
        """Paste lines from the clipboard
        """
        text = QApplication.clipboard().text()
        if text:
            with self:
                if self.textCursor().hasSelection():
                    startBlockNumber, _ = self._selectedBlockNumbers()
                    del self.lines[self._selectedLinesSlice()]
                    self.lines.insert(startBlockNumber, text)
                else:
                    line, col = self.cursorPosition
                    if col > 0:
                        line = line + 1
                    self.lines.insert(line, text)

    def _onShortcutCutLine(self):
        """Cut selected lines to the clipboard
        """
        self._onShortcutCopyLine()
        self._onShortcutDeleteLine()

    def _onShortcutDuplicateLine(self):
        """Duplicate selected text or current line
        """
        cursor = self.textCursor()
        if cursor.hasSelection():  # duplicate selection
            text = cursor.selectedText()
            selectionStart, selectionEnd = cursor.selectionStart(), cursor.selectionEnd()
            cursor.setPosition(selectionEnd)
            cursor.insertText(text)
            # restore selection
            cursor.setPosition(selectionStart)
            cursor.setPosition(selectionEnd, QTextCursor.KeepAnchor)
            self.setTextCursor(cursor)
        else:
            line = cursor.blockNumber()
            self.lines.insert(line + 1, self.lines[line])
            self.ensureCursorVisible()

        self._updateExtraSelections()  # newly inserted text might be highlighted as braces

    def _onCompletion(self):
        """Ctrl+Space handler.
        Invoke completer if so configured
        """
        if self._completer:
            self._completer.invokeCompletion()

    @property
    def kernel_client(self):
        return self._completer.kernel_client

    @kernel_client.setter
    def kernel_client(self, kernel_client):
        self._completer.kernel_client = kernel_client

    @property
    def kernel_manager(self):
        return self._completer.kernel_manager

    @kernel_manager.setter
    def kernel_manager(self, kernel_manager):
        self._completer.kernel_manager = kernel_manager

    @property
    def vimModeEnabled(self):
        return self._vim is not None

    @vimModeEnabled.setter
    def vimModeEnabled(self, enabled):
        if enabled:
            if self._vim is None:
                self._vim = Vim(self)
                self._vim.modeIndicationChanged.connect(self.vimModeIndicationChanged)
                self.vimModeEnabledChanged.emit(True)
        else:
            if self._vim is not None:
                self._vim.terminate()
                self._vim = None
                self.vimModeEnabledChanged.emit(False)

    @property
    def vimModeIndication(self):
        if self._vim is not None:
            return self._vim.indication()
        else:
            return (None, None)

    @property
    def selectedText(self):
        text = self.textCursor().selectedText()

        # replace unicode paragraph separator with habitual \n
        text = text.replace('\u2029', '\n')

        return text

    @selectedText.setter
    def selectedText(self, text):
        self.textCursor().insertText(text)

    @property
    def cursorPosition(self):
        cursor = self.textCursor()
        return cursor.block().blockNumber(), cursor.positionInBlock()

    @cursorPosition.setter
    def cursorPosition(self, pos):
        line, col = pos

        line = min(line, len(self.lines) - 1)
        lineText = self.lines[line]

        if col is not None:
            col = min(col, len(lineText))
        else:
            col = len(lineText) - len(lineText.lstrip())

        cursor = QTextCursor(self.document().findBlockByNumber(line))
        setPositionInBlock(cursor, col)
        self.setTextCursor(cursor)

    @property
    def absCursorPosition(self):
        return self.textCursor().position()

    @absCursorPosition.setter
    def absCursorPosition(self, pos):
        cursor = self.textCursor()
        cursor.setPosition(pos)
        self.setTextCursor(cursor)

    @property
    def selectedPosition(self):
        cursor = self.textCursor()
        cursorLine, cursorCol = cursor.blockNumber(), cursor.positionInBlock()

        cursor.setPosition(cursor.anchor())
        startLine, startCol = cursor.blockNumber(), cursor.positionInBlock()

        return ((startLine, startCol), (cursorLine, cursorCol))

    @selectedPosition.setter
    def selectedPosition(self, pos):
        anchorPos, cursorPos = pos
        anchorLine, anchorCol = anchorPos
        cursorLine, cursorCol = cursorPos

        anchorCursor = QTextCursor(self.document().findBlockByNumber(anchorLine))
        setPositionInBlock(anchorCursor, anchorCol)

        # just get absolute position
        cursor = QTextCursor(self.document().findBlockByNumber(cursorLine))
        setPositionInBlock(cursor, cursorCol)

        anchorCursor.setPosition(cursor.position(), QTextCursor.KeepAnchor)
        self.setTextCursor(anchorCursor)

    @property
    def absSelectedPosition(self):
        cursor = self.textCursor()
        return cursor.anchor(), cursor.position()

    @absSelectedPosition.setter
    def absSelectedPosition(self, pos):
        anchorPos, cursorPos = pos
        cursor = self.textCursor()
        cursor.setPosition(anchorPos)
        cursor.setPosition(cursorPos, QTextCursor.KeepAnchor)
        self.setTextCursor(cursor)

    def resetSelection(self):
        """Reset selection. Nothing will be selected.
        """
        cursor = self.textCursor()
        cursor.setPosition(cursor.position())
        self.setTextCursor(cursor)

    @property
    def eol(self):
        return self._eol

    @eol.setter
    def eol(self, eol):
        if not eol in ('\r', '\n', '\r\n'):
            raise ValueError("Invalid EOL value")
        if eol != self._eol:
            self._eol = eol
            self.eolChanged.emit(self._eol)

    @property
    def indentWidth(self):
        return self._indenter.width

    @indentWidth.setter
    def indentWidth(self, width):
        if self._indenter.width != width:
            self._indenter.width = width
            self._updateTabStopWidth()
            self.indentWidthChanged.emit(width)

    @property
    def indentUseTabs(self):
        return self._indenter.useTabs

    @indentUseTabs.setter
    def indentUseTabs(self, use):
        if use != self._indenter.useTabs:
            self._indenter.useTabs = use
            self.indentUseTabsChanged.emit(use)

    @property
    def lintMarks(self):
        return self._lintMarks

    @lintMarks.setter
    def lintMarks(self, marks):
        if self._lintMarks != marks:
            self._lintMarks = marks
            self.update()

    def _clearLintMarks(self):
        if not self._lintMarks:
            self._lintMarks = {}
            self.update()

    @property
    def drawSolidEdge(self):
        return self._drawSolidEdge

    @drawSolidEdge.setter
    def drawSolidEdge(self, val):
        self._drawSolidEdge = val
        if val:
            self._setSolidEdgeGeometry()
        self.viewport().update()
        self._solidEdgeLine.setVisible(val and self._lineLengthEdge is not None)

    @property
    def drawIndentations(self):
        return self._drawIndentations

    @drawIndentations.setter
    def drawIndentations(self, val):
        self._drawIndentations = val
        self.viewport().update()

    @property
    def lineLengthEdge(self):
        return self._lineLengthEdge

    @lineLengthEdge.setter
    def lineLengthEdge(self, val):
        if self._lineLengthEdge != val:
            self._lineLengthEdge = val
            self.viewport().update()
            self._solidEdgeLine.setVisible(val is not None and self._drawSolidEdge)

    @property
    def lineLengthEdgeColor(self):
        return self._lineLengthEdgeColor

    @lineLengthEdgeColor.setter
    def lineLengthEdgeColor(self, val):
        if self._lineLengthEdgeColor != val:
            self._lineLengthEdgeColor = val
            if self._lineLengthEdge is not None:
                self.viewport().update()

    @property
    def currentLineColor(self):
        return self._currentLineColor

    @currentLineColor.setter
    def currentLineColor(self, val):
        if self._currentLineColor != val:
            self._currentLineColor = val
            self.viewport().update()

    def replaceText(self, pos, length, text):
        """Replace length symbols from ``pos`` with new text.

        If ``pos`` is an integer, it is interpreted as absolute position,
        if a tuple - as ``(line, column)``
        """
        if isinstance(pos, tuple):
            pos = self.mapToAbsPosition(*pos)

        endPos = pos + length

        if not self.document().findBlock(pos).isValid():
            raise IndexError('Invalid start position %d' % pos)

        if not self.document().findBlock(endPos).isValid():
            raise IndexError('Invalid end position %d' % endPos)

        cursor = QTextCursor(self.document())
        cursor.setPosition(pos)
        cursor.setPosition(endPos, QTextCursor.KeepAnchor)

        cursor.insertText(text)

    def insertText(self, pos, text):
        """Insert text at position

        If ``pos`` is an integer, it is interpreted as absolute position,
        if a tuple - as ``(line, column)``
        """
        return self.replaceText(pos, 0, text)

    def updateViewport(self):
        """Recalculates geometry for all the margins and the editor viewport
        """
        cr = self.contentsRect()
        currentX = cr.left()
        top = cr.top()
        height = cr.height()

        marginWidth = 0
        if not self._line_number_margin.isHidden():
            width = self._line_number_margin.width()
            self._line_number_margin.setGeometry(QRect(currentX, top, width, height))
            currentX += width
            marginWidth += width

        if self._marginWidth != marginWidth:
            self._marginWidth = marginWidth
            self.updateViewportMargins()
        else:
            self._setSolidEdgeGeometry()

    def updateViewportMargins(self):
        """Sets the viewport margins and the solid edge geometry"""
        self.setViewportMargins(self._marginWidth, 0, 0, 0)
        self._setSolidEdgeGeometry()

    def setDocument(self, document) -> None:
        super().setDocument(document)
        self._lines.setDocument(document)
        # forces margins to update after setting a new document
        self.blockCountChanged.emit(self.blockCount())

    def _updateExtraSelections(self):
        """Highlight current line
        """
        cursorColumnIndex = self.textCursor().positionInBlock()

        bracketSelections = self._bracketHighlighter.extraSelections(self,
                                                                     self.textCursor().block(),
                                                                     cursorColumnIndex)

        selections = self._currentLineExtraSelections() + \
                     self._rectangularSelection.selections() + \
                     bracketSelections + \
                     self._userExtraSelections

        self._nonVimExtraSelections = selections

        if self._vim is None:
            allSelections = selections
        else:
            allSelections = selections + self._vim.extraSelections()

        QPlainTextEdit.setExtraSelections(self, allSelections)

    def _updateVimExtraSelections(self):
        QPlainTextEdit.setExtraSelections(self,
                                          self._nonVimExtraSelections + self._vim.extraSelections())

    def _setSolidEdgeGeometry(self):
        """Sets the solid edge line geometry if needed"""
        if self._lineLengthEdge is not None:
            cr = self.contentsRect()

            # contents margin usually gives 1
            # cursor rectangle left edge for the very first character usually
            # gives 4
            x = self.fontMetrics().width('9' * self._lineLengthEdge) + \
                self._marginWidth + \
                self.contentsMargins().left() + \
                self.__cursorRect(self.firstVisibleBlock(), 0, offset=0).left()
            self._solidEdgeLine.setGeometry(QRect(x, cr.top(), 1, cr.bottom()))

    viewport_margins_updated = Signal(float)

    def setViewportMargins(self, left, top, right, bottom):
        """
        Override to align function signature with first character.
        """
        super().setViewportMargins(left, top, right, bottom)

        cursor = QTextCursor(self.firstVisibleBlock())
        setPositionInBlock(cursor, 0)
        cursorRect = self.cursorRect(cursor).translated(0, 0)

        first_char_indent = self._marginWidth + \
                            self.contentsMargins().left() + \
                            cursorRect.left()

        self.viewport_margins_updated.emit(first_char_indent)

    def textBeforeCursor(self):
        """Text in current block from start to cursor position
        """
        cursor = self.textCursor()
        return cursor.block().text()[:cursor.positionInBlock()]

    def keyPressEvent(self, event):
        """QPlainTextEdit.keyPressEvent() implementation.
        Catch events, which may not be catched with QShortcut and call slots
        """
        self._lastKeyPressProcessedByParent = False

        cursor = self.textCursor()

        def shouldUnindentWithBackspace():
            text = cursor.block().text()
            spaceAtStartLen = len(text) - len(text.lstrip())

            return self.textBeforeCursor().endswith(self._indenter.text()) and \
                   not cursor.hasSelection() and \
                   cursor.positionInBlock() == spaceAtStartLen

        def atEnd():
            return cursor.positionInBlock() == cursor.block().length() - 1

        def shouldAutoIndent(event):
            return atEnd() and \
                   event.text() and \
                   event.text() in self._indenter.triggerCharacters()

        def backspaceOverwrite():
            with self:
                cursor.deletePreviousChar()
                cursor.insertText(' ')
                setPositionInBlock(cursor, cursor.positionInBlock() - 1)
                self.setTextCursor(cursor)

        def typeOverwrite(text):
            """QPlainTextEdit records text input in replace mode as 2 actions:
            delete char, and type char. Actions are undone separately. This is
            workaround for the Qt bug"""
            with self:
                if not atEnd():
                    cursor.deleteChar()
                cursor.insertText(text)

        # mac specific shortcuts,
        if sys.platform == 'darwin':
            # it seems weird to delete line on CTRL+Backspace on Windows,
            # that's for deleting words. But Mac's CMD maps to Qt's CTRL.
            if event.key() == Qt.Key_Backspace and event.modifiers() == Qt.ControlModifier:
                self.deleteLineAction.trigger()
                event.accept()
                return
        if event.matches(QKeySequence.InsertLineSeparator):
            event.ignore()
            return
        elif event.matches(QKeySequence.InsertParagraphSeparator):
            if self._vim is not None:
                if self._vim.keyPressEvent(event):
                    return
            self._insertNewBlock()
        elif event.matches(QKeySequence.Copy) and self._rectangularSelection.isActive():
            self._rectangularSelection.copy()
        elif event.matches(QKeySequence.Cut) and self._rectangularSelection.isActive():
            self._rectangularSelection.cut()
        elif self._rectangularSelection.isDeleteKeyEvent(event):
            self._rectangularSelection.delete()
        elif event.key() == Qt.Key_Insert and event.modifiers() == Qt.NoModifier:
            if self._vim is not None:
                self._vim.keyPressEvent(event)
            else:
                self.setOverwriteMode(not self.overwriteMode())
        elif event.key() == Qt.Key_Backspace and \
                shouldUnindentWithBackspace():
            self._indenter.onShortcutUnindentWithBackspace()
        elif event.key() == Qt.Key_Backspace and \
                not cursor.hasSelection() and \
                self.overwriteMode() and \
                cursor.positionInBlock() > 0:
            backspaceOverwrite()
        elif self.overwriteMode() and \
                event.text() and \
                isChar(event) and \
                not cursor.hasSelection() and \
                cursor.positionInBlock() < cursor.block().length():
            typeOverwrite(event.text())
            if self._vim is not None:
                self._vim.keyPressEvent(event)
        elif event.matches(QKeySequence.MoveToStartOfLine):
            if self._vim is not None and \
                    self._vim.keyPressEvent(event):
                return
            else:
                self._onShortcutHome(select=False)
        elif event.matches(QKeySequence.SelectStartOfLine):
            self._onShortcutHome(select=True)
        elif self._rectangularSelection.isExpandKeyEvent(event):
            self._rectangularSelection.onExpandKeyEvent(event)
        elif shouldAutoIndent(event):
            with self:
                super().keyPressEvent(event)
                self._indenter.autoIndentBlock(cursor.block(), event.text())
        else:
            if self._vim is not None:
                if self._vim.keyPressEvent(event):
                    return

            # make action shortcuts override keyboard events (non-default Qt behaviour)
            for action in self.actions():
                seq = action.shortcut()
                if seq.count() == 1 and seq[0] == event.key() | int(event.modifiers()):
                    action.trigger()
                    break
            else:
                self._lastKeyPressProcessedByParent = True
                super().keyPressEvent(event)

        if event.key() == Qt.Key_Escape:
            event.accept()

    def terminate(self):
        """ Terminate Qutepart instance.
        This method MUST be called before application stop to avoid crashes and
        some other interesting effects
        Call it on close to free memory and stop background highlighting
        """
        self.text = ''
        if self._completer:
            self._completer.terminate()

        if self._vim is not None:
            self._vim.terminate()

    def __enter__(self):
        """Context management method.
        Begin atomic modification
        """
        self._atomicModificationDepth = self._atomicModificationDepth + 1
        if self._atomicModificationDepth == 1:
            self.textCursor().beginEditBlock()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context management method.
        End atomic modification
        """
        self._atomicModificationDepth = self._atomicModificationDepth - 1
        if self._atomicModificationDepth == 0:
            self.textCursor().endEditBlock()
        return exc_type is None

    def setFont(self, font):
        """Set font and update tab stop width
        """
        self._fontBackup = font
        QPlainTextEdit.setFont(self, font)
        self._updateTabStopWidth()

        # text on line numbers may overlap, if font is bigger, than code font
        # Note: the line numbers margin recalculates its width and if it has
        #       been changed then it calls updateViewport() which in turn will
        #       update the solid edge line geometry. So there is no need of an
        #       explicit call self._setSolidEdgeGeometry() here.
        lineNumbersMargin = self._line_number_margin
        if lineNumbersMargin:
            lineNumbersMargin.setFont(font)

    def setup_completer_appearance(self, size, font):
        self._completer.setup_appearance(size, font)

    def setAutoComplete(self, enabled):
        self.auto_invoke_completions = enabled

    def showEvent(self, ev):
        """ Qt 5.big automatically changes font when adding document to workspace.
        Workaround this bug """
        super().setFont(self._fontBackup)
        return super().showEvent(ev)

    def _updateTabStopWidth(self):
        """Update tabstop width after font or indentation changed
        """
        self.setTabStopWidth(self.fontMetrics().horizontalAdvance(' ' * self._indenter.width))

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        if not isinstance(value, (list, tuple)) or \
                not all(isinstance(item, str) for item in value):
            raise TypeError('Invalid new value of "lines" attribute')
        self.setPlainText('\n'.join(value))

    def _resetCachedText(self):
        """Reset toPlainText() result cache
        """
        self._cachedText = None

    @property
    def text(self):
        if self._cachedText is None:
            self._cachedText = self.toPlainText()

        return self._cachedText

    @text.setter
    def text(self, text):
        self.setPlainText(text)

    def textForSaving(self):
        """Get text with correct EOL symbols. Use this method for saving a file to storage
        """
        lines = self.text.splitlines()
        if self.text.endswith('\n'):  # splitlines ignores last \n
            lines.append('')
        return self.eol.join(lines) + self.eol

    def _get_token_at(self, block, column):
        dataObject = block.userData()

        if not hasattr(dataObject, 'tokens'):
            tokens = list(self.document().highlighter._lexer.get_tokens_unprocessed(block.text()))
            dataObject = PygmentsBlockUserData(**{
                'syntax_stack': dataObject.syntax_stack,
                'tokens': tokens
            })
            block.setUserData(dataObject)
        else:
            tokens = dataObject.tokens

        for next_token in tokens:
            c, _, _ = next_token
            if c > column:
                break
            token = next_token
        _, token_type, _ = token

        return token_type

    def isComment(self, line, column):
        """Check if character at column is a comment
        """
        block = self.document().findBlockByNumber(line)

        # here, pygments' highlighter is implemented, so the dataobject
        # that is originally defined in Qutepart isn't the same

        # so I'm using pygments' parser, storing it in the data object

        dataObject = block.userData()
        if dataObject is None:
            return False
        if len(dataObject.syntax_stack) > 1:
            return True

        token_type = self._get_token_at(block, column)

        def recursive_is_type(token, parent_token):
            if token.parent is None:
                return False
            if token.parent is parent_token:
                return True
            return recursive_is_type(token.parent, parent_token)

        return recursive_is_type(token_type, Token.Comment)

    def isCode(self, blockOrBlockNumber, column):
        """Check if text at given position is a code.

        If language is not known, or text is not parsed yet, ``True`` is returned
        """
        if isinstance(blockOrBlockNumber, QTextBlock):
            block = blockOrBlockNumber
        else:
            block = self.document().findBlockByNumber(blockOrBlockNumber)

        # here, pygments' highlighter is implemented, so the dataobject
        # that is originally defined in Qutepart isn't the same

        # so I'm using pygments' parser, storing it in the data object

        dataObject = block.userData()
        if dataObject is None:
            return True
        if len(dataObject.syntax_stack) > 1:
            return False

        token_type = self._get_token_at(block, column)

        def recursive_is_type(token, parent_token):
            if token.parent is None:
                return False
            if token.parent is parent_token:
                return True
            return recursive_is_type(token.parent, parent_token)

        return not any(recursive_is_type(token_type, non_code_token)
                       for non_code_token
                       in (Token.Comment, Token.String))

    def _dropUserExtraSelections(self):
        if self._userExtraSelections:
            self.setExtraSelections([])

    def setExtraSelections(self, selections):
        """Set list of extra selections.
        Selections are list of tuples ``(startAbsolutePosition, length)``.
        Extra selections are reset on any text modification.

        This is reimplemented method of QPlainTextEdit, it has different signature.
        Do not use QPlainTextEdit method
        """

        def _makeQtExtraSelection(startAbsolutePosition, length):
            selection = QTextEdit.ExtraSelection()
            cursor = QTextCursor(self.document())
            cursor.setPosition(startAbsolutePosition)
            cursor.setPosition(startAbsolutePosition + length, QTextCursor.KeepAnchor)
            selection.cursor = cursor
            selection.format = self._userExtraSelectionFormat
            return selection

        self._userExtraSelections = [_makeQtExtraSelection(*item) for item in selections]
        self._updateExtraSelections()

    def mapToAbsPosition(self, line, column):
        """Convert line and column number to absolute position
        """
        block = self.document().findBlockByNumber(line)
        if not block.isValid():
            raise IndexError("Invalid line index %d" % line)
        if column >= block.length():
            raise IndexError("Invalid column index %d" % column)
        return block.position() + column

    def mapToLineCol(self, absPosition):
        """Convert absolute position to ``(line, column)``
        """
        block = self.document().findBlock(absPosition)
        if not block.isValid():
            raise IndexError("Invalid absolute position %d" % absPosition)

        return (block.blockNumber(),
                absPosition - block.position())

    def resizeEvent(self, event):
        """QWidget.resizeEvent() implementation.
        Adjust line number area
        """
        QPlainTextEdit.resizeEvent(self, event)
        self.updateViewport()

    def _insertNewBlock(self):
        """Enter pressed.
        Insert properly indented block
        """
        cursor = self.textCursor()
        atStartOfLine = cursor.positionInBlock() == 0
        with self:
            cursor.insertBlock()
            if not atStartOfLine:  # if whole line is moved down - just leave it as is
                self._indenter.autoIndentBlock(cursor.block())
        self.ensureCursorVisible()

    def calculate_real_position(self, point):
        x = point.x() + self._line_number_margin.width()
        return QPoint(x, point.y())

    def position_widget_at_cursor(self, widget):
        # Retrieve current screen height
        desktop = QApplication.desktop()
        srect = desktop.availableGeometry(desktop.screenNumber(widget))

        left, top, right, bottom = (srect.left(), srect.top(),
                                    srect.right(), srect.bottom())
        ancestor = widget.parent()
        if ancestor:
            left = max(left, ancestor.x())
            top = max(top, ancestor.y())
            right = min(right, ancestor.x() + ancestor.width())
            bottom = min(bottom, ancestor.y() + ancestor.height())

        point = self.cursorRect().bottomRight()
        point = self.calculate_real_position(point)
        point = self.mapToGlobal(point)
        # Move to left of cursor if not enough space on right
        widget_right = point.x() + widget.width()
        if widget_right > right:
            point.setX(point.x() - widget.width())
        # Push to right if not enough space on left
        if point.x() < left:
            point.setX(left)

        # Moving widget above if there is not enough space below
        widget_bottom = point.y() + widget.height()
        x_position = point.x()
        if widget_bottom > bottom:
            point = self.cursorRect().topRight()
            point = self.mapToGlobal(point)
            point.setX(x_position)
            point.setY(point.y() - widget.height())

        if ancestor is not None:
            # Useful only if we set parent to 'ancestor' in __init__
            point = ancestor.mapFromGlobal(point)

        widget.move(point)

    def insert_completion(self, completion, completion_position):
        """Insert a completion into the editor.

        completion_position is where the completion was generated.

        The replacement range is computed using the (LSP) completion's
        textEdit field if it exists. Otherwise, we replace from the
        start of the word under the cursor.
        """
        if not completion:
            return

        cursor = self.textCursor()

        start = completion['start']
        end = completion['end']
        text = completion['text']

        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.setTextCursor(cursor)

    def keyReleaseEvent(self, event):
        if self._lastKeyPressProcessedByParent and self._completer is not None:
            # A hacky way to do not show completion list after a event, processed by vim

            text = event.text()
            textTyped = (text and
                         event.modifiers() in (Qt.NoModifier, Qt.ShiftModifier)) and \
                        (text.isalpha() or text.isdigit() or text == '_')
            dotTyped = text == '.'

            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.PreviousWord, QTextCursor.KeepAnchor)
            importTyped = cursor.selectedText() in ['from ', 'import ']

            if (textTyped and self.auto_invoke_completions) \
                    or dotTyped or importTyped:
                self._completer.invokeCompletionIfAvailable()

        super().keyReleaseEvent(event)

    def mousePressEvent(self, mouseEvent):
        if mouseEvent.modifiers() in RectangularSelection.MOUSE_MODIFIERS and \
                mouseEvent.button() == Qt.LeftButton:
            self._rectangularSelection.mousePressEvent(mouseEvent)
        else:
            super().mousePressEvent(mouseEvent)

    def mouseMoveEvent(self, mouseEvent):
        if mouseEvent.modifiers() in RectangularSelection.MOUSE_MODIFIERS and \
                mouseEvent.buttons() == Qt.LeftButton:
            self._rectangularSelection.mouseMoveEvent(mouseEvent)
        else:
            super().mouseMoveEvent(mouseEvent)

    def _chooseVisibleWhitespace(self, text):
        result = [False for _ in range(len(text))]

        lastNonSpaceColumn = len(text.rstrip()) - 1

        # Draw not trailing whitespace
        if self.drawAnyWhitespace:
            # Any
            for column, char in enumerate(text[:lastNonSpaceColumn]):
                if char.isspace() and \
                        (char == '\t' or
                         column == 0 or
                         text[column - 1].isspace() or
                         ((column + 1) < lastNonSpaceColumn and
                          text[column + 1].isspace())):
                    result[column] = True
        elif self.drawIncorrectIndentation:
            # Only incorrect
            if self.indentUseTabs:
                # Find big space groups
                firstNonSpaceColumn = len(text) - len(text.lstrip())
                bigSpaceGroup = ' ' * self.indentWidth
                column = 0
                while True:
                    column = text.find(bigSpaceGroup, column, lastNonSpaceColumn)
                    if column == -1 or column >= firstNonSpaceColumn:
                        break

                    for index in range(column, column + self.indentWidth):
                        result[index] = True
                    while index < lastNonSpaceColumn and \
                            text[index] == ' ':
                        result[index] = True
                        index += 1
                    column = index
            else:
                # Find tabs:
                column = 0
                while column != -1:
                    column = text.find('\t', column, lastNonSpaceColumn)
                    if column != -1:
                        result[column] = True
                        column += 1

        # Draw trailing whitespace
        if self.drawIncorrectIndentation or self.drawAnyWhitespace:
            for column in range(lastNonSpaceColumn + 1, len(text)):
                result[column] = True

        return result

    def _drawIndentMarkersAndEdge(self, paintEventRect):
        """Draw indentation markers
        """
        painter = QPainter(self.viewport())

        def drawWhiteSpace(block, column, char):
            leftCursorRect = self.__cursorRect(block, column, 0)
            rightCursorRect = self.__cursorRect(block, column + 1, 0)
            if leftCursorRect.top() == rightCursorRect.top():  # if on the same visual line
                middleHeight = (leftCursorRect.top() + leftCursorRect.bottom()) / 2
                if char == ' ':
                    painter.setPen(Qt.transparent)
                    painter.setBrush(QBrush(Qt.gray))
                    xPos = (leftCursorRect.x() + rightCursorRect.x()) / 2
                    painter.drawRect(QRect(xPos, middleHeight, 2, 2))
                else:
                    painter.setPen(QColor(Qt.gray).lighter(factor=120))
                    painter.drawLine(leftCursorRect.x() + 3, middleHeight,
                                     rightCursorRect.x() - 3, middleHeight)

        def effectiveEdgePos(text):
            """Position of edge in a block.
            Defined by self._lineLengthEdge, but visible width of \t is more than 1,
            therefore effective position depends on count and position of \t symbols
            Return -1 if line is too short to have edge
            """
            if self._lineLengthEdge is None:
                return -1

            tabExtraWidth = self.indentWidth - 1
            fullWidth = len(text) + (text.count('\t') * tabExtraWidth)
            if fullWidth <= self._lineLengthEdge:
                return -1

            currentWidth = 0
            for pos, char in enumerate(text):
                if char == '\t':
                    # Qt indents up to indentation level, so visible \t width depends on position
                    currentWidth += (self.indentWidth - (currentWidth % self.indentWidth))
                else:
                    currentWidth += 1
                if currentWidth > self._lineLengthEdge:
                    return pos
            # line too narrow, probably visible \t width is small
            return -1

        def drawEdgeLine(block, edgePos):
            painter.setPen(QPen(QBrush(self._lineLengthEdgeColor), 0))
            rect = self.__cursorRect(block, edgePos, 0)
            painter.drawLine(rect.topLeft(), rect.bottomLeft())

        def drawIndentMarker(block, column):
            painter.setPen(QColor(Qt.darkGray).lighter())
            rect = self.__cursorRect(block, column, offset=0)
            painter.drawLine(rect.topLeft(), rect.bottomLeft())

        def drawIndentMarkers(block, text, column):
            # this was 6 blocks deep  ~irgolic
            while text.startswith(self._indenter.text()) and \
                    len(text) > indentWidthChars and \
                    text[indentWidthChars].isspace():

                if column != self._lineLengthEdge and \
                        (block.blockNumber(),
                         column) != cursorPos:  # looks ugly, if both drawn
                    # on some fonts line is drawn below the cursor, if offset is 1
                    # Looks like Qt bug
                    drawIndentMarker(block, column)

                text = text[indentWidthChars:]
                column += indentWidthChars

        indentWidthChars = len(self._indenter.text())
        cursorPos = self.cursorPosition

        for block in iterateBlocksFrom(self.firstVisibleBlock()):
            blockGeometry = self.blockBoundingGeometry(block).translated(self.contentOffset())
            if blockGeometry.top() > paintEventRect.bottom():
                break

            if block.isVisible() and blockGeometry.toRect().intersects(paintEventRect):

                # Draw indent markers, if good indentation is not drawn
                if self._drawIndentations:
                    text = block.text()
                    if not self.drawAnyWhitespace:
                        column = indentWidthChars
                        drawIndentMarkers(block, text, column)

                # Draw edge, but not over a cursor
                if not self._drawSolidEdge:
                    edgePos = effectiveEdgePos(block.text())
                    if edgePos not in (-1, cursorPos[1]):
                        drawEdgeLine(block, edgePos)

                if self.drawAnyWhitespace or \
                        self.drawIncorrectIndentation:
                    text = block.text()
                    for column, draw in enumerate(self._chooseVisibleWhitespace(text)):
                        if draw:
                            drawWhiteSpace(block, column, text[column])

    def paintEvent(self, event):
        """Paint event
        Draw indentation markers after main contents is drawn
        """
        super().paintEvent(event)
        self._drawIndentMarkersAndEdge(event.rect())

    def _currentLineExtraSelections(self):
        """QTextEdit.ExtraSelection, which highlightes current line
        """
        if self._currentLineColor is None:
            return []

        def makeSelection(cursor):
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(self._currentLineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            cursor.clearSelection()
            selection.cursor = cursor
            return selection

        rectangularSelectionCursors = self._rectangularSelection.cursors()
        if rectangularSelectionCursors:
            return [makeSelection(cursor) \
                    for cursor in rectangularSelectionCursors]
        else:
            return [makeSelection(self.textCursor())]

    def insertFromMimeData(self, source):
        if source.hasFormat(self._rectangularSelection.MIME_TYPE):
            self._rectangularSelection.paste(source)
        elif source.hasUrls():
            cursor = self.textCursor()
            filenames = [url.toLocalFile() for url in source.urls()]
            text = ', '.join("'" + f.replace("'", "'\"'\"'") + "'"
                             for f in filenames)
            cursor.insertText(text)
        else:
            super().insertFromMimeData(source)

    def __cursorRect(self, block, column, offset):
        cursor = QTextCursor(block)
        setPositionInBlock(cursor, column)
        return self.cursorRect(cursor).translated(offset, 0)

    def get_current_word_and_position(self, completion=False, help_req=False,
                                      valid_python_variable=True):
        """
        Return current word, i.e. word at cursor position, and the start
        position.
        """
        cursor = self.textCursor()
        cursor_pos = cursor.position()

        if cursor.hasSelection():
            # Removes the selection and moves the cursor to the left side
            # of the selection: this is required to be able to properly
            # select the whole word under cursor (otherwise, the same word is
            # not selected when the cursor is at the right side of it):
            cursor.setPosition(min([cursor.selectionStart(),
                                    cursor.selectionEnd()]))
        else:
            # Checks if the first character to the right is a white space
            # and if not, moves the cursor one word to the left (otherwise,
            # if the character to the left do not match the "word regexp"
            # (see below), the word to the left of the cursor won't be
            # selected), but only if the first character to the left is not a
            # white space too.
            def is_space(move):
                curs = self.textCursor()
                curs.movePosition(move, QTextCursor.KeepAnchor)
                return not str(curs.selectedText()).strip()

            def is_special_character(move):
                """Check if a character is a non-letter including numbers."""
                curs = self.textCursor()
                curs.movePosition(move, QTextCursor.KeepAnchor)
                text_cursor = str(curs.selectedText()).strip()
                return len(
                    re.findall(r'([^\d\W]\w*)', text_cursor, re.UNICODE)) == 0

            if help_req:
                if is_special_character(QTextCursor.PreviousCharacter):
                    cursor.movePosition(QTextCursor.NextCharacter)
                elif is_special_character(QTextCursor.NextCharacter):
                    cursor.movePosition(QTextCursor.PreviousCharacter)
            elif not completion:
                if is_space(QTextCursor.NextCharacter):
                    if is_space(QTextCursor.PreviousCharacter):
                        return None
                    cursor.movePosition(QTextCursor.WordLeft)
            else:
                if is_space(QTextCursor.PreviousCharacter):
                    return None
                if is_special_character(QTextCursor.NextCharacter):
                    cursor.movePosition(QTextCursor.WordLeft)

        cursor.select(QTextCursor.WordUnderCursor)
        text = str(cursor.selectedText())
        startpos = cursor.selectionStart()

        # Find a valid Python variable name
        if valid_python_variable:
            match = re.findall(r'([^\d\W]\w*)', text, re.UNICODE)
            if not match:
                return None
            else:
                text = match[0]

        if completion:
            text = text[:cursor_pos - startpos]

        return text, startpos

    def get_current_word(self, completion=False, help_req=False,
                         valid_python_variable=True):
        """Return current word, i.e. word at cursor position."""
        ret = self.get_current_word_and_position(
            completion=completion,
            help_req=help_req,
            valid_python_variable=valid_python_variable
        )

        if ret is not None:
            return ret[0]
        return None


class EdgeLine(QWidget):
    def __init__(self, editor):
        QWidget.__init__(self, editor)
        self.__editor = editor
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.__editor.lineLengthEdgeColor)


class LineNumberArea(QWidget):
    _LEFT_MARGIN = 5
    _RIGHT_MARGIN = 5

    def __init__(self, parent):
        """qpart: reference to the editor
           name: margin identifier
           bit_count: number of bits to be used by the margin
        """
        super().__init__(parent)

        self._editor = parent
        self._name = 'line_numbers'
        self._bit_count = 0
        self._bitRange = None
        self.__allocateBits()

        self._countCache = (-1, -1)
        self._editor.updateRequest.connect(self.__updateRequest)

        self.__width = self.__calculateWidth()
        self._editor.blockCountChanged.connect(self.__updateWidth)

    def __updateWidth(self, newBlockCount=None):
        newWidth = self.__calculateWidth()
        if newWidth != self.__width:
            self.__width = newWidth
            self._editor.updateViewport()

    def paintEvent(self, event):
        """QWidget.paintEvent() implementation
        """
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.palette().color(QPalette.Window))
        painter.setPen(Qt.black)

        block = self._editor.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(
            self._editor.blockBoundingGeometry(block).translated(
                self._editor.contentOffset()).top())
        bottom = top + int(self._editor.blockBoundingRect(block).height())

        boundingRect = self._editor.blockBoundingRect(block)
        availableWidth = self.__width - self._RIGHT_MARGIN - self._LEFT_MARGIN
        availableHeight = self._editor.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.drawText(self._LEFT_MARGIN, top,
                                 availableWidth, availableHeight,
                                 Qt.AlignRight, number)
                # if boundingRect.height() >= singleBlockHeight * 2:  # wrapped block
                #     painter.fillRect(1, top + singleBlockHeight,
                #                      self.__width - 2,
                #                      boundingRect.height() - singleBlockHeight - 2,
                #                      Qt.darkGreen)

            block = block.next()
            boundingRect = self._editor.blockBoundingRect(block)
            top = bottom
            bottom = top + int(boundingRect.height())
            blockNumber += 1

    def __calculateWidth(self):
        digits = len(str(max(1, self._editor.blockCount())))
        return self._LEFT_MARGIN + self._editor.fontMetrics().horizontalAdvance(
            '9') * digits + self._RIGHT_MARGIN

    def width(self):
        """Desired width. Includes text and margins
        """
        return self.__width

    def setFont(self, font):
        super().setFont(font)
        self.__updateWidth()

    def __allocateBits(self):
        """Allocates the bit range depending on the required bit count
        """
        if self._bit_count < 0:
            raise Exception("A margin cannot request negative number of bits")
        if self._bit_count == 0:
            return

        # Build a list of occupied ranges
        margins = [self._editor._line_number_margin]

        occupiedRanges = []
        for margin in margins:
            bitRange = margin.getBitRange()
            if bitRange is not None:
                # pick the right position
                added = False
                for index, r in enumerate(occupiedRanges):
                    r = occupiedRanges[index]
                    if bitRange[1] < r[0]:
                        occupiedRanges.insert(index, bitRange)
                        added = True
                        break
                if not added:
                    occupiedRanges.append(bitRange)

        vacant = 0
        for r in occupiedRanges:
            if r[0] - vacant >= self._bit_count:
                self._bitRange = (vacant, vacant + self._bit_count - 1)
                return
            vacant = r[1] + 1
        # Not allocated, i.e. grab the tail bits
        self._bitRange = (vacant, vacant + self._bit_count - 1)

    def __updateRequest(self, rect, dy):
        """Repaint line number area if necessary
        """
        if dy:
            self.scroll(0, dy)
        elif self._countCache[0] != self._editor.blockCount() or \
                self._countCache[1] != self._editor.textCursor().block().lineCount():

            # if block height not added to rect, last line number sometimes is not drawn
            blockHeight = self._editor.blockBoundingRect(self._editor.firstVisibleBlock()).height()

            self.update(0, rect.y(), self.width(), rect.height() + round(blockHeight))
            self._countCache = (
                self._editor.blockCount(), self._editor.textCursor().block().lineCount())

        if rect.contains(self._editor.viewport().rect()):
            self._editor.updateViewportMargins()

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
            raise Exception("The margin '" + self._name +
                            "' did not allocate any bits for the values")
        if value < 0:
            raise Exception("The margin '" + self._name +
                            "' must be a positive integer")

        if value >= 2 ** self._bit_count:
            raise Exception("The margin '" + self._name +
                            "' value exceeds the allocated bit range")

        newMarginValue = value << self._bitRange[0]
        currentUserState = block.userState()

        if currentUserState in [0, -1]:
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
            raise Exception("The margin '" + self._name +
                            "' did not allocate any bits for the values")
        val = block.userState()
        if val in [0, -1]:
            return 0

        # Shift the value to the right
        val >>= self._bitRange[0]

        # Apply the mask to the value
        mask = 2 ** self._bit_count - 1
        val &= mask
        return val

    def hide(self):
        """Override the QWidget::hide() method to properly recalculate the
           editor viewport.
        """
        if not self.isHidden():
            super().hide()
            self._editor.updateViewport()

    def show(self):
        """Override the QWidget::show() method to properly recalculate the
           editor viewport.
        """
        if self.isHidden():
            super().show()
            self._editor.updateViewport()

    def setVisible(self, val):
        """Override the QWidget::setVisible(bool) method to properly
           recalculate the editor viewport.
        """
        if val != self.isVisible():
            if val:
                super().setVisible(True)
            else:
                super().setVisible(False)
            self._editor.updateViewport()

    # Convenience methods

    def clear(self):
        """Convenience method to reset all the block values to 0
        """
        if self._bit_count == 0:
            return

        block = self._editor.document().begin()
        while block.isValid():
            if self.getBlockValue(block):
                self.setBlockValue(block, 0)
            block = block.next()

    # Methods for 1-bit margins
    def isBlockMarked(self, block):
        return self.getBlockValue(block) != 0

    def toggleBlockMark(self, block):
        self.setBlockValue(block, 0 if self.isBlockMarked(block) else 1)
