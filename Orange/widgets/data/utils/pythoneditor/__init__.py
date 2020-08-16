"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""qutepart --- Code editor component for PyQt and Pyside
=========================================================
"""

import sys
import os.path
import logging
import platform

from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtWidgets import QAction, QApplication, QDialog, QPlainTextEdit, QTextEdit, QWidget
from PyQt5.QtPrintSupport import QPrintDialog
from PyQt5.QtGui import QColor, QBrush, \
                        QFont, \
                        QIcon, QKeySequence, QPainter, QPen, QPalette, \
                        QTextCharFormat, QTextCursor, \
                        QTextBlock, QTextFormat

from qutepart.syntax import SyntaxManager
import qutepart.version


if 'sphinx-build' not in sys.argv[0]:
    # See explanation near `import sip` above
    from qutepart.syntaxhlighter import SyntaxHighlighter
    from qutepart.brackethlighter import BracketHighlighter
    from qutepart.completer import Completer
    from qutepart.lines import Lines
    from qutepart.rectangularselection import RectangularSelection
    import qutepart.sideareas
    from qutepart.indenter import Indenter
    import qutepart.vim

    def setPositionInBlock(cursor, positionInBlock, anchor=QTextCursor.MoveAnchor):
        return cursor.setPosition(cursor.block().position() + positionInBlock, anchor)


VERSION = qutepart.version.VERSION


logger = logging.getLogger('qutepart')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter("qutepart: %(message)s"))
logger.addHandler(consoleHandler)

logger.setLevel(logging.ERROR)


# After logging setup
import qutepart.syntax.loader
binaryParserAvailable = qutepart.syntax.loader.binaryParserAvailable


_ICONS_PATH = os.path.join(os.path.dirname(__file__), 'icons')

def getIcon(iconFileName):
    icon = QIcon.fromTheme(iconFileName)
    if icon.name() != iconFileName:
        # Use bundled fallback icon
        icon = QIcon(os.path.join(_ICONS_PATH, iconFileName))
    return icon


#Define for old Qt versions methods, which appeared in 4.7
if not hasattr(QTextCursor, 'positionInBlock'):
    def _positionInBlock(cursor):
        return cursor.position() - cursor.block().position()
    QTextCursor.positionInBlock = _positionInBlock



class EdgeLine(QWidget):
    def __init__(self, editor):
        QWidget.__init__(self, editor)
        self.__editor = editor
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.__editor.lineLengthEdgeColor)


class Qutepart(QPlainTextEdit):
    '''Qutepart is based on QPlainTextEdit, and you can use QPlainTextEdit methods,
    if you don't see some functionality here.

    **Text**

    ``text`` attribute holds current text. It may be read and written.::

        qpart.text = readFile()
        saveFile(qpart.text)

    This attribute always returns text, separated with ``\\n``. Use ``textForSaving()`` for get original text.

    It is recommended to use ``lines`` attribute whenever possible,
    because access to ``text`` might require long time on big files.
    Attribute is cached, only first read access after text has been changed in slow.

    **Selected text**

    ``selectedText`` attribute holds selected text. It may be read and written.
    Write operation replaces selection with new text. If nothing is selected - just inserts text::

        print qpart.selectedText  # print selection
        qpart.selectedText = 'new text'  # replace selection

    **Text lines**

    ``lines`` attribute, which represents text as list-of-strings like object
    and allows to modify it. Examples::

        qpart.lines[0]  # get the first line of the text
        qpart.lines[-1]  # get the last line of the text
        qpart.lines[2] = 'new text'  # replace 3rd line value with 'new text'
        qpart.lines[1:4]  # get 3 lines of text starting from the second line as list of strings
        qpart.lines[1:4] = ['new line 2', 'new line3', 'new line 4']  # replace value of 3 lines
        del qpart.lines[3]  # delete 4th line
        del qpart.lines[3:5]  # delete lines 4, 5, 6

        len(qpart.lines)  # get line count

        qpart.lines.append('new line')  # append new line to the end
        qpart.lines.insert(1, 'new line')  # insert new line before line 1

        print qpart.lines  # print all text as list of strings

        # iterate over lines.
        for lineText in qpart.lines:
            doSomething(lineText)

        qpart.lines = ['one', 'thow', 'three']  # replace whole text

    **Position and selection**

    * ``cursorPosition`` - cursor position as ``(line, column)``. Lines are numerated from zero. If column is set to ``None`` - cursor will be placed before first non-whitespace character. If line or column is bigger, than actual file, cursor will be placed to the last line, to the last column
    * ``absCursorPosition`` - cursor position as offset from the beginning of text.
    * ``selectedPosition`` - selection coordinates as ``((startLine, startCol), (cursorLine, cursorCol))``.
    * ``absSelectedPosition`` - selection coordinates as ``(startPosition, cursorPosition)`` where position is offset from the beginning of text.
    Rectangular selection is not available via API currently.

    **EOL, indentation, edge, current line**

    * ``eol`` - End Of Line character. Supported values are ``\\n``, ``\\r``, ``\\r\\n``. See comments for ``textForSaving()``
    * ``indentWidth`` - Width of ``Tab`` character, and width of one indentation level. Default is ``4``.
    * ``indentUseTabs`` - If True, ``Tab`` character inserts ``\\t``, otherwise - spaces. Default is ``False``.
    * ``lineLengthEdge`` - If not ``None`` - maximal allowed line width (i.e. 80 chars). Longer lines are marked with red (see ``lineLengthEdgeColor``) line. Default is ``None``.
    * ``lineLengthEdgeColor`` - Color of line length edge line. Default is red.
    * ``drawSolidEdge`` - Draw the edge as a solid vertical line. Default is ``False``.
    * ``drawIndentations`` - Draw indentations. Default is ``True``.
    * ``currentLineColor`` - Color of the current line background. If None then the current line is not highlighted. Default: #ffffa3

    **Visible white spaces**

    * ``drawIncorrectIndentation`` - Draw trailing whitespaces, tabs if text is indented with spaces, spaces if text is indented with tabs. Default is ``True``. Doesn't have any effect if ``drawAnyWhitespace`` is ``True``.
    * ``drawAnyWhitespace`` - Draw trailing and other whitespaces, used as indentation. Default is ``False``.

    **Autocompletion**

    Qutepart supports autocompletion, based on document contents.
    It is enabled, if ``completionEnabled`` is ``True``.
    ``completionThreshold`` is count of typed symbols, after which completion is shown.

    **Linters support**

    * ``lintMarks`` Linter messages as {lineNumber: (type, text)} dictionary. Cleared on any edit operation. Type is one of `Qutepart.LINT_ERROR, Qutepart.LINT_WARNING, Qutepart.LINT_NOTE)

    **Vim mode**

    ``vimModeEnabled`` - read-write property switches Vim mode. See also ``vimModeEnabledChanged``.
    ``vimModeIndication`` - An application shall display a label, which shows current Vim mode. This read-only property contains (QColor, str) to be displayed on the label. See also ``vimModeIndicationChanged``.

    **Actions**

    Component contains list of actions (QAction instances).
    Actions can be insered to some menu, a shortcut and an icon can be configured.

    Bookmarks:

    * ``toggleBookmarkAction`` - Set/Clear bookmark on current block
    * ``nextBookmarkAction`` - Jump to next bookmark
    * ``prevBookmarkAction`` - Jump to previous bookmark

    Scroll:

    * ``scrollUpAction`` - Scroll viewport Up
    * ``scrollDownAction`` - Scroll viewport Down
    * ``selectAndScrollUpAction`` - Select 1 line Up and scroll
    * ``selectAndScrollDownAction`` - Select 1 line Down and scroll

    Indentation:

    * ``increaseIndentAction`` - Increase indentation by 1 level
    * ``decreaseIndentAction`` - Decrease indentation by 1 level
    * ``autoIndentLineAction`` - Autoindent line
    * ``indentWithSpaceAction`` - Indent all selected lines by 1 space symbol
    * ``unIndentWithSpaceAction`` - Unindent all selected lines by 1 space symbol

    Lines:

    * ``moveLineUpAction`` - Move line Up
    * ``moveLineDownAction`` - Move line Down
    * ``deleteLineAction`` - Delete line
    * ``copyLineAction`` - Copy line
    * ``pasteLineAction`` - Paste line
    * ``cutLineAction`` - Cut line
    * ``duplicateLineAction`` - Duplicate line

    Other:
    * ``undoAction`` - Undo
    * ``redoAction`` - Redo
    * ``invokeCompletionAction`` - Invoke completion
    * ``printAction`` - Print file

    **Text modification and Undo/Redo**

    Sometimes, it is required to make few text modifications, which are Undo-Redoble as atomic operation.
    i.e. you want to indent (insert indentation) few lines of text, but user shall be able to
    Undo it in one step. In this case, you can use Qutepart as a context manager.::

        with qpart:
            qpart.modifySomeText()
            qpart.modifyOtherText()

    Nested atomic operations are joined in one operation

    **Signals**

    * ``userWarning(text)``` Warning, which shall be shown to the user on status bar. I.e. 'Rectangular selection area is too big'
    * ``languageChanged(langName)```              Language has changed. See also ``language()``
    * ``indentWidthChanged(int)``                 Indentation width changed. See also ``indentWidth``
    * ``indentUseTabsChanged(bool)``              Indentation uses tab property changed. See also ``indentUseTabs``
    * ``eolChanged(eol)``                         EOL mode changed. See also ``eol``.
    * ``vimModeEnabledChanged(enabled)            Vim mode has been enabled or disabled.
    * ``vimModeIndicationChanged(color, text)``   Vim mode changed. Parameters contain color and text to be displayed on an indicator. See also ``vimModeIndication``

    **Syntax parser**

    Qutepart supports two syntax parsers. One of them is written in C (faster) and the
    other in Python (slower). By default qutepart tries to load the faster parser and
    falls back to the slower one if there are import errors.
    If by some reasons a slower Python parser is preferred then qutepart can be
    instructed not to try to import the C parser. In order to do so an environment
    variable can be used (it needs to be set before the first import of qutepart), e.g.::

        import os
        os.environ['QPART_CPARSER'] = 'N'   # Python written syntax parser to be used

        import qutepart

    **Public methods**
    '''

    userWarning = pyqtSignal(str)
    languageChanged = pyqtSignal(str)
    indentWidthChanged = pyqtSignal(int)
    indentUseTabsChanged = pyqtSignal(bool)
    eolChanged = pyqtSignal(str)
    vimModeIndicationChanged = pyqtSignal(QColor, str)
    vimModeEnabledChanged = pyqtSignal(bool)

    LINT_ERROR = 'e'
    LINT_WARNING = 'w'
    LINT_NOTE = 'n'

    _DEFAULT_EOL = '\n'

    _DEFAULT_COMPLETION_THRESHOLD = 3
    _DEFAULT_COMPLETION_ENABLED = True

    _globalSyntaxManager = SyntaxManager()

    def __init__(self,
                 needMarkArea=True,
                 needLineNumbers=True,
                 needCompleter=True,
                 *args):
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
        self._currentLineColor = QColor('#ffffa3')
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
        palette.setColor(QPalette.Base, QColor('#ffffff'))
        palette.setColor(QPalette.Text, QColor('#000000'))
        self.setPalette(palette)

        self._highlighter = None
        self._bracketHighlighter = BracketHighlighter()

        self._lines = Lines(self)

        self.completionThreshold = self._DEFAULT_COMPLETION_THRESHOLD
        self.completionEnabled = self._DEFAULT_COMPLETION_ENABLED
        self._completer = None
        if needCompleter:
            self._completer = Completer(self)

        self._vim = None

        self._initActions()

        self._margins = []
        self._totalMarginWidth = -1

        if needLineNumbers:
            self.addMargin(qutepart.sideareas.LineNumberArea(self))
        if needMarkArea:
            self.addMargin(qutepart.sideareas.MarkArea(self))

        self._nonVimExtraSelections = []
        self._userExtraSelections = []  # we draw bracket highlighting, current line and extra selections by user
        self._userExtraSelectionFormat = QTextCharFormat()
        self._userExtraSelectionFormat.setBackground(QBrush(QColor('#ffee00')))

        self._lintMarks = {}

        self.cursorPositionChanged.connect(self._updateExtraSelections)
        self.textChanged.connect(self._dropUserExtraSelections)
        self.textChanged.connect(self._resetCachedText)
        self.textChanged.connect(self._clearLintMarks)

        fontFamilies = {'Windows':'Courier New',
                        'Darwin': 'Menlo'}
        fontFamily = fontFamilies.get(platform.system(), 'Monospace')
        self.setFont(QFont(fontFamily))

        self._updateExtraSelections()

    def terminate(self):
        """ Terminate Qutepart instance.
        This method MUST be called before application stop to avoid crashes and
        some other interesting effects
        Call it on close to free memory and stop background highlighting
        """
        self.text = ''
        if self._completer:
            self._completer.terminate()

        if self._highlighter is not None:
            self._highlighter.terminate()

        if self._vim is not None:
            self._vim.terminate()

    def _initActions(self):
        """Init shortcuts for text editing
        """

        def createAction(text, shortcut, slot, iconFileName=None):
            """Create QAction with given parameters and add to the widget
            """
            action = QAction(text, self)
            if iconFileName is not None:
                action.setIcon(getIcon(iconFileName))

            keySeq = shortcut if isinstance(shortcut, QKeySequence) else QKeySequence(shortcut)
            action.setShortcut(keySeq)
            action.setShortcutContext(Qt.WidgetShortcut)
            action.triggered.connect(slot)

            self.addAction(action)

            return action

        # scrolling
        self.scrollUpAction = createAction('Scroll up', 'Ctrl+Up',
                                           lambda: self._onShortcutScroll(down = False),
                                           'go-up')
        self.scrollDownAction = createAction('Scroll down', 'Ctrl+Down',
                                             lambda: self._onShortcutScroll(down = True),
                                             'go-down')
        self.selectAndScrollUpAction = createAction('Select and scroll Up', 'Ctrl+Shift+Up',
                                                    lambda: self._onShortcutSelectAndScroll(down = False))
        self.selectAndScrollDownAction = createAction('Select and scroll Down', 'Ctrl+Shift+Down',
                                                      lambda: self._onShortcutSelectAndScroll(down = True))

        # indentation
        self.increaseIndentAction = createAction('Increase indentation', 'Tab',
                                                 self._onShortcutIndent,
                                                 'format-indent-more')
        self.decreaseIndentAction = createAction('Decrease indentation', 'Shift+Tab',
                            lambda: self._indenter.onChangeSelectedBlocksIndent(increase = False),
                            'format-indent-less')
        self.autoIndentLineAction = createAction('Autoindent line', 'Ctrl+I',
                                                  self._indenter.onAutoIndentTriggered)
        self.indentWithSpaceAction = createAction('Indent with 1 space', 'Ctrl+Shift+Space',
                        lambda: self._indenter.onChangeSelectedBlocksIndent(increase=True,
                                                                              withSpace=True))
        self.unIndentWithSpaceAction = createAction('Unindent with 1 space', 'Ctrl+Shift+Backspace',
                            lambda: self._indenter.onChangeSelectedBlocksIndent(increase=False,
                                                                                  withSpace=True))

        # editing
        self.undoAction = createAction('Undo', QKeySequence.Undo,
                                       self.undo, 'edit-undo')
        self.redoAction = createAction('Redo', QKeySequence.Redo,
                                       self.redo, 'edit-redo')

        self.moveLineUpAction = createAction('Move line up', 'Alt+Up',
                                             lambda: self._onShortcutMoveLine(down = False), 'go-up')
        self.moveLineDownAction = createAction('Move line down', 'Alt+Down',
                                               lambda: self._onShortcutMoveLine(down = True), 'go-down')
        self.deleteLineAction = createAction('Delete line', 'Alt+Del', self._onShortcutDeleteLine, 'edit-delete')
        self.cutLineAction = createAction('Cut line', 'Alt+X', self._onShortcutCutLine, 'edit-cut')
        self.copyLineAction = createAction('Copy line', 'Alt+C', self._onShortcutCopyLine, 'edit-copy')
        self.pasteLineAction = createAction('Paste line', 'Alt+V', self._onShortcutPasteLine, 'edit-paste')
        self.duplicateLineAction = createAction('Duplicate line', 'Alt+D', self._onShortcutDuplicateLine)
        self.invokeCompletionAction = createAction('Invoke completion', 'Ctrl+Space', self._onCompletion)

        # other
        self.printAction = createAction('Print', 'Ctrl+P', self._onShortcutPrint, 'document-print')

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

        if exc_type is not None:
            return False

    def setFont(self, font):
        pass  # suppress dockstring for non-public method
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
        lineNumbersMargin = self.getMargin("line_numbers")
        if lineNumbersMargin:
            lineNumbersMargin.setFont(font)

    def showEvent(self, ev):
        pass  # suppress dockstring for non-public method
        """ Qt 5.big automatically changes font when adding document to workspace. Workaround this bug """
        super().setFont(self._fontBackup)
        return super().showEvent(ev)

    def _updateTabStopWidth(self):
        """Update tabstop width after font or indentation changed
        """
        self.setTabStopWidth(self.fontMetrics().width(' ' * self._indenter.width))

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        if not isinstance(value, (list, tuple)) or \
           not all([isinstance(item, str) for item in value]):
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
    def vimModeEnabled(self):
        return self._vim is not None

    @vimModeEnabled.setter
    def vimModeEnabled(self, enabled):
        if enabled:
            if self._vim is None:
                self._vim = qutepart.vim.Vim(self)
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

        If ``pos`` is an integer, it is interpreted as absolute position, if a tuple - as ``(line, column)``
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

        If ``pos`` is an integer, it is interpreted as absolute position, if a tuple - as ``(line, column)``
        """
        return self.replaceText(pos, 0, text)

    def detectSyntax(self,
                     xmlFileName=None,
                     mimeType=None,
                     language=None,
                     sourceFilePath=None,
                     firstLine=None):
        """Get syntax by next parameters (fill as many, as known):

            * name of XML file with syntax definition
            * MIME type of source file
            * Programming language name
            * Source file path
            * First line of source file

        First parameter in the list has the hightest priority.
        Old syntax is always cleared, even if failed to detect new.

        Method returns ``True``, if syntax is detected, and ``False`` otherwise
        """
        oldLanguage = self.language()

        self.clearSyntax()

        syntax = self._globalSyntaxManager.getSyntax(xmlFileName=xmlFileName,
                                                     mimeType=mimeType,
                                                     languageName=language,
                                                     sourceFilePath=sourceFilePath,
                                                     firstLine=firstLine)

        if syntax is not None:
            self._highlighter = SyntaxHighlighter(syntax, self)
            self._indenter.setSyntax(syntax)
            if self._completer:
                keywords = {kw for kwList in syntax.parser.lists.values() for kw in kwList}
                self._completer.setKeywords(keywords)

        newLanguage = self.language()
        if oldLanguage != newLanguage:
            self.languageChanged.emit(newLanguage)

        return syntax is not None

    def clearSyntax(self):
        """Clear syntax. Disables syntax highlighting

        This method might take long time, if document is big. Don't call it if you don't have to (i.e. in destructor)
        """
        if self._highlighter is not None:
            self._highlighter.terminate()
            self._highlighter = None
            self.languageChanged.emit(None)

    def language(self):
        """Get current language name.
        Return ``None`` for plain text
        """
        if self._highlighter is None:
            return None
        else:
            return self._highlighter.syntax().name

    def setCustomCompletions(self, wordSet):
        """Add a set of custom completions to the editors completions.

        This set is managed independently of the set of keywords and words from
        the current document, and can thus be changed at any time.

        """
        if not isinstance(wordSet, set):
            raise TypeError('"wordSet" is not a set: %s' % type(wordSet))
        if self._completer:
            self._completer.setCustomCompletions(wordSet)

    def isHighlightingInProgress(self):
        """Check if text highlighting is still in progress
        """
        return self._highlighter is not None and \
               self._highlighter.isInProgress()

    def isCode(self, blockOrBlockNumber, column):
        """Check if text at given position is a code.

        If language is not known, or text is not parsed yet, ``True`` is returned
        """
        if isinstance(blockOrBlockNumber, QTextBlock):
            block = blockOrBlockNumber
        else:
            block = self.document().findBlockByNumber(blockOrBlockNumber)

        return self._highlighter is None or \
               self._highlighter.isCode(block, column)

    def isComment(self, line, column):
        """Check if text at given position is a comment. Including block comments and here documents.

        If language is not known, or text is not parsed yet, ``False`` is returned
        """
        return self._highlighter is not None and \
               self._highlighter.isComment(self.document().findBlockByNumber(line), column)

    def isBlockComment(self, line, column):
        """Check if text at given position is a block comment.

        If language is not known, or text is not parsed yet, ``False`` is returned
        """
        return self._highlighter is not None and \
               self._highlighter.isBlockComment(self.document().findBlockByNumber(line), column)

    def isHereDoc(self, line, column):
        """Check if text at given position is a here document.

        If language is not known, or text is not parsed yet, ``False`` is returned
        """
        return self._highlighter is not None and \
               self._highlighter.isHereDoc(self.document().findBlockByNumber(line), column)

    def _dropUserExtraSelections(self):
        if self._userExtraSelections:
            self.setExtraSelections([])

    def setExtraSelections(self, selections):
        """Set list of extra selections.
        Selections are list of tuples ``(startAbsolutePosition, length)``.
        Extra selections are reset on any text modification.

        This is reimplemented method of QPlainTextEdit, it has different signature. Do not use QPlainTextEdit method
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

    def updateViewport(self):
        pass # suppress docstring for non-public method
        """Recalculates geometry for all the margins and the editor viewport
        """
        cr = self.contentsRect()
        currentX = cr.left()
        top = cr.top()
        height = cr.height()

        totalMarginWidth = 0
        for margin in self._margins:
            if not margin.isHidden():
                width = margin.width()
                margin.setGeometry(QRect(currentX, top, width, height))
                currentX += width
                totalMarginWidth += width

        if self._totalMarginWidth != totalMarginWidth:
            self._totalMarginWidth = totalMarginWidth
            self.updateViewportMargins()
        else:
            self._setSolidEdgeGeometry()

    def updateViewportMargins(self):
        pass # suppress docstring for non-public method
        """Sets the viewport margins and the solid edge geometry
        """
        self.setViewportMargins(self._totalMarginWidth, 0, 0, 0)
        self._setSolidEdgeGeometry()

    def resizeEvent(self, event):
        pass # suppress docstring for non-public method
        """QWidget.resizeEvent() implementation.
        Adjust line number area
        """
        QPlainTextEdit.resizeEvent(self, event)
        self.updateViewport()
        return

    def _setSolidEdgeGeometry(self):
        """Sets the solid edge line geometry if needed"""
        if self._lineLengthEdge is not None:
            cr = self.contentsRect()

            # contents margin usually gives 1
            # cursor rectangle left edge for the very first character usually
            # gives 4
            x = self.fontMetrics().width('9' * self._lineLengthEdge) + \
                self._totalMarginWidth + \
                self.contentsMargins().left() + \
                self.__cursorRect(self.firstVisibleBlock(), 0, offset=0).left()
            self._solidEdgeLine.setGeometry(QRect(x, cr.top(), 1, cr.bottom()))

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

    def textBeforeCursor(self):
        pass  # suppress docstring for non-API method, used by internal classes
        """Text in current block from start to cursor position
        """
        cursor = self.textCursor()
        return cursor.block().text()[:cursor.positionInBlock()]

    def keyPressEvent(self, event):
        pass # suppress dockstring for non-public method
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

        if event.matches(QKeySequence.InsertParagraphSeparator):
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
            qutepart.vim.isChar(event) and \
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
                    super(Qutepart, self).keyPressEvent(event)
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
                if event.text() and event.modifiers() == Qt.AltModifier:
                    return  # alt+letter is a shortcut. Not mine
                else:
                    self._lastKeyPressProcessedByParent = True
                    super(Qutepart, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._lastKeyPressProcessedByParent and self._completer is not None:
            """ A hacky way to do not show completion list after a event, processed by vim
            """

            text = event.text()
            textTyped = (text and \
                         event.modifiers() in (Qt.NoModifier, Qt.ShiftModifier)) and \
                         (text.isalpha() or text.isdigit() or text == '_')

            if textTyped or \
            (event.key() == Qt.Key_Backspace and self._completer.isVisible()):
                self._completer.invokeCompletionIfAvailable()

        super(Qutepart, self).keyReleaseEvent(event)

    def mousePressEvent(self, mouseEvent):
        pass  # suppress docstring for non-public method
        if mouseEvent.modifiers() in RectangularSelection.MOUSE_MODIFIERS and \
           mouseEvent.button() == Qt.LeftButton:
            self._rectangularSelection.mousePressEvent(mouseEvent)
        else:
            super(Qutepart, self).mousePressEvent(mouseEvent)

    def mouseMoveEvent(self, mouseEvent):
        pass  # suppress docstring for non-public method
        if mouseEvent.modifiers() in RectangularSelection.MOUSE_MODIFIERS and \
           mouseEvent.buttons() == Qt.LeftButton:
            self._rectangularSelection.mouseMoveEvent(mouseEvent)
        else:
            super(Qutepart, self).mouseMoveEvent(mouseEvent)

    def _chooseVisibleWhitespace(self, text):
        result = [False for _ in range(len(text))]

        lastNonSpaceColumn = len(text.rstrip()) - 1

        # Draw not trailing whitespace
        if self.drawAnyWhitespace:
            # Any
            for column, char in enumerate(text[:lastNonSpaceColumn]):
                if char.isspace() and \
                   (char == '\t' or \
                    column == 0 or \
                    text[column - 1].isspace() or \
                    ((column + 1) < lastNonSpaceColumn and \
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
            else:  # line too narrow, probably visible \t width is small
                return -1

        def drawEdgeLine(block, edgePos):
            painter.setPen(QPen(QBrush(self._lineLengthEdgeColor), 0))
            rect = self.__cursorRect(block, edgePos, 0)
            painter.drawLine(rect.topLeft(), rect.bottomLeft())

        def drawIndentMarker(block, column):
            painter.setPen(QColor(Qt.blue).lighter())
            rect = self.__cursorRect(block, column, offset=0)
            painter.drawLine(rect.topLeft(), rect.bottomLeft())

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
                        while text.startswith(self._indenter.text()) and \
                              len(text) > indentWidthChars and \
                              text[indentWidthChars].isspace():

                            if column != self._lineLengthEdge and \
                               (block.blockNumber(), column) != cursorPos:  # looks ugly, if both drawn
                                """on some fonts line is drawn below the cursor, if offset is 1
                                Looks like Qt bug"""
                                drawIndentMarker(block, column)

                            text = text[indentWidthChars:]
                            column += indentWidthChars

                # Draw edge, but not over a cursor
                if not self._drawSolidEdge:
                    edgePos = effectiveEdgePos(block.text())
                    if edgePos != -1 and edgePos != cursorPos[1]:
                        drawEdgeLine(block, edgePos)

                if self.drawAnyWhitespace or \
                   self.drawIncorrectIndentation:
                    text = block.text()
                    for column, draw in enumerate(self._chooseVisibleWhitespace(text)):
                        if draw:
                            drawWhiteSpace(block, column, text[column])

    def paintEvent(self, event):
        pass # suppress dockstring for non-public method
        """Paint event
        Draw indentation markers after main contents is drawn
        """
        super(Qutepart, self).paintEvent(event)
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
        QPlainTextEdit.setExtraSelections(self, self._nonVimExtraSelections + self._vim.extraSelections())

    def _onShortcutIndent(self):
        if self.textCursor().hasSelection():
            self._indenter.onChangeSelectedBlocksIndent(increase=True)
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

            # if operaiton is UnDone, marks are located incorrectly
            markMargin = self.getMargin("mark_area")
            if markMargin:
                markMargin.clearBookmarks(startBlock, endBlock.next())

            _moveBlock(blockToMove, startBlockNumber)

            self._selectLines(startBlockNumber + 1, endBlockNumber + 1)
        else:  # move previous block down
            blockToMove = startBlock.previous()
            if not blockToMove.isValid():
                return

            # if operaiton is UnDone, marks are located incorrectly
            markMargin = self.getMargin("mark_area")
            if markMargin:
                markMargin.clearBookmarks(startBlock, endBlock)

            _moveBlock(blockToMove, endBlockNumber)

            self._selectLines(startBlockNumber - 1, endBlockNumber - 1)

        if markMargin:
            markMargin.update()

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
        lines = self.lines[self._selectedLinesSlice()]
        text = QApplication.clipboard().text()
        if text:
            with self:
                if self.textCursor().hasSelection():
                    startBlockNumber, endBlockNumber = self._selectedBlockNumbers()
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
        lines = self.lines[self._selectedLinesSlice()]

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

    def _onShortcutPrint(self):
        """Ctrl+P handler.
        Show dialog, print file
        """
        dialog = QPrintDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            printer = dialog.printer()
            self.print_(printer)

    def _onCompletion(self):
        """Ctrl+Space handler.
        Invoke completer if so configured
        """
        if self._completer:
            self._completer.invokeCompletion()

    def insertFromMimeData(self, source):
        pass # suppress docstring for non-public method
        if source.hasFormat(self._rectangularSelection.MIME_TYPE):
            self._rectangularSelection.paste(source)
        else:
            super(Qutepart, self).insertFromMimeData(source)

    def __cursorRect(self, block, column, offset):
        cursor = QTextCursor(block)
        setPositionInBlock(cursor, column)
        return self.cursorRect(cursor).translated(offset, 0)

    def getMargins(self):
        """Provides the list of margins
        """
        return self._margins

    def addMargin(self, margin, index=None):
        """Adds a new margin.
           index: index in the list of margins. Default: to the end of the list
        """
        if index is None:
            self._margins.append(margin)
        else:
            self._margins.insert(index, margin)
        if margin.isVisible():
            self.updateViewport()

    def getMargin(self, name):
        """Provides the requested margin.
           Returns a reference to the margin if found and None otherwise
        """
        for margin in self._margins:
            if margin.getName() == name:
                return margin
        return None

    def delMargin(self, name):
        """Deletes a margin.
           Returns True if the margin was deleted and False otherwise.
        """
        for index, margin in enumerate(self._margins):
            if margin.getName() == name:
                visible = margin.isVisible()
                margin.clear()
                margin.deleteLater()
                del self._margins[index]
                if visible:
                    self.updateViewport()
                return True
        return False


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
