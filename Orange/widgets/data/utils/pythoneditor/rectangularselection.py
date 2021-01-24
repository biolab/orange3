"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QApplication, QTextEdit
from PyQt5.QtGui import QKeyEvent, QKeySequence, QPalette, QTextCursor


class RectangularSelection:
    """This class does not replresent any object, but is part of Qutepart
    It just groups together Qutepart rectangular selection methods and fields
    """

    MIME_TYPE = 'text/rectangular-selection'

    # any of this modifiers with mouse select text
    MOUSE_MODIFIERS = (Qt.AltModifier | Qt.ControlModifier,
                       Qt.AltModifier | Qt.ShiftModifier,
                       Qt.AltModifier)

    _MAX_SIZE = 256

    def __init__(self, qpart):
        self._qpart = qpart
        self._start = None

        qpart.cursorPositionChanged.connect(self._reset)  # disconnected during Alt+Shift+...
        qpart.textChanged.connect(self._reset)
        qpart.selectionChanged.connect(self._reset)  # disconnected during Alt+Shift+...

    def _reset(self):
        """Cursor moved while Alt is not pressed, or text modified.
        Reset rectangular selection"""
        if self._start is not None:
            self._start = None
            self._qpart._updateExtraSelections()  # pylint: disable=protected-access

    def isDeleteKeyEvent(self, keyEvent):
        """Check if key event should be handled as Delete command"""
        return self._start is not None and \
               (keyEvent.matches(QKeySequence.Delete) or \
                (keyEvent.key() == Qt.Key_Backspace and keyEvent.modifiers() == Qt.NoModifier))

    def delete(self):
        """Del or Backspace pressed. Delete selection"""
        with self._qpart:
            for cursor in self.cursors():
                if cursor.hasSelection():
                    cursor.deleteChar()

    @staticmethod
    def isExpandKeyEvent(keyEvent):
        """Check if key event should expand rectangular selection"""
        return keyEvent.modifiers() & Qt.ShiftModifier and \
               keyEvent.modifiers() & Qt.AltModifier and \
               keyEvent.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Down, Qt.Key_Up,
                                  Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Home, Qt.Key_End)

    def onExpandKeyEvent(self, keyEvent):
        """One of expand selection key events"""
        if self._start is None:
            currentBlockText = self._qpart.textCursor().block().text()
            line = self._qpart.cursorPosition[0]
            visibleColumn = self._realToVisibleColumn(currentBlockText,
                                                      self._qpart.cursorPosition[1])
            self._start = (line, visibleColumn)
        modifiersWithoutAltShift = keyEvent.modifiers() & (~(Qt.AltModifier | Qt.ShiftModifier))
        newEvent = QKeyEvent(keyEvent.type(),
                             keyEvent.key(),
                             modifiersWithoutAltShift,
                             keyEvent.text(),
                             keyEvent.isAutoRepeat(),
                             keyEvent.count())

        self._qpart.cursorPositionChanged.disconnect(self._reset)
        self._qpart.selectionChanged.disconnect(self._reset)
        super(self._qpart.__class__, self._qpart).keyPressEvent(newEvent)
        self._qpart.cursorPositionChanged.connect(self._reset)
        self._qpart.selectionChanged.connect(self._reset)
        # extra selections will be updated, because cursor has been moved

    def _visibleCharPositionGenerator(self, text):
        currentPos = 0
        yield currentPos

        for char in text:
            if char == '\t':
                currentPos += self._qpart.indentWidth
                # trim reminder. If width('\t') == 4,   width('abc\t') == 4
                currentPos = currentPos // self._qpart.indentWidth * self._qpart.indentWidth
            else:
                currentPos += 1
            yield currentPos

    def _realToVisibleColumn(self, text, realColumn):
        """If \t is used, real position of symbol in block and visible position differs
        This function converts real to visible
        """
        generator = self._visibleCharPositionGenerator(text)
        for _ in range(realColumn):
            val = next(generator)
        val = next(generator)
        return val

    def _visibleToRealColumn(self, text, visiblePos):
        """If \t is used, real position of symbol in block and visible position differs
        This function converts visible to real.
        Bigger value is returned, if visiblePos is in the middle of \t, None if text is too short
        """
        if visiblePos == 0:
            return 0
        elif not '\t' in text:
            return visiblePos
        else:
            currentIndex = 1
            for currentVisiblePos in self._visibleCharPositionGenerator(text):
                if currentVisiblePos >= visiblePos:
                    return currentIndex - 1
                currentIndex += 1

            return None

    def cursors(self):
        """Cursors for rectangular selection.
        1 cursor for every line
        """
        cursors = []
        if self._start is not None:
            startLine, startVisibleCol = self._start
            currentLine, currentCol = self._qpart.cursorPosition
            if abs(startLine - currentLine) > self._MAX_SIZE or \
               abs(startVisibleCol - currentCol) > self._MAX_SIZE:
                # Too big rectangular selection freezes the GUI
                self._qpart.userWarning.emit('Rectangular selection area is too big')
                self._start = None
                return []

            currentBlockText = self._qpart.textCursor().block().text()
            currentVisibleCol = self._realToVisibleColumn(currentBlockText, currentCol)

            for lineNumber in range(min(startLine, currentLine),
                                    max(startLine, currentLine) + 1):
                block = self._qpart.document().findBlockByNumber(lineNumber)
                cursor = QTextCursor(block)
                realStartCol = self._visibleToRealColumn(block.text(), startVisibleCol)
                realCurrentCol = self._visibleToRealColumn(block.text(), currentVisibleCol)
                if realStartCol is None:
                    realStartCol = block.length()  # out of range value
                if realCurrentCol is None:
                    realCurrentCol = block.length()  # out of range value

                cursor.setPosition(cursor.block().position() +
                                   min(realStartCol, block.length() - 1))
                cursor.setPosition(cursor.block().position() +
                                   min(realCurrentCol, block.length() - 1),
                                   QTextCursor.KeepAnchor)
                cursors.append(cursor)

        return cursors

    def selections(self):
        """Build list of extra selections for rectangular selection"""
        selections = []
        cursors = self.cursors()
        if cursors:
            background = self._qpart.palette().color(QPalette.Highlight)
            foreground = self._qpart.palette().color(QPalette.HighlightedText)
            for cursor in cursors:
                selection = QTextEdit.ExtraSelection()
                selection.format.setBackground(background)
                selection.format.setForeground(foreground)
                selection.cursor = cursor

                selections.append(selection)

        return selections

    def isActive(self):
        """Some rectangle is selected"""
        return self._start is not None

    def copy(self):
        """Copy to the clipboard"""
        data = QMimeData()
        text = '\n'.join([cursor.selectedText() \
                            for cursor in self.cursors()])
        data.setText(text)
        data.setData(self.MIME_TYPE, text.encode('utf8'))
        QApplication.clipboard().setMimeData(data)

    def cut(self):
        """Cut action. Copy and delete
        """
        cursorPos = self._qpart.cursorPosition
        topLeft = (min(self._start[0], cursorPos[0]),
                   min(self._start[1], cursorPos[1]))
        self.copy()
        self.delete()

        # Move cursor to top-left corner of the selection,
        # so that if text gets pasted again, original text will be restored
        self._qpart.cursorPosition = topLeft

    def _indentUpTo(self, text, width):
        """Add space to text, so text width will be at least width.
        Return text, which must be added
        """
        visibleTextWidth = self._realToVisibleColumn(text, len(text))
        diff = width - visibleTextWidth
        if diff <= 0:
            return ''
        elif self._qpart.indentUseTabs and \
                all(char == '\t' for char in text):  # if using tabs and only tabs in text
            return '\t' * (diff // self._qpart.indentWidth) + \
                   ' ' * (diff % self._qpart.indentWidth)
        else:
            return ' ' * int(diff)

    def paste(self, mimeData):
        """Paste recrangular selection.
        Add space at the beginning of line, if necessary
        """
        if self.isActive():
            self.delete()
        elif self._qpart.textCursor().hasSelection():
            self._qpart.textCursor().deleteChar()

        text = bytes(mimeData.data(self.MIME_TYPE)).decode('utf8')
        lines = text.splitlines()
        cursorLine, cursorCol = self._qpart.cursorPosition
        if cursorLine + len(lines) > len(self._qpart.lines):
            for _ in range(cursorLine + len(lines) - len(self._qpart.lines)):
                self._qpart.lines.append('')

        with self._qpart:
            for index, line in enumerate(lines):
                currentLine = self._qpart.lines[cursorLine + index]
                newLine = currentLine[:cursorCol] + \
                          self._indentUpTo(currentLine, cursorCol) + \
                          line + \
                          currentLine[cursorCol:]
                self._qpart.lines[cursorLine + index] = newLine
        self._qpart.cursorPosition = cursorLine, cursorCol

    def mousePressEvent(self, mouseEvent):
        cursor = self._qpart.cursorForPosition(mouseEvent.pos())
        self._start = cursor.block().blockNumber(), cursor.positionInBlock()

    def mouseMoveEvent(self, mouseEvent):
        cursor = self._qpart.cursorForPosition(mouseEvent.pos())

        self._qpart.cursorPositionChanged.disconnect(self._reset)
        self._qpart.selectionChanged.disconnect(self._reset)
        self._qpart.setTextCursor(cursor)
        self._qpart.cursorPositionChanged.connect(self._reset)
        self._qpart.selectionChanged.connect(self._reset)
        # extra selections will be updated, because cursor has been moved
