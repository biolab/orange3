"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from PyQt5.QtGui import QTextCursor

# pylint: disable=pointless-string-statement

MAX_SEARCH_OFFSET_LINES = 128


class Indenter:
    """Qutepart functionality, related to indentation

    Public attributes:
        width           Indent width
        useTabs         Indent uses Tabs (instead of spaces)
    """
    _DEFAULT_INDENT_WIDTH = 4
    _DEFAULT_INDENT_USE_TABS = False

    def __init__(self, qpart):
        self._qpart = qpart

        self.width = self._DEFAULT_INDENT_WIDTH
        self.useTabs = self._DEFAULT_INDENT_USE_TABS

        self._smartIndenter = IndentAlgPython(qpart, self)

    def text(self):
        """Get indent text as \t or string of spaces
        """
        if self.useTabs:
            return '\t'
        else:
            return ' ' * self.width

    def triggerCharacters(self):
        """Trigger characters for smart indentation"""
        return self._smartIndenter.TRIGGER_CHARACTERS

    def autoIndentBlock(self, block, char='\n'):
        """Indent block after Enter pressed or trigger character typed
        """
        currentText = block.text()
        spaceAtStartLen = len(currentText) - len(currentText.lstrip())
        currentIndent = currentText[:spaceAtStartLen]
        indent = self._smartIndenter.computeIndent(block, char)
        if indent is not None and indent != currentIndent:
            self._qpart.replaceText(block.position(), spaceAtStartLen, indent)

    def onChangeSelectedBlocksIndent(self, increase, withSpace=False):
        """Tab or Space pressed and few blocks are selected, or Shift+Tab pressed
        Insert or remove text from the beginning of blocks
        """

        def blockIndentation(block):
            text = block.text()
            return text[:len(text) - len(text.lstrip())]

        def cursorAtSpaceEnd(block):
            cursor = QTextCursor(block)
            cursor.setPosition(block.position() + len(blockIndentation(block)))
            return cursor

        def indentBlock(block):
            cursor = cursorAtSpaceEnd(block)
            cursor.insertText(' ' if withSpace else self.text())

        def spacesCount(text):
            return len(text) - len(text.rstrip(' '))

        def unIndentBlock(block):
            currentIndent = blockIndentation(block)

            if currentIndent.endswith('\t'):
                charsToRemove = 1
            elif withSpace:
                charsToRemove = 1 if currentIndent else 0
            else:
                if self.useTabs:
                    charsToRemove = min(spacesCount(currentIndent), self.width)
                else:  # spaces
                    if currentIndent.endswith(self.text()):  # remove indent level
                        charsToRemove = self.width
                    else:  # remove all spaces
                        charsToRemove = min(spacesCount(currentIndent), self.width)

            if charsToRemove:
                cursor = cursorAtSpaceEnd(block)
                cursor.setPosition(cursor.position() - charsToRemove, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()

        cursor = self._qpart.textCursor()

        startBlock = self._qpart.document().findBlock(cursor.selectionStart())
        endBlock = self._qpart.document().findBlock(cursor.selectionEnd())
        if (cursor.selectionStart() != cursor.selectionEnd() and
                endBlock.position() == cursor.selectionEnd() and
                endBlock.previous().isValid()):
            # do not indent not selected line if indenting multiple lines
            endBlock = endBlock.previous()

        indentFunc = indentBlock if increase else unIndentBlock

        if startBlock != endBlock:  # indent multiply lines
            stopBlock = endBlock.next()

            block = startBlock

            with self._qpart:
                while block != stopBlock:
                    indentFunc(block)
                    block = block.next()

            newCursor = QTextCursor(startBlock)
            newCursor.setPosition(endBlock.position() + len(endBlock.text()),
                                  QTextCursor.KeepAnchor)
            self._qpart.setTextCursor(newCursor)
        else:  # indent 1 line
            indentFunc(startBlock)

    def onShortcutIndentAfterCursor(self):
        """Tab pressed and no selection. Insert text after cursor
        """
        cursor = self._qpart.textCursor()

        def insertIndent():
            if self.useTabs:
                cursor.insertText('\t')
            else:  # indent to integer count of indents from line start
                charsToInsert = self.width - (len(self._qpart.textBeforeCursor()) % self.width)
                cursor.insertText(' ' * charsToInsert)

        if cursor.positionInBlock() == 0:  # if no any indent - indent smartly
            block = cursor.block()
            self.autoIndentBlock(block, '')

            # if no smart indentation - just insert one indent
            if self._qpart.textBeforeCursor() == '':
                insertIndent()
        else:
            insertIndent()

    def onShortcutUnindentWithBackspace(self):
        """Backspace pressed, unindent
        """
        assert self._qpart.textBeforeCursor().endswith(self.text())

        charsToRemove = len(self._qpart.textBeforeCursor()) % len(self.text())
        if charsToRemove == 0:
            charsToRemove = len(self.text())

        cursor = self._qpart.textCursor()
        cursor.setPosition(cursor.position() - charsToRemove, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

    def onAutoIndentTriggered(self):
        """Indent current line or selected lines
        """
        cursor = self._qpart.textCursor()

        startBlock = self._qpart.document().findBlock(cursor.selectionStart())
        endBlock = self._qpart.document().findBlock(cursor.selectionEnd())

        if startBlock != endBlock:  # indent multiply lines
            stopBlock = endBlock.next()

            block = startBlock

            with self._qpart:
                while block != stopBlock:
                    self.autoIndentBlock(block, '')
                    block = block.next()
        else:  # indent 1 line
            self.autoIndentBlock(startBlock, '')


class IndentAlgBase:
    """Base class for indenters
    """
    TRIGGER_CHARACTERS = ""  # indenter is called, when user types Enter of one of trigger chars

    def __init__(self, qpart, indenter):
        self._qpart = qpart
        self._indenter = indenter

    def indentBlock(self, block):
        """Indent the block
        """
        self._setBlockIndent(block, self.computeIndent(block, ''))

    def computeIndent(self, block, char):
        """Compute indent for the block.
        Basic alorightm, which knows nothing about programming languages
        May be used by child classes
        """
        prevBlockText = block.previous().text()  # invalid block returns empty text
        if char == '\n' and \
                prevBlockText.strip() == '':  # continue indentation, if no text
            return self._prevBlockIndent(block)
        else:  # be smart
            return self.computeSmartIndent(block, char)

    def computeSmartIndent(self, block, char):
        """Compute smart indent.
        Block is current block.
        Char is typed character. \n or one of trigger chars
        Return indentation text, or None, if indentation shall not be modified

        Implementation might return self._prevNonEmptyBlockIndent(), if doesn't have
        any ideas, how to indent text better
        """
        raise NotImplementedError()

    def _qpartIndent(self):
        """Return text previous block, which is non empty (contains something, except spaces)
        Return '', if not found
        """
        return self._indenter.text()

    def _increaseIndent(self, indent):
        """Add 1 indentation level
        """
        return indent + self._qpartIndent()

    def _decreaseIndent(self, indent):
        """Remove 1 indentation level
        """
        if indent.endswith(self._qpartIndent()):
            return indent[:-len(self._qpartIndent())]
        else:  # oops, strange indentation, just return previous indent
            return indent

    def _makeIndentFromWidth(self, width):
        """Make indent text with specified with.
        Contains width count of spaces, or tabs and spaces
        """
        if self._indenter.useTabs:
            tabCount, spaceCount = divmod(width, self._indenter.width)
            return ('\t' * tabCount) + (' ' * spaceCount)
        else:
            return ' ' * width

    def _makeIndentAsColumn(self, block, column, offset=0):
        """ Make indent equal to column indent.
        Shiftted by offset
        """
        blockText = block.text()
        textBeforeColumn = blockText[:column]
        tabCount = textBeforeColumn.count('\t')

        visibleColumn = column + (tabCount * (self._indenter.width - 1))
        return self._makeIndentFromWidth(visibleColumn + offset)

    def _setBlockIndent(self, block, indent):
        """Set blocks indent. Modify text in qpart
        """
        currentIndent = self._blockIndent(block)
        self._qpart.replaceText((block.blockNumber(), 0), len(currentIndent), indent)

    @staticmethod
    def iterateBlocksFrom(block):
        """Generator, which iterates QTextBlocks from block until the End of a document
        But, yields not more than MAX_SEARCH_OFFSET_LINES
        """
        count = 0
        while block.isValid() and count < MAX_SEARCH_OFFSET_LINES:
            yield block
            block = block.next()
            count += 1

    @staticmethod
    def iterateBlocksBackFrom(block):
        """Generator, which iterates QTextBlocks from block until the Start of a document
        But, yields not more than MAX_SEARCH_OFFSET_LINES
        """
        count = 0
        while block.isValid() and count < MAX_SEARCH_OFFSET_LINES:
            yield block
            block = block.previous()
            count += 1

    @classmethod
    def iterateCharsBackwardFrom(cls, block, column):
        if column is not None:
            text = block.text()[:column]
            for index, char in enumerate(reversed(text)):
                yield block, len(text) - index - 1, char
            block = block.previous()

        for b in cls.iterateBlocksBackFrom(block):
            for index, char in enumerate(reversed(b.text())):
                yield b, len(b.text()) - index - 1, char

    def findBracketBackward(self, block, column, bracket):
        """Search for a needle and return (block, column)
        Raise ValueError, if not found
        """
        if bracket in ('(', ')'):
            opening = '('
            closing = ')'
        elif bracket in ('[', ']'):
            opening = '['
            closing = ']'
        elif bracket in ('{', '}'):
            opening = '{'
            closing = '}'
        else:
            raise AssertionError('Invalid bracket "%s"' % bracket)

        depth = 1
        for foundBlock, foundColumn, char in self.iterateCharsBackwardFrom(block, column):
            if not self._qpart.isComment(foundBlock.blockNumber(), foundColumn):
                if char == opening:
                    depth = depth - 1
                elif char == closing:
                    depth = depth + 1

                if depth == 0:
                    return foundBlock, foundColumn
        raise ValueError('Not found')

    def findAnyBracketBackward(self, block, column):
        """Search for a needle and return (block, column)
        Raise ValueError, if not found

        NOTE this methods ignores strings and comments
        """
        depth = {'()': 1,
                 '[]': 1,
                 '{}': 1
                 }

        for foundBlock, foundColumn, char in self.iterateCharsBackwardFrom(block, column):
            if self._qpart.isCode(foundBlock.blockNumber(), foundColumn):
                for brackets in depth:
                    opening, closing = brackets
                    if char == opening:
                        depth[brackets] -= 1
                        if depth[brackets] == 0:
                            return foundBlock, foundColumn
                    elif char == closing:
                        depth[brackets] += 1
        raise ValueError('Not found')

    @staticmethod
    def _lastNonSpaceChar(block):
        textStripped = block.text().rstrip()
        if textStripped:
            return textStripped[-1]
        else:
            return ''

    @staticmethod
    def _firstNonSpaceChar(block):
        textStripped = block.text().lstrip()
        if textStripped:
            return textStripped[0]
        else:
            return ''

    @staticmethod
    def _firstNonSpaceColumn(text):
        return len(text) - len(text.lstrip())

    @staticmethod
    def _lastNonSpaceColumn(text):
        return len(text.rstrip())

    @classmethod
    def _lineIndent(cls, text):
        return text[:cls._firstNonSpaceColumn(text)]

    @classmethod
    def _blockIndent(cls, block):
        if block.isValid():
            return cls._lineIndent(block.text())
        else:
            return ''

    @classmethod
    def _prevBlockIndent(cls, block):
        prevBlock = block.previous()

        if not block.isValid():
            return ''

        return cls._lineIndent(prevBlock.text())

    @classmethod
    def _prevNonEmptyBlockIndent(cls, block):
        return cls._blockIndent(cls._prevNonEmptyBlock(block))

    @staticmethod
    def _prevNonEmptyBlock(block):
        if not block.isValid():
            return block

        block = block.previous()
        while block.isValid() and \
                len(block.text().strip()) == 0:
            block = block.previous()
        return block

    @staticmethod
    def _nextNonEmptyBlock(block):
        if not block.isValid():
            return block

        block = block.next()
        while block.isValid() and \
                len(block.text().strip()) == 0:
            block = block.next()
        return block

    @staticmethod
    def _nextNonSpaceColumn(block, column):
        """Returns the column with a non-whitespace characters
        starting at the given cursor position and searching forwards.
        """
        textAfter = block.text()[column:]
        if textAfter.strip():
            spaceLen = len(textAfter) - len(textAfter.lstrip())
            return column + spaceLen
        else:
            return -1


class IndentAlgPython(IndentAlgBase):
    """Indenter for Python language.
    """

    def _computeSmartIndent(self, block, column):
        """Compute smart indent for case when cursor is on (block, column)
        """
        lineStripped = block.text()[:column].strip()  # empty text from invalid block is ok
        spaceLen = len(block.text()) - len(block.text().lstrip())

        """Move initial search position to bracket start, if bracket was closed
        l = [1,
             2]|
        """
        if lineStripped and \
                lineStripped[-1] in ')]}':
            try:
                backward = self.findBracketBackward(block, spaceLen + len(lineStripped) - 1,
                                                    lineStripped[-1])
                foundBlock, foundColumn = backward
            except ValueError:
                pass
            else:
                return self._computeSmartIndent(foundBlock, foundColumn)

        """Unindent if hanging indentation finished
        func(a,
             another_func(a,
                          b),|
        """
        if len(lineStripped) > 1 and \
                lineStripped[-1] == ',' and \
                lineStripped[-2] in ')]}':

            try:
                foundBlock, foundColumn = self.findBracketBackward(block,
                                                                   len(block.text()[
                                                                       :column].rstrip()) - 2,
                                                                   lineStripped[-2])
            except ValueError:
                pass
            else:
                return self._computeSmartIndent(foundBlock, foundColumn)

        """Check hanging indentation
        call_func(x,
                  y,
                  z
        But
        call_func(x,
            y,
            z
        """
        try:
            foundBlock, foundColumn = self.findAnyBracketBackward(block,
                                                                  column)
        except ValueError:
            pass
        else:
            # indent this way only line, which contains 'y', not 'z'
            if foundBlock.blockNumber() == block.blockNumber():
                return self._makeIndentAsColumn(foundBlock, foundColumn + 1)

        # finally, a raise, pass, and continue should unindent
        if lineStripped in ('continue', 'break', 'pass', 'raise', 'return') or \
                lineStripped.startswith('raise ') or \
                lineStripped.startswith('return '):
            return self._decreaseIndent(self._blockIndent(block))

        """
        for:

        func(a,
             b):
        """
        if lineStripped.endswith(':'):
            newColumn = spaceLen + len(lineStripped) - 1
            prevIndent = self._computeSmartIndent(block, newColumn)
            return self._increaseIndent(prevIndent)

        """ Generally, when a brace is on its own at the end of a regular line
        (i.e a data structure is being started), indent is wanted.
        For example:
        dictionary = {
            'foo': 'bar',
        }
        """
        if lineStripped.endswith('{['):
            return self._increaseIndent(self._blockIndent(block))

        return self._blockIndent(block)

    def computeSmartIndent(self, block, char):
        block = self._prevNonEmptyBlock(block)
        column = len(block.text())
        return self._computeSmartIndent(block, column)
