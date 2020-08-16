"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
# maximum number of lines we look backwards/forward to find out the indentation
# level (the bigger the number, the longer might be the delay)
MAX_SEARCH_OFFSET_LINES = 128


class IndentAlgNone:
    """No any indentation
    """
    def __init__(self, qpart):
        pass

    def computeSmartIndent(self, block, char):
        return ''


class IndentAlgBase(IndentAlgNone):
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
        raise NotImplemented()

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

        for block in cls.iterateBlocksBackFrom(block):
            for index, char in enumerate(reversed(block.text())):
                yield block, len(block.text()) - index - 1, char

    def findBracketBackward(self, block, column, bracket):
        """Search for a needle and return (block, column)
        Raise ValueError, if not found

        NOTE this method ignores comments
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
        else:
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
                for brackets in depth.keys():
                    opening, closing = brackets
                    if char == opening:
                        depth[brackets] -= 1
                        if depth[brackets] == 0:
                            return foundBlock, foundColumn
                    elif char == closing:
                        depth[brackets] += 1
        else:
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

    def _lastColumn(self, block):
        """Returns the last non-whitespace column in the given line.
        If there are only whitespaces in the line, the return value is -1.
        """
        text = block.text()
        index = len(block.text()) - 1
        while index >= 0 and \
              (text[index].isspace() or \
               self._qpart.isComment(block.blockNumber(), index)):
            index -= 1

        return index

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


class IndentAlgNormal(IndentAlgBase):
    """Class automatically computes indentation for lines
    This is basic indenter, which knows nothing about programming languages
    """
    def computeSmartIndent(self, block, char):
        return self._prevNonEmptyBlockIndent(block)
