"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from qutepart.indenter.base import IndentAlgBase

import re

# Indent after lines that match this regexp
rxIndent = re.compile(r'^\s*(def|if|unless|for|while|until|class|module|else|elsif|case|when|begin|rescue|ensure|catch)\b')

# Unindent lines that match this regexp
rxUnindent = re.compile(r'^\s*((end|when|else|elsif|rescue|ensure)\b|[\]\}])(.*)$')

rxBlockEnd = re.compile(r'\s*end$')


class Statement:
    def __init__(self, qpart, startBlock, endBlock):
        self._qpart = qpart
        self.startBlock = startBlock
        self.endBlock = endBlock

    # Convert to string for debugging
    def __str__(self):
        return "{ %d, %d}" % (self.startBlock.blockNumber(), self.endBlock.blockNumber())

    def offsetToCursor(self, offset):
        # Return (block, column)
        # TODO Provide helper function for this when API is converted to using cursors:
        block = self.startBlock
        while block != self.endBlock.next() and \
              len(block.text()) < offset:
            offset -= len(block.text())
            block = block.next()

        return block, offset

    def isCode(self, offset):
        # Return document.isCode at the given offset in a statement
        block, column = self.offsetToCursor(offset)
        return self._qpart.isCode(block.blockNumber(), column)

    def isComment(self, offset):
        # Return document.isComment at the given offset in a statement
        block, column = self.offsetToCursor(offset)
        return self._qpart.isComment(block.blockNumber(), column)

    def indent(self):
        # Return the indent at the beginning of the statement
        return IndentAlgRuby._blockIndent(self.startBlock)

    def content(self):
        # Return the content of the statement from the document
        cnt = ""
        block = self.startBlock
        while block != self.endBlock.next():
            text = block.text()
            if text.endswith('\\'):
                cnt += text[:-1]
                cnt += ' '
            else:
                cnt += text
            block = block.next()
        return cnt


class IndentAlgRuby(IndentAlgBase):
    """Indenter for Ruby
    """
    TRIGGER_CHARACTERS = "cdefhilnrsuw}]"

    def _isCommentBlock(self, block):
        text = block.text()
        firstColumn = self._firstNonSpaceColumn(text)
        return firstColumn == len(text) or self._isComment(block, firstColumn)

    def _isComment(self, block, column):
        return self._qpart.isComment(block.blockNumber(), column)

    def _prevNonCommentBlock(self, block):
        """Return the closest non-empty line, ignoring comments
        (result <= line). Return -1 if the document
        """
        block = self._prevNonEmptyBlock(block)
        while block.isValid() and self._isCommentBlock(block):
            block = self._prevNonEmptyBlock(block)
        return block

    @staticmethod
    def _isBlockContinuing(block):
        return block.text().endswith('\\')

    def _isLastCodeColumn(self, block, column):
        """Return true if the given column is at least equal to the column that
        contains the last non-whitespace character at the given line, or if
        the rest of the line is a comment.
        """
        return column >= self._lastColumn(block) or \
               self._isComment(block, self._nextNonSpaceColumn(block, column + 1))

    @staticmethod
    def testAtEnd(stmt, rx):
        """Look for a pattern at the end of the statement.

        Returns true if the pattern is found, in a position
        that is not inside a string or a comment, and the position +
        the length of the matching part is either the end of the
        statement, or a comment.

        The regexp must be global, and the search is continued until
        a match is found, or the end of the string is reached.
        """
        for match in rx.finditer(stmt.content()):
            if stmt.isCode(match.start()):
                if match.end() == len(stmt.content()):
                    return True
                if stmt.isComment(match.end()):
                    return True
        else:
            return False

    def lastAnchor(self, block, column):
        """Find the last open bracket before the current line.
        Return (block, column, char) or (None, None, None)
        """
        currentPos = -1
        currentBlock = None
        currentColumn = None
        currentChar = None
        for char in '({[':
            try:
                foundBlock, foundColumn = self.findBracketBackward(block, column, char)
            except ValueError:
                continue
            else:
                pos = foundBlock.position() + foundColumn
                if pos > currentPos:
                    currentBlock = foundBlock
                    currentColumn = foundColumn
                    currentChar = char
                    currentPos = pos

        return currentBlock, currentColumn, currentChar

    def isStmtContinuing(self, block):
        #Is there an open parenthesis?

        foundBlock, foundColumn, foundChar = self.lastAnchor(block, block.length())
        if foundBlock is not None:
            return True

        stmt = Statement(self._qpart, block, block)
        rx = re.compile(r'(\+|\-|\*|\/|\=|&&|\|\||\band\b|\bor\b|,)\s*')
        return self.testAtEnd(stmt, rx)

    def findStmtStart(self, block):
        """Return the first line that is not preceded by a "continuing" line.
        Return currBlock if currBlock <= 0
        """
        prevBlock = self._prevNonCommentBlock(block)
        while prevBlock.isValid() and \
              (((prevBlock == block.previous()) and self._isBlockContinuing(prevBlock)) or \
               self.isStmtContinuing(prevBlock)):
            block = prevBlock
            prevBlock = self._prevNonCommentBlock(block)
        return block

    @staticmethod
    def _isValidTrigger(block, ch):
        """check if the trigger characters are in the right context,
        otherwise running the indenter might be annoying to the user
        """
        if ch == "" or ch == "\n":
            return True # Explicit align or new line

        match = rxUnindent.match(block.text())
        return match is not None and \
               match.group(3) == ""

    def findPrevStmt(self, block):
        """Returns a tuple that contains the first and last line of the
        previous statement before line.
        """
        stmtEnd = self._prevNonCommentBlock(block)
        stmtStart = self.findStmtStart(stmtEnd)
        return Statement(self._qpart, stmtStart, stmtEnd)

    def isBlockStart(self, stmt):
        if rxIndent.search(stmt.content()):
            return True

        rx = re.compile(r'((\bdo\b|\{)(\s*\|.*\|)?\s*)')

        return self.testAtEnd(stmt, rx)

    @staticmethod
    def isBlockEnd(stmt):
        return rxUnindent.match(stmt.content())

    def findBlockStart(self, block):
        nested = 0
        stmt = Statement(self._qpart, block, block)
        while True:
            if not stmt.startBlock.isValid():
                return stmt
            stmt = self.findPrevStmt(stmt.startBlock)
            if self.isBlockEnd(stmt):
                nested += 1

            if self.isBlockStart(stmt):
                if nested == 0:
                    return stmt
                else:
                    nested -= 1

    def computeSmartIndent(self, block, ch):
        """indent gets three arguments: line, indentWidth in spaces,
        typed character indent
        """
        if not self._isValidTrigger(block, ch):
            return None

        prevStmt = self.findPrevStmt(block)
        if not prevStmt.endBlock.isValid():
            return None  # Can't indent the first line

        prevBlock = self._prevNonEmptyBlock(block)

        # HACK Detect here documents
        if self._qpart.isHereDoc(prevBlock.blockNumber(), prevBlock.length() - 2):
          return None

        # HACK Detect embedded comments
        if self._qpart.isBlockComment(prevBlock.blockNumber(), prevBlock.length() - 2):
            return None

        prevStmtCnt = prevStmt.content()
        prevStmtInd = prevStmt.indent()

        # Are we inside a parameter list, array or hash?
        foundBlock, foundColumn, foundChar = self.lastAnchor(block, 0)
        if foundBlock is not None:
            shouldIndent = foundBlock == prevStmt.endBlock or \
                           self.testAtEnd(prevStmt, re.compile(',\s*'))
            if (not self._isLastCodeColumn(foundBlock, foundColumn)) or \
                self.lastAnchor(foundBlock, foundColumn)[0] is not None:
                # TODO This is alignment, should force using spaces instead of tabs:
                if shouldIndent:
                    foundColumn += 1
                    nextCol = self._nextNonSpaceColumn(foundBlock, foundColumn)
                    if nextCol > 0 and \
                       (not self._isComment(foundBlock, nextCol)):
                        foundColumn = nextCol

                # Keep indent of previous statement, while aligning to the anchor column
                if len(prevStmtInd) > foundColumn:
                    return prevStmtInd
                else:
                    return self._makeIndentAsColumn(foundBlock, foundColumn)
            else:
                indent = self._blockIndent(foundBlock)
                if shouldIndent:
                    indent = self._increaseIndent(indent)
                return indent

        # Handle indenting of multiline statements.
        if (prevStmt.endBlock == block.previous() and \
             self._isBlockContinuing(prevStmt.endBlock)) or \
           self.isStmtContinuing(prevStmt.endBlock):
            if prevStmt.startBlock == prevStmt.endBlock:
                if ch == '' and \
                   len(self._blockIndent(block)) > len(self._blockIndent(prevStmt.endBlock)):
                    return None  # Don't force a specific indent level when aligning manually
                return self._increaseIndent(self._increaseIndent(prevStmtInd))
            else:
                return self._blockIndent(prevStmt.endBlock)

        if rxUnindent.match(block.text()):
            startStmt = self.findBlockStart(block)
            if startStmt.startBlock.isValid():
                return startStmt.indent()
            else:
                return None

        if self.isBlockStart(prevStmt) and not rxBlockEnd.search(prevStmt.content()):
            return self._increaseIndent(prevStmtInd)
        elif re.search(r'[\[\{]\s*$', prevStmtCnt) is not None:
            return self._increaseIndent(prevStmtInd)

        # Keep current
        return prevStmtInd
