"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from qutepart.indenter.base import IndentAlgBase


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
                foundBlock, foundColumn = self.findBracketBackward(block,
                                                                   spaceLen + len(lineStripped) - 1,
                                                                   lineStripped[-1])
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
                                                                   len(block.text()[:column].rstrip()) - 2,
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
