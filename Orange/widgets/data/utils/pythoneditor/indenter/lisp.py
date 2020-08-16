"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
import re

from qutepart.indenter.base import IndentAlgBase

class IndentAlgLisp(IndentAlgBase):
    TRIGGER_CHARACTERS = ";"

    def computeSmartIndent(self, block, ch):
        """special rules: ;;; -> indent 0
                          ;;  -> align with next line, if possible
                          ;   -> usually on the same line as code -> ignore
        """
        if re.search(r'^\s*;;;', block.text()):
            return ''
        elif re.search(r'^\s*;;', block.text()):
            #try to align with the next line
            nextBlock = self._nextNonEmptyBlock(block)
            if nextBlock.isValid():
                return self._blockIndent(nextBlock)

        try:
            foundBlock, foundColumn = self.findBracketBackward(block, 0, '(')
        except ValueError:
            return ''
        else:
            return self._increaseIndent(self._blockIndent(foundBlock))
