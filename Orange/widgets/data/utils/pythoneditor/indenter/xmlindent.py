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

class IndentAlgXml(IndentAlgBase):
    """Indenter for XML files
    """
    TRIGGER_CHARACTERS = "/>"

    def computeSmartIndent(self, block, char):
        """Compute indent for the block
        """
        lineText = block.text()
        prevLineText = self._prevNonEmptyBlock(block).text()

        alignOnly = char == ''

        if alignOnly:
            # XML might be all in one line, in which case we want to break that up.
            tokens = re.split(r'>\s*<', lineText)

            if len(tokens) > 1:

                prevIndent = self._lineIndent(prevLineText)

                for index, newLine in enumerate(tokens):
                    if index > 0:
                        newLine = '<' + newLine

                    if index < len(tokens) - 1:
                        newLine = newLine + '>'
                    if re.match(r'^\s*</', newLine):
                        char = '/'
                    elif re.match(r'\\>[^<>]*$', newLine):
                        char = '>'
                    else:
                        char = '\n'

                    indentation = self.processChar(newLine, prevLineText, char)
                    newLine = indentation + newLine

                    tokens[index] = newLine
                    prevLineText = newLine;

                self._qpart.lines[block.blockNumber()] =  '\n'.join(tokens)
                return None
            else:  # no tokens, do not split line, just compute indent
                if re.search(r'^\s*</', lineText):
                    char = '/'
                elif re.search(r'>[^<>]*', lineText):
                    char = '>'
                else:
                    char = '\n'

        return self.processChar(lineText, prevLineText, char)

    def processChar(self, lineText, prevLineText, char):
        prevIndent = self._lineIndent(prevLineText)
        if char == '/':
            if not re.match(r'^\s*</', lineText):
                # might happen when we have something like <foo bar="asdf/ycxv">
                # don't change indentation then
                return prevIndent

            if not re.match(r'\s*<[^/][^>]*[^/]>[^<>]*$', prevLineText):
                # decrease indent when we write </ and prior line did not start a tag
                return self._decreaseIndent(prevIndent)
        elif char == '>':
            # increase indent width when we write <...> or <.../> but not </...>
            # and the prior line didn't close a tag
            if not prevLineText:  # first line, zero indent
                return ''
            if re.match(r'^<(\?xml|!DOCTYPE).*', prevLineText):
                return ''
            elif re.match(r'^<(\?xml|!DOCTYPE).*', lineText):
                return ''
            elif re.match('^\s*</', lineText):
                #closing tag, decrease indentation when previous didn't open a tag
                if re.match(r'\s*<[^/][^>]*[^/]>[^<>]*$', prevLineText):
                    # keep indent when prev line opened a tag
                    return prevIndent;
                else:
                    return self._decreaseIndent(prevIndent)
            elif re.search(r'<([/!][^>]+|[^>]+/)>\s*$', prevLineText):
                # keep indent when prev line closed a tag or was empty or a comment
                return prevIndent

            return self._increaseIndent(prevIndent)
        elif char == '\n':
            if re.match(r'^<(\?xml|!DOCTYPE)', prevLineText):
                return ''
            elif re.search(r'<([^/!]|[^/!][^>]*[^/])>[^<>]*$', prevLineText):
                # increase indent when prev line opened a tag (but not for comments)
                return self._increaseIndent(prevIndent)

        return prevIndent
