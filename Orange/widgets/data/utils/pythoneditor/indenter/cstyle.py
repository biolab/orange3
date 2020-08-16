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

# User configuration
CFG_INDENT_CASE = True  # indent 'case' and 'default' in a switch?
CFG_INDENT_NAMESPACE = True  # indent after 'namespace'?
CFG_AUTO_INSERT_STAR = True  # auto insert '*' in C-comments
CFG_SNAP_SLASH = True  # snap '/' to '*/' in C-comments
CFG_AUTO_INSERT_SLACHES = False  # auto insert '//' after C++-comments
CFG_ACCESS_MODIFIERS = 1  # indent level of access modifiers, relative to the class indent level
                          # set to -1 to disable auto-indendation after access modifiers.

# indent gets three arguments: line, indentwidth in spaces, typed character
# indent

# specifies the characters which should trigger indent, beside the default '\n'

DEBUG_MODE = False

def dbg(*args):
    if (DEBUG_MODE):
        print(args)

#global variables and functions

INDENT_WIDTH = 4
MODE = "C"


class IndentAlgCStyle(IndentAlgBase):
    TRIGGER_CHARACTERS = "{})/:;#"

    @staticmethod
    def _prevNonEmptyBlock(block):
        """Reimplemented base indenter level. Skips comments
        """
        block = block.previous()
        while block.isValid() and \
              (len(block.text().strip()) == 0 or \
               block.text().startswith('//') or \
               block.text().startswith('#')):
            block = block.previous()

        return block

    def findTextBackward(self, block, column, needle):
        """Search for a needle and return (block, column)
        Raise ValueError, if not found
        """
        if column is not None:
            index = block.text()[:column].rfind(needle)
        else:
            index = block.text().rfind(needle)

        if index != -1:
            return block, index

        for block in self.iterateBlocksBackFrom(block.previous()):
            column = block.text().rfind(needle)
            if column != -1:
                return block, column

        raise ValueError('Not found')

    def findLeftBrace(self, block, column):
        """Search for a corresponding '{' and return its indentation
        If not found return None
        """
        block, column = self.findBracketBackward(block, column, '{')  # raise ValueError if not found

        try:
            block, column = self.tryParenthesisBeforeBrace(block, column)
        except ValueError:
            pass # leave previous values
        return self._blockIndent(block)

    def tryParenthesisBeforeBrace(self, block, column):
        """ Character at (block, column) has to be a '{'.
        Now try to find the right line for indentation for constructs like:
          if (a == b
              and c == d) { <- check for ')', and find '(', then return its indentation
        Returns input params, if no success, otherwise block and column of '('
        """
        text = block.text()[:column - 1].rstrip()
        if not text.endswith(')'):
            raise ValueError()
        return self.findBracketBackward(block, len(text) - 1, '(')

    def trySwitchStatement(self, block):
        """Check for default and case keywords and assume we are in a switch statement.
        Try to find a previous default, case or switch and return its indentation or
        None if not found.
        """
        if not re.match(r'^\s*(default\s*|case\b.*):', block.text()):
            return None

        for block in self.iterateBlocksBackFrom(block.previous()):
            text = block.text()
            if re.match(r"^\s*(default\s*|case\b.*):", text):
                dbg("trySwitchStatement: success in line %d" % block.blockNumber())
                return self._lineIndent(text)
            elif re.match(r"^\s*switch\b", text):
                if CFG_INDENT_CASE:
                    return self._increaseIndent(self._lineIndent(text))
                else:
                    return self._lineIndent(text)

        return None

    def tryAccessModifiers(self, block):
        """Check for private, protected, public, signals etc... and assume we are in a
        class definition. Try to find a previous private/protected/private... or
        class and return its indentation or null if not found.
        """

        if CFG_ACCESS_MODIFIERS < 0:
            return None

        if not re.match(r'^\s*((public|protected|private)\s*(slots|Q_SLOTS)?|(signals|Q_SIGNALS)\s*):\s*$', block.text()):
            return None

        try:
            block, notUsedColumn = self.findBracketBackward(block, 0, '{')
        except ValueError:
            return None

        indentation = self._blockIndent(block)
        for i in range(CFG_ACCESS_MODIFIERS):
            indentation = self._increaseIndent(indentation)

        dbg("tryAccessModifiers: success in line %d" % block.blockNumber())
        return indentation

    def tryCComment(self, block):
        """C comment checking. If the previous line begins with a "/*" or a "* ", then
        return its leading white spaces + ' *' + the white spaces after the *
        return: filler string or null, if not in a C comment
        """
        indentation = None

        prevNonEmptyBlock = self._prevNonEmptyBlock(block)
        if not prevNonEmptyBlock.isValid():
            return None

        prevNonEmptyBlockText = prevNonEmptyBlock.text()

        if prevNonEmptyBlockText.endswith('*/'):
            try:
                foundBlock, notUsedColumn = self.findTextBackward(prevNonEmptyBlock, prevNonEmptyBlock.length(), '/*')
            except ValueError:
                foundBlock = None

            if foundBlock is not None:
                dbg("tryCComment: success (1) in line %d" % foundBlock.blockNumber())
                return self._lineIndent(foundBlock.text())

        if prevNonEmptyBlock != block.previous():
            # inbetween was an empty line, so do not copy the "*" character
            return None

        blockTextStripped = block.text().strip()
        prevBlockTextStripped = prevNonEmptyBlockText.strip()

        if prevBlockTextStripped.startswith('/*') and not '*/' in prevBlockTextStripped:
            indentation = self._blockIndent(prevNonEmptyBlock)
            if CFG_AUTO_INSERT_STAR:
                # only add '*', if there is none yet.
                indentation += ' '
                if not blockTextStripped.endswith('*'):
                    indentation += '*'
                secondCharIsSpace = len(blockTextStripped) > 1 and blockTextStripped[1].isspace()
                if not secondCharIsSpace and \
                   not blockTextStripped.endswith("*/"):
                    indentation += ' '
            dbg("tryCComment: success (2) in line %d" % block.blockNumber())
            return indentation

        elif prevBlockTextStripped.startswith('*') and \
             (len(prevBlockTextStripped) == 1 or prevBlockTextStripped[1].isspace()):

            # in theory, we could search for opening /*, and use its indentation
            # and then one alignment character. Let's not do this for now, though.
            indentation = self._lineIndent(prevNonEmptyBlockText)
            # only add '*', if there is none yet.
            if CFG_AUTO_INSERT_STAR and not blockTextStripped.startswith('*'):
                indentation += '*'
                if len(blockTextStripped) < 2 or not blockTextStripped[1].isspace():
                    indentation += ' '

            dbg("tryCComment: success (2) in line %d" % block.blockNumber())
            return indentation

        return None

    def tryCppComment(self, block):
        """C++ comment checking. when we want to insert slashes:
        #, #/, #! #/<, #!< and ##...
        return: filler string or null, if not in a star comment
        NOTE: otherwise comments get skipped generally and we use the last code-line
        """
        if not block.previous().isValid() or \
           not CFG_AUTO_INSERT_SLACHES:
            return None

        prevLineText = block.previous().text()

        indentation = None
        comment = prevLineText.lstrip().startswith('#')

        # allowed are: #, #/, #! #/<, #!< and ##...
        if comment:
            prevLineText = block.previous().text()
            lstrippedText = block.previous().text().lstrip()
            if len(lstrippedText) >= 4:
                char3 = lstrippedText[2]
                char4 = lstrippedText[3]

            indentation = self._lineIndent(prevLineText)

            if CFG_AUTO_INSERT_SLACHES:
                if prevLineText[2:4] == '//':
                    # match ##... and replace by only two: #
                    match = re.match(r'^\s*(\/\/)', prevLineText)
                elif (char3 == '/' or char3 == '!'):
                    # match #/, #!, #/< and #!
                    match = re.match(r'^\s*(\/\/[\/!][<]?\s*)', prevLineText)
                else:
                    # only #, nothing else:
                    match = re.match(r'^\s*(\/\/\s*)', prevLineText)

                if match is not None:
                    self._qpart.insertText((block.blockNumber(), 0), match.group(1))

        if indentation is not None:
            dbg("tryCppComment: success in line %d" % block.previous().blockNumber())

        return indentation

    def tryBrace(self, block):
        def _isNamespace(block):
            if not block.text().strip():
                block = block.previous()

            return re.match(r'^\s*namespace\b', block.text()) is not None

        currentBlock = self._prevNonEmptyBlock(block)
        if not currentBlock.isValid():
            return None

        indentation = None

        if currentBlock.text().rstrip().endswith('{'):
            try:
                foundBlock, notUsedColumn = self.tryParenthesisBeforeBrace(currentBlock, len(currentBlock.text().rstrip()))
            except ValueError:  # not found
                indentation = self._blockIndent(currentBlock)
                if CFG_INDENT_NAMESPACE or not _isNamespace(block):
                    # take its indentation and add one indentation level
                    indentation = self._increaseIndent(indentation)
            else:  # found
                indentation = self._increaseIndent(self._blockIndent(foundBlock))


        if indentation is not None:
            dbg("tryBrace: success in line %d" % block.blockNumber())
        return indentation

    def tryCKeywords(self, block, isBrace):
        """
        Check for if, else, while, do, switch, private, public, protected, signals,
        default, case etc... keywords, as we want to indent then. If   is
        non-null/True, then indentation is not increased.
        Note: The code is written to be called *after* tryCComment and tryCppComment!
        """
        currentBlock = self._prevNonEmptyBlock(block)
        if not currentBlock.isValid():
            return None

        # if line ends with ')', find the '(' and check this line then.

        if currentBlock.text().rstrip().endswith(')'):
            try:
                foundBlock, foundColumn = self.findBracketBackward(currentBlock, len(currentBlock.text()), '(')
            except ValueError:
                pass
            else:
                currentBlock = foundBlock

        # found non-empty line
        currentBlockText = currentBlock.text()
        if re.match(r'^\s*(if\b|for|do\b|while|switch|[}]?\s*else|((private|public|protected|case|default|signals|Q_SIGNALS).*:))', currentBlockText) is None:
            return None

        indentation = None

        # ignore trailing comments see: https:#bugs.kde.org/show_bug.cgi?id=189339
        try:
            index = currentBlockText.index('//')
        except ValueError:
            pass
        else:
            currentBlockText = currentBlockText[:index]

        # try to ignore lines like: if (a) b; or if (a) { b; }
        if not currentBlockText.endswith(';') and \
           not currentBlockText.endswith('}'):
            # take its indentation and add one indentation level
            indentation = self._lineIndent(currentBlockText)
            if not isBrace:
                indentation = self._increaseIndent(indentation)
        elif currentBlockText.endswith(';'):
            # stuff like:
            # for(int b;
            #     b < 10;
            #     --b)
            try:
                foundBlock, foundColumn = self.findBracketBackward(currentBlock, None, '(')
            except ValueError:
                pass
            else:
                dbg("tryCKeywords: success 1 in line %d" % block.blockNumber())
                return self._makeIndentAsColumn(foundBlock, foundColumn, 1)
        if indentation is not None:
            dbg("tryCKeywords: success in line %d" % block.blockNumber())

        return indentation

    def tryCondition(self, block):
        """ Search for if, do, while, for, ... as we want to indent then.
        Return null, if nothing useful found.
        Note: The code is written to be called *after* tryCComment and tryCppComment!
        """
        currentBlock = self._prevNonEmptyBlock(block)
        if not currentBlock.isValid():
            return None

        # found non-empty line
        currentText = currentBlock.text()
        if currentText.rstrip().endswith(';') and \
           re.search(r'^\s*(if\b|[}]?\s*else|do\b|while\b|for)', currentText) is None:
            # idea: we had something like:
            #   if/while/for (expression)
            #       statement();  <-- we catch this trailing ';'
            # Now, look for a line that starts with if/for/while, that has one
            # indent level less.
            currentIndentation = self._lineIndent(currentText)
            if not currentIndentation:
                return None

            for block in self.iterateBlocksBackFrom(currentBlock.previous()):
                if block.text().strip(): # not empty
                    indentation = self._blockIndent(block)

                    if len(indentation) < len(currentIndentation):
                        if re.search(r'^\s*(if\b|[}]?\s*else|do\b|while\b|for)[^{]*$', block.text()) is not None:
                            dbg("tryCondition: success in line %d" % block.blockNumber())
                            return indentation
                        break

        return None

    def tryStatement(self, block):
        """ If the non-empty line ends with ); or ',', then search for '(' and return its
        indentation; also try to ignore trailing comments.
        """
        currentBlock = self._prevNonEmptyBlock(block)

        if not currentBlock.isValid():
            return None

        indentation = None

        currentBlockText = currentBlock.text()
        if currentBlockText.endswith('('):
            # increase indent level
            dbg("tryStatement: success 1 in line %d" % block.blockNumber())
            return self._increaseIndent(self._lineIndent(currentBlockText))

        alignOnSingleQuote = self._qpart.language() in ('PHP/PHP', 'JavaScript')
        # align on strings "..."\n => below the opening quote
        # multi-language support: [\.+] for javascript or php
        pattern =  '^(.*)'                   # any                                                  group 1
        pattern += '([,"\'\\)])'             # one of [ , " ' )                                     group 2
        pattern += '(;?)'                    # optional ;                                           group 3
        pattern += '\s*[\.+]?\s*'            # optional spaces  optional . or +   optional spaces
        pattern += '(//.*|/\\*.*\\*/\s*)?$'  # optional(//any  or  /*any*/spaces)                   group 4
        match = re.match(pattern, currentBlockText)
        if match is not None:
            alignOnAnchor = len(match.group(3)) == 0 and match.group(2) != ')'
            # search for opening ", ' or (
            if match.group(2) == '"' or (alignOnSingleQuote and match.group(2) == "'"):
                startIndex = len(match.group(1))
                while True:
                    # start from matched closing ' or "
                    # find string opener
                    for i in range(startIndex - 1, 0, -1):
                        # make sure it's not commented out
                        if currentBlockText[i] == match.group(2) and (i == 0 or currentBlockText[i - 1] != '\\'):
                            # also make sure that this is not a line like '#include "..."' <-- we don't want to indent here
                            if re.match(r'^#include', currentBlockText):
                                dbg("tryStatement: success 2 in line %d" % block.blockNumber())
                                return indentation

                            break

                    if not alignOnAnchor and currentBlock.previous().isValid():
                        # when we finished the statement (;) we need to get the first line and use it's indentation
                        # i.e.: $foo = "asdf"; -> align on $
                        i -= 1 # skip " or '
                        # skip whitespaces and stuff like + or . (for PHP, JavaScript, ...)
                        for i in range(i, 0, -1):
                            if currentBlockText[i] in (' ', '\t', '.', '+'):
                                continue
                            else:
                                break

                        if i > 0:
                            # there's something in this line, use it's indentation
                            break
                        else:
                            # go to previous line
                            currentBlock = currentBlock.previous()
                            currentBlockText = currentBlock.text()
                            startIndex = len(currentBlockText)
                    else:
                        break

            elif match.group(2) == ',' and not '(' in currentBlockText:
                # assume a function call: check for '(' brace
                # - if not found, use previous indentation
                # - if found, compare the indentation depth of current line and open brace line
                #   - if current indentation depth is smaller, use that
                #   - otherwise, use the '(' indentation + following white spaces
                currentIndentation = self._blockIndent(currentBlock)
                try:
                    foundBlock, foundColumn = self.findBracketBackward(currentBlock, len(match.group(1)), '(')
                except ValueError:
                    indentation = currentIndentation
                else:
                    indentWidth = foundColumn + 1
                    text = foundBlock.text()
                    while indentWidth < len(text) and text[indentWidth].isspace():
                        indentWidth += 1
                    indentation = self._makeIndentAsColumn(foundBlock, indentWidth)

            else:
                try:
                    foundBlock, foundColumn = self.findBracketBackward(currentBlock, len(match.group(1)), '(')
                except ValueError:
                    pass
                else:
                    if alignOnAnchor:
                        if not match.group(2) in ('"', "'"):
                            foundColumn += 1
                        foundBlockText = foundBlock.text()
                        while foundColumn < len(foundBlockText) and \
                              foundBlockText[foundColumn].isspace():
                            foundColumn += 1
                        indentation = self._makeIndentAsColumn(foundBlock, foundColumn)
                    else:
                        currentBlock = foundBlock
                        indentation = self._blockIndent(currentBlock)
        elif currentBlockText.rstrip().endswith(';'):
            indentation = self._blockIndent(currentBlock)

        if indentation is not None:
            dbg("tryStatement: success in line %d" % currentBlock.blockNumber())
        return indentation

    def tryMatchedAnchor(self, block, autoIndent):
        """
        find out whether we pressed return in something like {} or () or [] and indent properly:
         {}
         becomes:
         {
           |
         }
        """
        oposite = { ')': '(',
                    '}': '{',
                    ']': '['}

        char = self._firstNonSpaceChar(block)
        if not char in oposite.keys():
            return None

        # we pressed enter in e.g. ()
        try:
            foundBlock, foundColumn = self.findBracketBackward(block, 0, oposite[char])
        except ValueError:
            return None

        if autoIndent:
            # when aligning only, don't be too smart and just take the indent level of the open anchor
            return self._blockIndent(foundBlock)

        lastChar = self._lastNonSpaceChar(block.previous())
        charsMatch = ( lastChar == '(' and char == ')' ) or \
                     ( lastChar == '{' and char == '}' ) or \
                     ( lastChar == '[' and char == ']' )

        indentation = None
        if (not charsMatch) and char != '}':
            # otherwise check whether the last line has the expected
            # indentation, if not use it instead and place the closing
            # anchor on the level of the opening anchor
            expectedIndentation = self._increaseIndent(self._blockIndent(foundBlock))
            actualIndentation = self._increaseIndent(self._blockIndent(block.previous()))
            indentation = None
            if len(expectedIndentation) <= len(actualIndentation):
                if lastChar == ',':
                    # use indentation of last line instead and place closing anchor
                    # in same column of the opening anchor
                    self._qpart.insertText((block.blockNumber(), self._firstNonSpaceColumn(block.text())), '\n')
                    self._qpart.cursorPosition = (block.blockNumber(), len(actualIndentation))
                    # indent closing anchor
                    self._setBlockIndent(block.next(), self._makeIndentAsColumn(foundBlock, foundColumn))
                    indentation = actualIndentation
                elif expectedIndentation == self._blockIndent(block.previous()):
                    # otherwise don't add a new line, just use indentation of closing anchor line
                    indentation = self._blockIndent(foundBlock)
                else:
                    # otherwise don't add a new line, just align on closing anchor
                    indentation = self._makeIndentAsColumn(foundBlock, foundColumn)

                dbg("tryMatchedAnchor: success in line %d" % foundBlock.blockNumber())
                return indentation

        # otherwise we i.e. pressed enter between (), [] or when we enter before curly brace
        # increase indentation and place closing anchor on the next line
        indentation = self._blockIndent(foundBlock)
        self._qpart.replaceText((block.blockNumber(), 0), len(self._blockIndent(block)), "\n")
        self._qpart.cursorPosition = (block.blockNumber(), len(indentation))
        # indent closing brace
        self._setBlockIndent(block.next(), indentation)
        dbg("tryMatchedAnchor: success in line %d" % foundBlock.blockNumber())
        return self._increaseIndent(indentation)

    def indentLine(self, block, autoIndent):
        """ Indent line.
        Return filler or null.
        """
        indent = None
        if indent is None:
            indent = self.tryMatchedAnchor(block, autoIndent)
        if indent is None:
            indent = self.tryCComment(block)
        if indent is None and not autoIndent:
            indent = self.tryCppComment(block)
        if indent is None:
            indent = self.trySwitchStatement(block)
        if indent is None:
            indent = self.tryAccessModifiers(block)
        if indent is None:
            indent = self.tryBrace(block)
        if indent is None:
            indent = self.tryCKeywords(block, block.text().lstrip().startswith('{'))
        if indent is None:
            indent = self.tryCondition(block)
        if indent is None:
            indent = self.tryStatement(block)

        if indent is not None:
            return indent
        else:
            dbg("Nothing matched")
            return self._prevNonEmptyBlockIndent(block)

    def processChar(self, block, c):
        if c == ';' or (not (c in self.TRIGGER_CHARACTERS)):
            return self._blockIndent(block)

        column = self._qpart.cursorPosition[1]
        blockIndent = self._blockIndent(block)
        firstCharAfterIndent = column == (len(blockIndent) + 1)

        if firstCharAfterIndent and c == '{':
            # todo: maybe look for if etc.
            indent = self.tryBrace(block)
            if indent is None:
                indent = self.tryCKeywords(block, True)
            if indent is None:
                indent = self.tryCComment(block); # checks, whether we had a "*/"
            if indent is None:
                indent = self.tryStatement(block)
            if indent is None:
                indent = blockIndent

            return indent
        elif firstCharAfterIndent and c == '}':
            try:
                indentation = self.findLeftBrace(block, self._firstNonSpaceColumn(block.text()))
            except ValueError:
                return blockIndent
            else:
                return indentation
        elif CFG_SNAP_SLASH and c == '/' and block.text().endswith(' /'):
            # try to snap the string "* /" to "*/"
            match = re.match(r'^(\s*)\*\s+\/\s*$', block.text())
            if match is not None:
                self._qpart.lines[block.blockNumber()] = match.group(1) + '*/'
            dbg("snapSlash at block %d" % block.blockNumber())
            return blockIndent
        elif c == ':':
            # todo: handle case, default, signals, private, public, protected, Q_SIGNALS
            indent = self.trySwitchStatement(block)
            if indent is None:
                indent = self.tryAccessModifiers(block)
            if indent is None:
                indent = blockIndent
            return indent
        elif c == ')' and firstCharAfterIndent:
            # align on start of identifier of function call
            try:
                foundBlock, foundColumn = self.findBracketBackward(block, column - 1, '(')
            except ValueError:
                pass
            else:
                text = foundBlock.text()[:foundColumn]
                match  = re.search(r'\b(\w+)\s*$', text)
                if match is not None:
                    return self._makeIndentAsColumn(foundBlock, match.start())
        elif firstCharAfterIndent and c == '#' and self._qpart.language() in ('C', 'C++'):
            # always put preprocessor stuff upfront
            return ''
        return blockIndent

    def computeSmartIndent(self, block, char):
        autoIndent = char == ""

        if char != '\n' and not autoIndent:
            return self.processChar(block, char)

        return self.indentLine(block, autoIndent)
