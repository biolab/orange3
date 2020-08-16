"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Module computes indentation for block
It contains implementation of indenters, which are supported by katepart xml files
"""

import logging

logger = logging.getLogger('qutepart')


from PyQt5.QtGui import QTextCursor


def _getSmartIndenter(indenterName, qpart, indenter):
    """Get indenter by name.
    Available indenters are none, normal, cstyle, haskell, lilypond, lisp, python, ruby, xml
    Indenter name is not case sensitive
    Raise KeyError if not found
    indentText is indentation, which shall be used. i.e. '\t' for tabs, '    ' for 4 space symbols
    """
    indenterName = indenterName.lower()

    if indenterName in ('haskell', 'lilypond'):  # not supported yet
        logger.warning('Smart indentation for %s not supported yet. But you could be a hero who implemented it' % indenterName)
        from qutepart.indenter.base import IndentAlgNormal as indenterClass
    elif 'none' == indenterName:
        from qutepart.indenter.base import IndentAlgBase as indenterClass
    elif 'normal' == indenterName:
        from qutepart.indenter.base import IndentAlgNormal as indenterClass
    elif 'cstyle' == indenterName:
        from qutepart.indenter.cstyle import IndentAlgCStyle as indenterClass
    elif 'python' == indenterName:
        from qutepart.indenter.python import IndentAlgPython as indenterClass
    elif 'ruby' == indenterName:
        from qutepart.indenter.ruby import IndentAlgRuby as indenterClass
    elif 'xml' == indenterName:
        from qutepart.indenter.xmlindent import IndentAlgXml as indenterClass
    elif 'haskell' == indenterName:
        from qutepart.indenter.haskell import IndenterHaskell as indenterClass
    elif 'lilypond' == indenterName:
        from qutepart.indenter.lilypond import IndenterLilypond as indenterClass
    elif 'lisp' == indenterName:
        from qutepart.indenter.lisp import IndentAlgLisp as indenterClass
    elif 'scheme' == indenterName:
        from qutepart.indenter.scheme import IndentAlgScheme as indenterClass
    else:
        raise KeyError("Indenter %s not found" % indenterName)

    return indenterClass(qpart, indenter)


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

        self._smartIndenter = _getSmartIndenter('normal', self._qpart, self)

    def setSyntax(self, syntax):
        """Choose smart indentation algorithm according to syntax"""
        self._smartIndenter = self._chooseSmartIndenter(syntax)

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
        if(cursor.selectionStart() != cursor.selectionEnd() and
           endBlock.position() == cursor.selectionEnd() and
           endBlock.previous().isValid()):
            endBlock = endBlock.previous()  # do not indent not selected line if indenting multiple lines

        indentFunc = indentBlock if increase else unIndentBlock

        if startBlock != endBlock:  # indent multiply lines
            stopBlock = endBlock.next()

            block = startBlock

            with self._qpart:
                while block != stopBlock:
                    indentFunc(block)
                    block = block.next()

            newCursor = QTextCursor(startBlock)
            newCursor.setPosition(endBlock.position() + len(endBlock.text()), QTextCursor.KeepAnchor)
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

    def _chooseSmartIndenter(self, syntax):
        """Get indenter for syntax
        """
        if syntax.indenter is not None:
            try:
                return _getSmartIndenter(syntax.indenter, self._qpart, self)
            except KeyError:
                logger.error("Indenter '%s' is not finished yet. But you can do it!" % syntax.indenter)

        try:
            return _getSmartIndenter(syntax.name, self._qpart, self)
        except KeyError:
            pass

        return _getSmartIndenter('normal', self._qpart, self)
