"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Source file parser and highlighter
"""

import os.path
import fnmatch
import json
import threading
import logging
import re

_logger = logging.getLogger('qutepart')

class TextFormat:
    """Text format definition.

    Public attributes:
        color          : Font color, #rrggbb or #rgb
        background     : Font background, #rrggbb or #rgb
        selectionColor : Color of selected text
        italic         : Italic font, bool
        bold           : Bold font, bool
        underline      : Underlined font, bool
        strikeOut      : Striked out font
        spellChecking  : Text will be spell checked
        textType       : 'c' for comments, 's' for strings, ' ' for other.
    """
    def __init__(self, color = '#000000',
                       background = '#ffffff',
                       selectionColor = '#0000ff',
                       italic = False,
                       bold = False,
                       underline = False,
                       strikeOut = False,
                       spellChecking = False):

        self.color = color
        self.background = background
        self.selectionColor = selectionColor
        self.italic = italic
        self.bold = bold
        self.underline = underline
        self.strikeOut = strikeOut
        self.spellChecking = spellChecking
        self.textType = ' '  # modified later

    def __cmp__(self, other):
        return cmp(self.__dict__, other.__dict__)


class Syntax:
    """Syntax. Programming language parser definition

    Public attributes:
        name            Name
        section         Section
        extensions      File extensions
        mimetype        File mime type
        version         XML definition version
        kateversion     Required Kate parser version
        priority        XML definition priority
        author          Author
        license         License
        hidden          Shall be hidden in the menu
        indenter        Indenter for the syntax. Possible values are
                            none, normal, cstyle, haskell, lilypond, lisp, python, ruby, xml
                        None, if not set by xml file
    """
    def __init__(self, manager):
        self.manager = manager
        self.parser = None

    def __str__(self):
        res = 'Syntax\n'
        res += ' name: %s\n' % self.name
        res += ' section: %s\n' % self.section
        res += ' extensions: %s\n' % self.extensions
        res += ' mimetype: %s\n' % self.mimetype
        res += ' version: %s\n' % self.version
        res += ' kateversion: %s\n' % self.kateversion
        res += ' priority: %s\n' % self.priority
        res += ' author: %s\n' % self.author
        res += ' license: %s\n' % self.license
        res += ' hidden: %s\n' % self.hidden
        res += ' indenter: %s\n' % self.indenter
        res += str(self.parser)

        return res

    def _setParser(self, parser):
        self.parser = parser
        # performance optimization, avoid 1 function call
        self.highlightBlock = parser.highlightBlock
        self.parseBlock = parser.parseBlock

    def highlightBlock(self, text, prevLineData):
        """Parse line of text and return
            (lineData, highlightedSegments)
        where
            lineData is data, which shall be saved and used for parsing next line
            highlightedSegments is list of touples (segmentLength, segmentFormat)
        """
        #self.parser.parseAndPrintBlockTextualResults(text, prevLineData)
        return self.parser.highlightBlock(text, prevLineData)

    def parseBlock(self, text, prevLineData):
        """Parse line of text and return
            lineData
        where
            lineData is data, which shall be saved and used for parsing next line

        This is quicker version of highlighBlock, which doesn't return results,
        but only parsers the block and produces data, which is necessary for parsing next line.
        Use it for invisible lines
        """
        return self.parser.parseBlock(text, prevLineData)

    def _getTextType(self, lineData, column):
        """Get text type (letter)
        """
        if lineData is None:
            return ' '  # default is code

        textTypeMap = lineData[1]
        if column >= len(textTypeMap):  # probably, not actual data, not updated yet
            return ' '

        return textTypeMap[column]

    def isCode(self, lineData, column):
        """Check if text at given position is a code
        """
        return self._getTextType(lineData, column) ==  ' '

    def isComment(self, lineData, column):
        """Check if text at given position is a comment. Including block comments and here documents
        """
        return self._getTextType(lineData, column) in 'cbh'

    def isBlockComment(self, lineData, column):
        """Check if text at given position is a block comment
        """
        return self._getTextType(lineData, column) ==  'b'

    def isHereDoc(self, lineData, column):
        """Check if text at given position is a here document
        """
        return self._getTextType(lineData, column) ==  'h'


class SyntaxManager:
    """SyntaxManager holds references to loaded Syntax'es and allows to find or
    load Syntax by its name or by source file name
    """
    def __init__(self):
        self._loadedSyntaxesLock = threading.RLock()
        self._loadedSyntaxes = {}
        syntaxDbPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "syntax_db.json")
        with open(syntaxDbPath, encoding='utf-8') as syntaxDbFile:
            syntaxDb = json.load(syntaxDbFile)
        self._syntaxNameToXmlFileName = syntaxDb['syntaxNameToXmlFileName']
        self._mimeTypeToXmlFileName = syntaxDb['mimeTypeToXmlFileName']
        self._firstLineToXmlFileName = syntaxDb['firstLineToXmlFileName']
        globToXmlFileName = syntaxDb['extensionToXmlFileName']

        # Applying glob patterns is really slow. Therefore they are compiled to reg exps
        self._extensionToXmlFileName = \
                {re.compile(fnmatch.translate(glob)): xmlFileName \
                        for glob, xmlFileName in globToXmlFileName.items()}

    def _getSyntaxByXmlFileName(self, xmlFileName):
        """Get syntax by its xml file name
        """
        import qutepart.syntax.loader  # delayed import for avoid cross-imports problem

        with self._loadedSyntaxesLock:
            if not xmlFileName in self._loadedSyntaxes:
                xmlFilePath = os.path.join(os.path.dirname(__file__), "data", "xml", xmlFileName)
                syntax = Syntax(self)
                self._loadedSyntaxes[xmlFileName] = syntax
                qutepart.syntax.loader.loadSyntax(syntax, xmlFilePath)

            return self._loadedSyntaxes[xmlFileName]

    def _getSyntaxByLanguageName(self, syntaxName):
        """Get syntax by its name. Name is defined in the xml file
        """
        xmlFileName = self._syntaxNameToXmlFileName[syntaxName]
        return self._getSyntaxByXmlFileName(xmlFileName)

    def _getSyntaxBySourceFileName(self, name):
        """Get syntax by source name of file, which is going to be highlighted
        """
        for regExp, xmlFileName in self._extensionToXmlFileName.items():
            if regExp.match(name):
                return self._getSyntaxByXmlFileName(xmlFileName)
        else:
            raise KeyError("No syntax for " + name)

    def _getSyntaxByMimeType(self, mimeType):
        """Get syntax by first line of the file
        """
        xmlFileName = self._mimeTypeToXmlFileName[mimeType]
        return self._getSyntaxByXmlFileName(xmlFileName)

    def _getSyntaxByFirstLine(self, firstLine):
        """Get syntax by first line of the file
        """
        for pattern, xmlFileName in self._firstLineToXmlFileName.items():
            if fnmatch.fnmatch(firstLine, pattern):
                return self._getSyntaxByXmlFileName(xmlFileName)
        else:
            raise KeyError("No syntax for " + firstLine)

    def getSyntax(self,
                  xmlFileName=None,
                  mimeType=None,
                  languageName=None,
                  sourceFilePath=None,
                  firstLine=None):
        """Get syntax by one of parameters:
            * xmlFileName
            * mimeType
            * languageName
            * sourceFilePath
        First parameter in the list has biggest priority
        """
        syntax = None

        if syntax is None and xmlFileName is not None:
            try:
                syntax = self._getSyntaxByXmlFileName(xmlFileName)
            except KeyError:
                _logger.warning('No xml definition %s' % xmlFileName)

        if syntax is None and mimeType is not None:
            try:
                syntax = self._getSyntaxByMimeType(mimeType)
            except KeyError:
                _logger.warning('No syntax for mime type %s' % mimeType)

        if syntax is None and languageName is not None:
            try:
                syntax = self._getSyntaxByLanguageName(languageName)
            except KeyError:
                _logger.warning('No syntax for language %s' % languageName)

        if syntax is None and sourceFilePath is not None:
            baseName = os.path.basename(sourceFilePath)
            try:
                syntax = self._getSyntaxBySourceFileName(baseName)
            except KeyError:
                pass

        if syntax is None and firstLine is not None:
            try:
                syntax = self._getSyntaxByFirstLine(firstLine)
            except KeyError:
                pass

        return syntax
