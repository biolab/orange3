"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""This module is a set of functions, which load Parser from Kate XML files
"""

import os
import copy
import sys
import xml.etree.ElementTree
import re
import logging

from qutepart.syntax.colortheme import ColorTheme
from qutepart.syntax import TextFormat

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QFont


_logger = logging.getLogger('qutepart')

# See the Qutepart.__init__() documentation about the syntax parser selection
binaryParserAvailable = False
if os.environ.get('QPART_CPARSER', 'Y') in ('Y', 'y', '1'):
    try:
        import qutepart.syntax.cParser as _parserModule
        binaryParserAvailable = True
    except ImportError:
        _logger.warning('Failed to import quick parser in C. Using slow parser for syntax highlighting')

if binaryParserAvailable == False:
    import qutepart.syntax.parser as _parserModule


_seqReplacer = re.compile('\\\\.')

_escapeSequences = \
{  '\\': '\\',
  'a': '\a',
  'b': '\b',
  'f': '\f',
  'n': '\n',
  'r': '\r',
  't': '\t',}


def _processEscapeSequences(replaceText):
    """Replace symbols like \n \\, etc
    """
    def _replaceFunc(escapeMatchObject):
        char = escapeMatchObject.group(0)[1]
        if char in _escapeSequences:
            return _escapeSequences[char]

        return escapeMatchObject.group(0)  # no any replacements, return original value

    return _seqReplacer.sub(_replaceFunc, replaceText)


def _convertFormat(format):
    if format == TextFormat():
        return None  # Do not apply default format. Performance optimization

    qtFormat = QTextCharFormat()

    foregroundColor = QColor(format.color)
    backgroundColor = QColor(format.background)

    if foregroundColor != Qt.black:
        qtFormat.setForeground(foregroundColor)

    if backgroundColor != Qt.white:
        qtFormat.setBackground(backgroundColor)

    qtFormat.setFontItalic(format.italic)
    qtFormat.setFontWeight(QFont.Bold if format.bold else QFont.Normal)
    qtFormat.setFontUnderline(format.underline)
    qtFormat.setFontStrikeOut(format.strikeOut)

    return qtFormat


_DEFAULT_ATTRIBUTE_TO_STYLE_MAP = \
{
    'alert' : 'dsAlert',
    'base-n integer' : 'dsBaseN',
    'character' : 'dsChar',
    'string char': 'dsChar',
    'comment' : 'dsComment',
    'data type' : 'dsDataType',
    'decimal/value' : 'dsDecVal',
    'error' : 'dsError',
    'floating point' : 'dsFloat',
    'function' : 'dsFunction',
    'keyword' : 'dsKeyword',
    'normal' : 'dsNormal',
    'others' : 'dsOthers',
    'region marker' : 'dsRegionMarker',
    'string' : 'dsString'
}

_KNOWN_STYLES = set([  "dsNormal",
                       "dsKeyword",
                       "dsDataType",
                       "dsDecVal",
                       "dsBaseN",
                       "dsFloat",
                       "dsChar",
                       "dsString",
                       "dsComment",
                       "dsOthers",
                       "dsAlert",
                       "dsFunction",
                       "dsRegionMarker",
                       "dsError",
                       "CustomTmpForDebugging"])

_DEFAULT_DELIMINATOR = " \t.():!+,-<=>%&*/;?[]^{|}~\\"

def _parseBoolAttribute(value):
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    else:
        raise UserWarning("Invalid bool attribute value '%s'" % value)

def _safeGetRequiredAttribute(xmlElement, name, default):
    if name in xmlElement.attrib:
        return str(xmlElement.attrib[name])
    else:
        _logger.warning("Required attribute '%s' is not set for element '%s'", name, xmlElement.tag)
        return default


def _getContext(contextName, parser, defaultValue):
    if not contextName:
        return defaultValue
    if contextName in parser.contexts:
        return parser.contexts[contextName]
    elif contextName.startswith('##') and \
         parser.syntax.manager is not None:  # might be None, if loader is used by regenerate-definitions-db.py
        syntaxName = contextName[2:]
        parser = parser.syntax.manager.getSyntax(languageName = syntaxName).parser
        return parser.defaultContext
    elif (not contextName.startswith('##')) and \
         '##' in contextName and \
         contextName.count('##') == 1 and \
         parser.syntax.manager is not None:  # might be None, if loader is used by regenerate-definitions-db.py
        name, syntaxName = contextName.split('##')
        parser = parser.syntax.manager.getSyntax(languageName = syntaxName).parser
        return parser.contexts[name]
    else:
        _logger.warning('Invalid context name %s', repr(contextName))
        return parser.defaultContext


def _makeContextSwitcher(contextOperation, parser):
    popsCount = 0
    contextToSwitch = None

    rest = contextOperation
    while rest.startswith('#pop'):
        popsCount += 1
        rest = rest[len('#pop'):]
        if rest.startswith('!'):
            rest = rest[1:]

    if rest == '#stay':
        if popsCount:
            _logger.warning("Invalid context operation '%s'", contextOperation)
    else:
        contextToSwitch = _getContext(rest, parser, None)

    if popsCount > 0 or contextToSwitch != None:
        return _parserModule.ContextSwitcher(popsCount, contextToSwitch, contextOperation)
    else:
        return None


################################################################################
##                               Rules
################################################################################

def _loadIncludeRules(parentContext, xmlElement, attributeToFormatMap):
    contextName = _safeGetRequiredAttribute(xmlElement, "context", None)

    context = _getContext(contextName, parentContext.parser, parentContext.parser.defaultContext)

    abstractRuleParams = _loadAbstractRuleParams(parentContext,
                                                 xmlElement,
                                                 attributeToFormatMap)
    return _parserModule.IncludeRules(abstractRuleParams, context)

def _simpleLoader(classObject):
    def _load(parentContext, xmlElement, attributeToFormatMap):
        abstractRuleParams = _loadAbstractRuleParams(parentContext,
                                                     xmlElement,
                                                     attributeToFormatMap)
        return classObject(abstractRuleParams)
    return _load

def _loadChildRules(context, xmlElement, attributeToFormatMap):
    """Extract rules from Context or Rule xml element
    """
    rules = []
    for ruleElement in xmlElement.getchildren():
        if not ruleElement.tag in _ruleClassDict:
            raise ValueError("Not supported rule '%s'" % ruleElement.tag)
        rule = _ruleClassDict[ruleElement.tag](context, ruleElement, attributeToFormatMap)
        rules.append(rule)
    return rules

def _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap):
    # attribute
    attribute = xmlElement.attrib.get("attribute", None)
    if attribute is not None:
        attribute = attribute.lower()  # not case sensitive
        try:
            format = attributeToFormatMap[attribute]
            textType = format.textType if format is not None else ' '
            if format is not None:
                format = _convertFormat(format)
        except KeyError:
            _logger.warning('Unknown rule attribute %s', attribute)
            format = parentContext.format
            textType = parentContext.textType
    else:
        format = None
        textType = None

    # context
    contextText = xmlElement.attrib.get("context", '#stay')
    context = _makeContextSwitcher(contextText, parentContext.parser)

    lookAhead = _parseBoolAttribute(xmlElement.attrib.get("lookAhead", "false"))
    firstNonSpace = _parseBoolAttribute(xmlElement.attrib.get("firstNonSpace", "false"))
    dynamic = _parseBoolAttribute(xmlElement.attrib.get("dynamic", "false"))

    # TODO beginRegion
    # TODO endRegion

    column = xmlElement.attrib.get("column", None)
    if column is not None:
        column = int(column)
    else:
        column = -1

    return _parserModule.AbstractRuleParams(parentContext, format, textType, attribute, context, lookAhead, firstNonSpace, dynamic, column)

def _loadDetectChar(parentContext, xmlElement, attributeToFormatMap):
    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)

    char = _safeGetRequiredAttribute(xmlElement, "char", None)
    if char is not None:
        char = _processEscapeSequences(char)

    index = 0
    if abstractRuleParams.dynamic:
        try:
            index = int(char)
        except ValueError:
            _logger.warning('Invalid DetectChar char %s', char)
            index = 0
        char = None
        if index <= 0:
            _logger.warning('Too little DetectChar index %d', index)
            index = 0

    return _parserModule.DetectChar(abstractRuleParams, str(char), index)

def _loadDetect2Chars(parentContext, xmlElement, attributeToFormatMap):
    char = _safeGetRequiredAttribute(xmlElement, 'char', None)
    char1 = _safeGetRequiredAttribute(xmlElement, 'char1', None)
    if char is None or char1 is None:
        string = None
    else:
        string = _processEscapeSequences(char) + _processEscapeSequences(char1)

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.Detect2Chars(abstractRuleParams, string)

def _loadAnyChar(parentContext, xmlElement, attributeToFormatMap):
    string = _safeGetRequiredAttribute(xmlElement, 'String', '')
    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.AnyChar(abstractRuleParams, string)

def _loadStringDetect(parentContext, xmlElement, attributeToFormatMap):
    string = _safeGetRequiredAttribute(xmlElement, 'String', None)

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.StringDetect(abstractRuleParams,
                                      string)

def _loadWordDetect(parentContext, xmlElement, attributeToFormatMap):
    word = _safeGetRequiredAttribute(xmlElement, "String", "")
    insensitive = _parseBoolAttribute(xmlElement.attrib.get("insensitive", "false"))

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)

    return _parserModule.WordDetect(abstractRuleParams, word, insensitive)

def _loadKeyword(parentContext, xmlElement, attributeToFormatMap):
    string = _safeGetRequiredAttribute(xmlElement, 'String', None)
    try:
        words = parentContext.parser.lists[string]
    except KeyError:
        _logger.warning("List '%s' not found", string)

        words = list()

    insensitive = _parseBoolAttribute(xmlElement.attrib.get("insensitive", "false"))

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.keyword(abstractRuleParams, words, insensitive)

def _loadRegExpr(parentContext, xmlElement, attributeToFormatMap):
    def _processCraracterCodes(text):
        """QRegExp use \0ddd notation for character codes, where d in octal digit
        i.e. \0377 is character with code 255 in the unicode table
        Convert such notation to unicode text
        """
        text = str(text)
        def replFunc(matchObj):
            matchText = matchObj.group(0)
            charCode = eval('0o' + matchText[2:])
            return chr(charCode)
        return re.sub(r"\\0\d\d\d", replFunc, text)

    insensitive = _parseBoolAttribute(xmlElement.attrib.get('insensitive', 'false'))
    minimal = _parseBoolAttribute(xmlElement.attrib.get('minimal', 'false'))
    string = _safeGetRequiredAttribute(xmlElement, 'String', None)

    if string is not None:
        string = _processCraracterCodes(string)

        strippedString = string.strip('(')

        wordStart = strippedString.startswith('\\b')
        lineStart = strippedString.startswith('^')
        if len(strippedString) > 1 and strippedString[1] == '|':  # ^|blabla   This condition is not ideal but will cover majority of cases
            wordStart = False
            lineStart = False
    else:
        wordStart = False
        lineStart = False

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.RegExpr(abstractRuleParams,
                                 string, insensitive, minimal, wordStart, lineStart)

def _loadAbstractNumberRule(rule, parentContext, xmlElement):
    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.NumberRule(abstractRuleParams, childRules)

def _loadInt(parentContext, xmlElement, attributeToFormatMap):
    childRules = _loadChildRules(parentContext, xmlElement, attributeToFormatMap)
    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.Int(abstractRuleParams, childRules)

def _loadFloat(parentContext, xmlElement, attributeToFormatMap):
    childRules = _loadChildRules(parentContext, xmlElement, attributeToFormatMap)
    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.Float(abstractRuleParams, childRules)

def _loadRangeDetect(parentContext, xmlElement, attributeToFormatMap):
    char = _safeGetRequiredAttribute(xmlElement, "char", 'char is not set')
    char1 = _safeGetRequiredAttribute(xmlElement, "char1", 'char1 is not set')

    abstractRuleParams = _loadAbstractRuleParams(parentContext, xmlElement, attributeToFormatMap)
    return _parserModule.RangeDetect(abstractRuleParams, char, char1)


_ruleClassDict = \
{
    'DetectChar': _loadDetectChar,
    'Detect2Chars': _loadDetect2Chars,
    'AnyChar': _loadAnyChar,
    'StringDetect': _loadStringDetect,
    'WordDetect': _loadWordDetect,
    'RegExpr': _loadRegExpr,
    'keyword': _loadKeyword,
    'Int': _loadInt,
    'Float': _loadFloat,
    'HlCOct': _simpleLoader(_parserModule.HlCOct),
    'HlCHex': _simpleLoader(_parserModule.HlCHex),
    'HlCStringChar': _simpleLoader(_parserModule.HlCStringChar),
    'HlCChar': _simpleLoader(_parserModule.HlCChar),
    'RangeDetect': _loadRangeDetect,
    'LineContinue': _simpleLoader(_parserModule.LineContinue),
    'IncludeRules': _loadIncludeRules,
    'DetectSpaces': _simpleLoader(_parserModule.DetectSpaces),
    'DetectIdentifier': _simpleLoader(_parserModule.DetectIdentifier)
}

################################################################################
##                               Context
################################################################################


def _loadContexts(highlightingElement, parser, attributeToFormatMap):
    contextsElement = highlightingElement.find('contexts')

    xmlElementList = contextsElement.findall('context')
    contextList = []
    for xmlElement in xmlElementList:
        name = _safeGetRequiredAttribute(xmlElement,
                                         'name',
                                         'Error: context name is not set!!!')
        context = _parserModule.Context(parser, name)
        contextList.append(context)

    defaultContext = contextList[0]

    contextDict = {}
    for context in contextList:
        contextDict[context.name] = context

    parser.setContexts(contextDict, defaultContext)

    # parse contexts stage 2: load contexts
    for xmlElement, context in zip(xmlElementList, contextList):
        _loadContext(context, xmlElement, attributeToFormatMap)


def _loadContext(context, xmlElement, attributeToFormatMap):
    """Construct context from XML element
    Contexts are at first constructed, and only then loaded, because when loading context,
    _makeContextSwitcher must have references to all defined contexts
    """
    attribute = _safeGetRequiredAttribute(xmlElement, 'attribute', '<not set>').lower()
    if attribute != '<not set>':  # there are no attributes for internal contexts, used by rules. See perl.xml
        try:
            format = attributeToFormatMap[attribute]
        except KeyError:
            _logger.warning('Unknown context attribute %s', attribute)
            format = TextFormat()
    else:
        format = None

    textType = format.textType if format is not None else ' '
    if format is not None:
        format = _convertFormat(format)

    lineEndContextText = xmlElement.attrib.get('lineEndContext', '#stay')
    lineEndContext = _makeContextSwitcher(lineEndContextText,  context.parser)
    lineBeginContextText = xmlElement.attrib.get('lineBeginContext', '#stay')
    lineBeginContext = _makeContextSwitcher(lineBeginContextText, context.parser)
    lineEmptyContextText = xmlElement.attrib.get('lineEmptyContext', '#stay')
    lineEmptyContext = _makeContextSwitcher(lineEmptyContextText, context.parser)

    if _parseBoolAttribute(xmlElement.attrib.get('fallthrough', 'false')):
        fallthroughContextText = _safeGetRequiredAttribute(xmlElement, 'fallthroughContext', '#stay')
        fallthroughContext = _makeContextSwitcher(fallthroughContextText, context.parser)
    else:
        fallthroughContext = None

    dynamic = _parseBoolAttribute(xmlElement.attrib.get('dynamic', 'false'))

    context.setValues(attribute, format, lineEndContext, lineBeginContext, lineEmptyContext, fallthroughContext, dynamic, textType)

    # load rules
    rules = _loadChildRules(context, xmlElement, attributeToFormatMap)
    context.setRules(rules)

################################################################################
##                               Syntax
################################################################################
def _textTypeForDefStyleName(attribute, defStyleName):
    """ ' ' for code
        'c' for comments
        'b' for block comments
        'h' for here documents
    """
    if 'here' in attribute.lower() and defStyleName == 'dsOthers':
        return 'h'  # ruby
    elif 'block' in attribute.lower() and defStyleName == 'dsComment':
        return 'b'
    elif defStyleName in ('dsString', 'dsRegionMarker', 'dsChar', 'dsOthers'):
        return 's'
    elif defStyleName == 'dsComment':
        return 'c'
    else:
        return ' '

def _makeFormat(defaultTheme, defaultStyleName, textType, item=None):
    format = copy.copy(defaultTheme.format[defaultStyleName])

    format.textType = textType

    if item is not None:
        caseInsensitiveAttributes = {}
        for key, value in item.attrib.items():
            caseInsensitiveAttributes[key.lower()] = value.lower()

        if 'color' in caseInsensitiveAttributes:
            format.color = caseInsensitiveAttributes['color']
        if 'selColor' in caseInsensitiveAttributes:
            format.selectionColor = caseInsensitiveAttributes['selColor']
        if 'italic' in caseInsensitiveAttributes:
            format.italic = _parseBoolAttribute(caseInsensitiveAttributes['italic'])
        if 'bold' in caseInsensitiveAttributes:
            format.bold = _parseBoolAttribute(caseInsensitiveAttributes['bold'])
        if 'underline' in caseInsensitiveAttributes:
            format.underline = _parseBoolAttribute(caseInsensitiveAttributes['underline'])
        if 'strikeout' in caseInsensitiveAttributes:
            format.strikeOut = _parseBoolAttribute(caseInsensitiveAttributes['strikeout'])
        if 'spellChecking' in caseInsensitiveAttributes:
            format.spellChecking = _parseBoolAttribute(caseInsensitiveAttributes['spellChecking'])

    return format

def _loadAttributeToFormatMap(highlightingElement):
    defaultTheme = ColorTheme(TextFormat)
    attributeToFormatMap = {}

    itemDatasElement = highlightingElement.find('itemDatas')
    if itemDatasElement is not None:
        for item in itemDatasElement.findall('itemData'):
            attribute = item.get('name').lower()
            defaultStyleName = item.get('defStyleNum')

            if not defaultStyleName in defaultTheme.format:
                _logger.warning("Unknown default style '%s'", defaultStyleName)
                defaultStyleName = 'dsNormal'

            format = _makeFormat(defaultTheme,
                                 defaultStyleName,
                                 _textTypeForDefStyleName(attribute, defaultStyleName),
                                 item)

            attributeToFormatMap[attribute] = format

    # HACK not documented, but 'normal' attribute is used by some parsers without declaration
    if not 'normal' in attributeToFormatMap:
        attributeToFormatMap['normal'] = _makeFormat(defaultTheme, 'dsNormal',
                                                     _textTypeForDefStyleName('normal', 'dsNormal'))
    if not 'string' in attributeToFormatMap:
        attributeToFormatMap['string'] = _makeFormat(defaultTheme, 'dsString',
                                                     _textTypeForDefStyleName('string', 'dsString'))

    return attributeToFormatMap

def _loadLists(root, highlightingElement):
    lists = {}  # list name: list
    for listElement in highlightingElement.findall('list'):
        # Sometimes item.text is none. Broken xml files
        items = [str(item.text.strip()) \
                    for item in listElement.findall('item') \
                        if item.text is not None]
        name = _safeGetRequiredAttribute(listElement, 'name', 'Error: list name is not set!!!')
        lists[name] = items

    return lists

def _makeKeywordsLowerCase(listDict):
    # Make all keywords lowercase, if syntax is not case sensitive
    for keywordList in listDict.values():
        for index, keyword in enumerate(keywordList):
            keywordList[index] = keyword.lower()

def _loadSyntaxDescription(root, syntax):
    syntax.name = _safeGetRequiredAttribute(root, 'name', 'Error: .parser name is not set!!!')
    syntax.section = _safeGetRequiredAttribute(root, 'section', 'Error: Section is not set!!!')
    syntax.extensions = [_f for _f in _safeGetRequiredAttribute(root, 'extensions', '').split(';') if _f]
    syntax.firstLineGlobs = [_f for _f in root.attrib.get('firstLineGlobs', '').split(';') if _f]
    syntax.mimetype = [_f for _f in root.attrib.get('mimetype', '').split(';') if _f]
    syntax.version = root.attrib.get('version', None)
    syntax.kateversion = root.attrib.get('kateversion', None)
    syntax.priority = int(root.attrib.get('priority', '0'))
    syntax.author = root.attrib.get('author', None)
    syntax.license = root.attrib.get('license', None)
    syntax.hidden = _parseBoolAttribute(root.attrib.get('hidden', 'false'))

    # not documented
    syntax.indenter = root.attrib.get('indenter', None)


def loadSyntax(syntax, filePath = None):
    _logger.debug("Loading syntax %s", filePath)
    with open(filePath, 'r', encoding='utf-8') as definitionFile:
        try:
            root = xml.etree.ElementTree.parse(definitionFile).getroot()
        except Exception as ex:
            print('When opening %s:' % filePath, file=sys.stderr)
            raise

    highlightingElement = root.find('highlighting')

    _loadSyntaxDescription(root, syntax)

    deliminatorSet = set(_DEFAULT_DELIMINATOR)

    # parse lists
    lists = _loadLists(root, highlightingElement)

    # parse itemData
    keywordsCaseSensitive = True

    generalElement = root.find('general')
    if generalElement is not None:
        keywordsElement = generalElement.find('keywords')

        if keywordsElement is not None:
            keywordsCaseSensitive = _parseBoolAttribute(keywordsElement.get('casesensitive', "true"))

            if not keywordsCaseSensitive:
                _makeKeywordsLowerCase(lists)

            if 'weakDeliminator' in keywordsElement.attrib:
                weakSet = keywordsElement.attrib['weakDeliminator']
                deliminatorSet.difference_update(weakSet)

            if 'additionalDeliminator' in keywordsElement.attrib:
                additionalSet = keywordsElement.attrib['additionalDeliminator']
                deliminatorSet.update(additionalSet)

        indentationElement = generalElement.find('indentation')

        if indentationElement is not None and \
           'mode' in indentationElement.attrib:
            syntax.indenter = indentationElement.attrib['mode']

    deliminatorSetAsString = ''.join(list(deliminatorSet))
    debugOutputEnabled = _logger.isEnabledFor(logging.DEBUG)  # for cParser
    parser = _parserModule.Parser(syntax, deliminatorSetAsString, lists, keywordsCaseSensitive, debugOutputEnabled)
    syntax._setParser(parser)
    attributeToFormatMap = _loadAttributeToFormatMap(highlightingElement)

    # parse contexts
    _loadContexts(highlightingElement, syntax.parser, attributeToFormatMap)

    return syntax
