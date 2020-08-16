"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Kate syntax definition parser and representation

Do not use this module directly. Use 'syntax' module

Read http://kate-editor.org/2005/03/24/writing-a-syntax-highlighting-file/
if you want to understand something


'attribute' property of rules and contexts contains not an original string,
but value from itemDatas section (style name)

'context', 'lineBeginContext', 'lineEndContext', 'fallthroughContext' properties
contain not a text value, but ContextSwitcher object
"""

import re
import logging

_logger = logging.getLogger('qutepart')

_numSeqReplacer = re.compile('%\d+')


class ContextStack:
    def __init__(self, contexts, data):
        """Create default context stack for syntax
        Contains default context on the top
        """
        self._contexts = contexts
        self._data = data

    def pop(self, count):
        """Returns new context stack, which doesn't contain few levels
        """
        if len(self._contexts) - 1 < count:
            _logger.error("#pop value is too big %d", len(self._contexts))
            if len(self._contexts) > 1:
                return ContextStack(self._contexts[:1], self._data[:1])
            else:
                return self

        return ContextStack(self._contexts[:-count], self._data[:-count])

    def append(self, context, data):
        """Returns new context, which contains current stack and new frame
        """
        return ContextStack(self._contexts + [context], self._data + [data])

    def currentContext(self):
        """Get current context
        """
        return self._contexts[-1]

    def currentData(self):
        """Get current data
        """
        return self._data[-1]


class ContextSwitcher:
    """Class parses 'context', 'lineBeginContext', 'lineEndContext', 'fallthroughContext'
    and modifies context stack according to context operation
    """
    def __init__(self, popsCount, contextToSwitch, contextOperation):
        self._popsCount = popsCount
        self._contextToSwitch = contextToSwitch
        self._contextOperation = contextOperation

    def __str__(self):
        return self._contextOperation

    def getNextContextStack(self, contextStack, data=None):
        """Apply modification to the contextStack.
        This method never modifies input parameter list
        """
        if self._popsCount:
            contextStack = contextStack.pop(self._popsCount)

        if self._contextToSwitch is not None:
            if not self._contextToSwitch.dynamic:
                data = None
            contextStack = contextStack.append(self._contextToSwitch, data)

        return contextStack


class TextToMatchObject:
    """Peace of text, which shall be matched.
    Contains pre-calculated and pre-checked data for performance optimization
    """
    def __init__(self, currentColumnIndex, wholeLineText, deliminatorSet, contextData):
        self.currentColumnIndex = currentColumnIndex
        self.wholeLineText = wholeLineText
        self.text = wholeLineText[currentColumnIndex:]
        self.textLen = len(self.text)

        self.firstNonSpace = not bool(wholeLineText[:currentColumnIndex].strip())

        self.isWordStart = currentColumnIndex == 0 or \
                         wholeLineText[currentColumnIndex - 1].isspace() or \
                         wholeLineText[currentColumnIndex - 1] in deliminatorSet

        self.word = None
        if self.isWordStart:
            wordEndIndex = 0
            for index, char in enumerate(self.text):
                if char in deliminatorSet:
                    wordEndIndex = index
                    break
            else:
                wordEndIndex = len(wholeLineText)

            if wordEndIndex != 0:
                self.word = self.text[:wordEndIndex]

        self.contextData = contextData


class RuleTryMatchResult:
    def __init__(self, rule, length, data=None):
        self.rule = rule
        self.length = length
        self.data = data

        if rule.lookAhead:
            self.length = 0


class AbstractRuleParams:
    """Parameters, passed to the AbstractRule constructor
    """
    def __init__(self, parentContext, format, textType, attribute, context, lookAhead, firstNonSpace, dynamic, column):
        self.parentContext = parentContext
        self.format = format
        self.textType = textType
        self.attribute = attribute
        self.context = context
        self.lookAhead = lookAhead
        self.firstNonSpace = firstNonSpace
        self.dynamic = dynamic
        self.column = column


class AbstractRule:
    """Base class for rule classes
    Public attributes:
        parentContext
        format              May be None
        textType            May be None
        attribute           May be None
        context
        lookAhead
        firstNonSpace
        column          -1 if not set
        dynamic
    """

    _seqReplacer = re.compile('%\d+')

    def __init__(self, params):
        self.parentContext = params.parentContext
        self.format = params.format
        self.textType = params.textType
        self.attribute = params.attribute
        self.context = params.context
        self.lookAhead = params.lookAhead
        self.firstNonSpace = params.firstNonSpace
        self.dynamic = params.dynamic
        self.column = params.column

    def __str__(self):
        """Serialize.
        For debug logs
        """
        res = '\t\tRule %s\n' % self.shortId()
        res += '\t\t\tstyleName: %s\n' % (self.attribute or 'None')
        res += '\t\t\tcontext: %s\n' % self.context
        return res

    def shortId(self):
        """Get short ID string of the rule. Used for logs
        i.e. "DetectChar(x)"
        """
        raise NotImplementedError(str(self.__class__))

    def tryMatch(self, textToMatchObject):
        """Try to find themselves in the text.
        Returns (contextStack, count, matchedRule) or (contextStack, None, None) if doesn't match
        """
        # Skip if column doesn't match
        if self.column != -1 and \
           self.column != textToMatchObject.currentColumnIndex:
            return None

        if self.firstNonSpace and \
           (not textToMatchObject.firstNonSpace):
            return None

        return self._tryMatch(textToMatchObject)


class DetectChar(AbstractRule):
    """Public attributes:
        char
    """
    def __init__(self, abstractRuleParams, char, index):
        AbstractRule.__init__(self, abstractRuleParams)
        self.char = char
        self.index = index

    def shortId(self):
        return 'DetectChar(%s, %d)' % (self.char, self.index)

    def _tryMatch(self, textToMatchObject):
        if self.char is None and self.index == 0:
            return None

        if self.dynamic:
            index = self.index - 1
            if index >= len(textToMatchObject.contextData):
                _logger.error('Invalid DetectChar index %d', index)
                return None

            if len(textToMatchObject.contextData[index]) != 1:
                    _logger.error('Too long DetectChar string %s', textToMatchObject.contextData[index])
                    return None

            string = textToMatchObject.contextData[index]
        else:
            string = self.char

        if textToMatchObject.text[0] == string:
            return RuleTryMatchResult(self, 1)
        return None


class Detect2Chars(AbstractRule):
    """Public attributes
        string
    """
    def __init__(self, abstractRuleParams, string):
        AbstractRule.__init__(self, abstractRuleParams)
        self.string = string

    def shortId(self):
        return 'Detect2Chars(%s)' % self.string

    def _tryMatch(self, textToMatchObject):
        if self.string is None:
            return None

        if textToMatchObject.text.startswith(self.string):
            return RuleTryMatchResult(self, len(self.string))

        return None


class AnyChar(AbstractRule):
    """Public attributes:
        string
    """
    def __init__(self, abstractRuleParams, string):
        AbstractRule.__init__(self, abstractRuleParams)
        self.string = string

    def shortId(self):
        return 'AnyChar(%s)' % self.string

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.text[0] in self.string:
            return RuleTryMatchResult(self, 1)

        return None


class StringDetect(AbstractRule):
    """Public attributes:
        string
    """
    def __init__(self, abstractRuleParams, string):
        AbstractRule.__init__(self, abstractRuleParams)
        self.string = string

    def shortId(self):
        return 'StringDetect(%s)' % self.string

    def _tryMatch(self, textToMatchObject):
        if self.string is None:
            return None

        if self.dynamic:
            string = self._makeDynamicSubsctitutions(self.string, textToMatchObject.contextData)
            if not string:
                return None
        else:
            string = self.string

        if textToMatchObject.text.startswith(string):
            return RuleTryMatchResult(self, len(string))

        return None

    @staticmethod
    def _makeDynamicSubsctitutions(string, contextData):
        """For dynamic rules, replace %d patterns with actual strings
        Python function, which is used by C extension.
        """
        def _replaceFunc(escapeMatchObject):
            stringIndex = escapeMatchObject.group(0)[1]
            index = int(stringIndex)
            if index < len(contextData):
                return contextData[index]
            else:
                return escapeMatchObject.group(0)  # no any replacements, return original value

        return _numSeqReplacer.sub(_replaceFunc, string)


class WordDetect(AbstractRule):
    """Public attributes:
        words
    """
    def __init__(self, abstractRuleParams, word, insensitive):
        AbstractRule.__init__(self, abstractRuleParams)
        self.word = word
        self.insensitive = insensitive

    def shortId(self):
        return 'WordDetect(%s, %d)' % (self.word, self.insensitive)

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.word is None:
            return None

        if self.insensitive or \
           (not self.parentContext.parser.keywordsCaseSensitive):
            wordToCheck = textToMatchObject.word.lower()
        else:
            wordToCheck = textToMatchObject.word

        if wordToCheck == self.word:
            return RuleTryMatchResult(self, len(wordToCheck))
        else:
            return None


class keyword(AbstractRule):
    """Public attributes:
        string
        words
    """
    def __init__(self, abstractRuleParams, words, insensitive):
        AbstractRule.__init__(self, abstractRuleParams)
        self.words = set(words)
        self.insensitive = insensitive

    def shortId(self):
        return 'keyword(%s, %d)' % (' '.join(list(self.words)), self.insensitive)

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.word is None:
            return None

        if self.insensitive or \
           (not self.parentContext.parser.keywordsCaseSensitive):
            wordToCheck = textToMatchObject.word.lower()
        else:
            wordToCheck = textToMatchObject.word

        if wordToCheck in self.words:
            return RuleTryMatchResult(self, len(wordToCheck))
        else:
            return None


class RegExpr(AbstractRule):
    """ Public attributes:
        regExp
        wordStart
        lineStart
    """
    def __init__(self, abstractRuleParams,
                 string, insensitive, minimal, wordStart, lineStart):
        AbstractRule.__init__(self, abstractRuleParams)
        self.string = string
        self.insensitive = insensitive
        self.minimal = minimal
        self.wordStart = wordStart
        self.lineStart = lineStart

        if self.dynamic:
            self.regExp = None
        else:
            self.regExp = self._compileRegExp(string, insensitive, minimal)


    def shortId(self):
        return 'RegExpr( %s )' % self.string

    def _tryMatch(self, textToMatchObject):
        """Tries to parse text. If matched - saves data for dynamic context
        """
        # Special case. if pattern starts with \b, we have to check it manually,
        # because string is passed to .match(..) without beginning
        if self.wordStart and \
           (not textToMatchObject.isWordStart):
                return None

        #Special case. If pattern starts with ^ - check column number manually
        if self.lineStart and \
           textToMatchObject.currentColumnIndex > 0:
            return None

        if self.dynamic:
            string = self._makeDynamicSubsctitutions(self.string, textToMatchObject.contextData)
            regExp = self._compileRegExp(string, self.insensitive, self.minimal)
        else:
            regExp = self.regExp

        if regExp is None:
            return None

        wholeMatch, groups = self._matchPattern(regExp, textToMatchObject.text)
        if wholeMatch is not None:
            count = len(wholeMatch)
            return RuleTryMatchResult(self, count, groups)
        else:
            return None

    @staticmethod
    def _makeDynamicSubsctitutions(string, contextData):
        """For dynamic rules, replace %d patterns with actual strings
        Escapes reg exp symbols in the pattern
        Python function, used by C code
        """
        def _replaceFunc(escapeMatchObject):
            stringIndex = escapeMatchObject.group(0)[1]
            index = int(stringIndex)
            if index < len(contextData):
                return re.escape(contextData[index])
            else:
                return escapeMatchObject.group(0)  # no any replacements, return original value

        return _numSeqReplacer.sub(_replaceFunc, string)

    @staticmethod
    def _compileRegExp(string, insensitive, minimal):
        """Compile regular expression.
        Python function, used by C code

        NOTE minimal flag is not supported here, but supported on PCRE
        """
        flags = 0
        if insensitive:
            flags = re.IGNORECASE

        string = string.replace('[_[:alnum:]]', '[\\w\\d]') # ad-hoc fix for C++ parser
        string = string.replace('[:digit:]', '\\d')
        string = string.replace('[:blank:]', '\\s')

        try:
            return re.compile(string, flags)
        except (re.error, AssertionError) as ex:
            _logger.warning("Invalid pattern '%s': %s", string, str(ex))
            return None

    @staticmethod
    def _matchPattern(regExp, string):
        """Try to match pattern.
        Returns tuple (whole match, groups) or (None, None)
        Python function, used by C code
        """
        match = regExp.match(string)
        if match is not None and match.group(0):
            return match.group(0), (match.group(0), ) + match.groups()
        else:
            return None, None


class AbstractNumberRule(AbstractRule):
    """Base class for Int and Float rules.
    This rules can have child rules

    Public attributes:
        childRules
    """
    def __init__(self, abstractRuleParams, childRules):
        AbstractRule.__init__(self, abstractRuleParams)
        self.childRules = childRules

    def _tryMatch(self, textToMatchObject):
        """Try to find themselves in the text.
        Returns (count, matchedRule) or (None, None) if doesn't match
        """

        # andreikop: This check is not described in kate docs, and I haven't found it in the code
        if not textToMatchObject.isWordStart:
            return None

        index = self._tryMatchText(textToMatchObject.text)
        if index is None:
            return None

        if textToMatchObject.currentColumnIndex + index < len(textToMatchObject.wholeLineText):
            newTextToMatchObject = TextToMatchObject(textToMatchObject.currentColumnIndex + index,
                                                      textToMatchObject.wholeLineText,
                                                      self.parentContext.parser.deliminatorSet,
                                                      textToMatchObject.contextData)
            for rule in self.childRules:
                ruleTryMatchResult = rule.tryMatch(newTextToMatchObject)
                if ruleTryMatchResult is not None:
                    index += ruleTryMatchResult.length
                    break
                # child rule context and attribute ignored

        return RuleTryMatchResult(self, index)

    def _countDigits(self, text):
        """Count digits at start of text
        """
        index = 0
        while index < len(text):
            if not text[index].isdigit():
                break
            index += 1
        return index


class Int(AbstractNumberRule):
    def shortId(self):
        return 'Int()'

    def _tryMatchText(self, text):
        matchedLength = self._countDigits(text)

        if matchedLength:
            return matchedLength
        else:
            return None


class Float(AbstractNumberRule):
    def shortId(self):
        return 'Float()'

    def _tryMatchText(self, text):

        haveDigit = False
        havePoint = False

        matchedLength = 0

        digitCount = self._countDigits(text[matchedLength:])
        if digitCount:
            haveDigit = True
            matchedLength += digitCount

        if len(text) > matchedLength and text[matchedLength] == '.':
            havePoint = True
            matchedLength += 1

        digitCount = self._countDigits(text[matchedLength:])
        if digitCount:
            haveDigit = True
            matchedLength += digitCount

        if len(text) > matchedLength and text[matchedLength].lower() == 'e':
            matchedLength += 1

            if len(text) > matchedLength and text[matchedLength] in '+-':
                matchedLength += 1

            haveDigitInExponent = False

            digitCount = self._countDigits(text[matchedLength:])
            if digitCount:
                haveDigitInExponent = True
                matchedLength += digitCount

            if not haveDigitInExponent:
                return None

            return matchedLength
        else:
            if not havePoint:
                return None

        if matchedLength and haveDigit:
            return matchedLength
        else:
            return None


class HlCOct(AbstractRule):
    def shortId(self):
        return 'HlCOct'

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.text[0] != '0':
            return None

        index = 1
        while index < len(textToMatchObject.text) and textToMatchObject.text[index] in '01234567':
            index += 1

        if index == 1:
            return None

        if index < len(textToMatchObject.text) and textToMatchObject.text[index].upper() in 'LU':
            index += 1

        return RuleTryMatchResult(self, index)


class HlCHex(AbstractRule):
    def shortId(self):
        return 'HlCHex'

    def _tryMatch(self, textToMatchObject):
        if len(textToMatchObject.text) < 3:
            return None

        if textToMatchObject.text[:2].upper() != '0X':
            return None

        index = 2
        while index < len(textToMatchObject.text) and textToMatchObject.text[index].upper() in '0123456789ABCDEF':
            index += 1

        if index == 2:
            return None

        if index < len(textToMatchObject.text) and textToMatchObject.text[index].upper() in 'LU':
            index += 1

        return RuleTryMatchResult(self, index)


def _checkEscapedChar(text):
    index = 0
    if len(text) > 1 and text[0] == '\\':
        index = 1

        if text[index] in "abefnrtv'\"?\\":
            index += 1
        elif text[index] == 'x':  # if it's like \xff, eat the x
            index += 1
            while index < len(text) and text[index].upper() in '0123456789ABCDEF':
                index += 1
            if index == 2:  # no hex digits
                return None
        elif text[index] in '01234567':
            while index < 4 and index < len(text) and text[index] in '01234567':
                index += 1
        else:
            return None

        return index

    return None


class HlCStringChar(AbstractRule):
    def shortId(self):
        return 'HlCStringChar'

    def _tryMatch(self, textToMatchObject):
        res = _checkEscapedChar(textToMatchObject.text)
        if res is not None:
            return RuleTryMatchResult(self, res)
        else:
            return None


class HlCChar(AbstractRule):
    def shortId(self):
        return 'HlCChar'

    def _tryMatch(self, textToMatchObject):
        if len(textToMatchObject.text) > 2 and textToMatchObject.text[0] == "'" and textToMatchObject.text[1] != "'":
            result = _checkEscapedChar(textToMatchObject.text[1:])
            if result is not None:
                index = 1 + result
            else:  # 1 not escaped character
                index = 1 + 1

            if index < len(textToMatchObject.text) and textToMatchObject.text[index] == "'":
                return RuleTryMatchResult(self, index + 1)

        return None


class RangeDetect(AbstractRule):
    """Public attributes:
        char
        char1
    """
    def __init__(self, abstractRuleParams, char, char1):
        AbstractRule.__init__(self, abstractRuleParams)
        self.char = char
        self.char1 = char1

    def shortId(self):
        return 'RangeDetect(%s, %s)' % (self.char, self.char1)

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.text.startswith(self.char):
            end = textToMatchObject.text.find(self.char1, 1)
            if end > 0:
                return RuleTryMatchResult(self, end + 1)

        return None


class LineContinue(AbstractRule):
    def shortId(self):
        return 'LineContinue'

    def _tryMatch(self, textToMatchObject):
        if textToMatchObject.text == '\\':
            return RuleTryMatchResult(self, 1)

        return None


class IncludeRules(AbstractRule):
    def __init__(self, abstractRuleParams, context):
        AbstractRule.__init__(self, abstractRuleParams)
        self.context = context

    def __str__(self):
        """Serialize.
        For debug logs
        """
        res = '\t\tRule %s\n' % self.shortId()
        res += '\t\t\tstyleName: %s\n' % (self.attribute or 'None')
        return res

    def shortId(self):
        return "IncludeRules(%s)" % self.context.name

    def _tryMatch(self, textToMatchObject):
        """Try to find themselves in the text.
        Returns (count, matchedRule) or (None, None) if doesn't match
        """
        for rule in self.context.rules:
            ruleTryMatchResult = rule.tryMatch(textToMatchObject)
            if ruleTryMatchResult is not None:
                _logger.debug('\tmatched rule %s at %d in included context %s/%s',
                              rule.shortId(),
                              textToMatchObject.currentColumnIndex,
                              self.context.parser.syntax.name,
                              self.context.name)
                return ruleTryMatchResult
        else:
            return None


class DetectSpaces(AbstractRule):
    def shortId(self):
        return 'DetectSpaces()'

    def _tryMatch(self, textToMatchObject):
        spaceLen = len(textToMatchObject.text) - len(textToMatchObject.text.lstrip())
        if spaceLen:
            return RuleTryMatchResult(self, spaceLen)
        else:
            return None


class DetectIdentifier(AbstractRule):
    _regExp = re.compile('[a-zA-Z][a-zA-Z0-9_]*')
    def shortId(self):
        return 'DetectIdentifier()'

    def _tryMatch(self, textToMatchObject):
        match = DetectIdentifier._regExp.match(textToMatchObject.text)
        if match is not None and match.group(0):
            return RuleTryMatchResult(self, len(match.group(0)))

        return None


class Context:
    """Highlighting context

    Public attributes:
        attribute
        lineEndContext
        lineBeginContext
        fallthroughContext
        dynamic
        rules
        textType     ' ' : code, 'c' : comment
    """
    def __init__(self, parser, name):
        # Will be initialized later, after all context has been created
        self.parser = parser
        self.name = name

    def setValues(self, attribute, format, lineEndContext, lineBeginContext, lineEmptyContext, fallthroughContext, dynamic, textType):
        self.attribute = attribute
        self.format = format
        self.lineEndContext = lineEndContext
        self.lineBeginContext = lineBeginContext
        self.lineEmptyContext = lineEmptyContext
        self.fallthroughContext = fallthroughContext
        self.dynamic = dynamic
        self.textType = textType

    def setRules(self, rules):
        self.rules = rules

    def __str__(self):
        """Serialize.
        For debug logs
        """
        res = '\tContext %s\n' % self.name
        res += '\t\t%s: %s\n' % ('attribute', self.attribute)
        res += '\t\t%s: %s\n' % ('lineEndContext', self.lineEndContext)
        res += '\t\t%s: %s\n' % ('lineBeginContext', self.lineBeginContext)
        res += '\t\t%s: %s\n' % ('lineEmptyContext', self.lineEmptyContext)
        if self.fallthroughContext is not None:
            res += '\t\t%s: %s\n' % ('fallthroughContext', self.fallthroughContext)
        res += '\t\t%s: %s\n' % ('dynamic', self.dynamic)

        for rule in self.rules:
            res += str(rule)
        return res

    def parseBlock(self, contextStack, currentColumnIndex, text):
        """Parse block
        Exits, when reached end of the text, or when context is switched
        Returns (length, newContextStack, highlightedSegments, lineContinue)
        """
        startColumnIndex = currentColumnIndex
        countOfNotMatchedSymbols = 0
        highlightedSegments = []
        textTypeMap = []
        ruleTryMatchResult = None
        while currentColumnIndex < len(text):
            textToMatchObject = TextToMatchObject(currentColumnIndex,
                                                   text,
                                                   self.parser.deliminatorSet,
                                                   contextStack.currentData())
            for rule in self.rules:
                ruleTryMatchResult = rule.tryMatch(textToMatchObject)
                if ruleTryMatchResult is not None:  # if something matched
                    _logger.debug('\tmatched rule %s at %d',
                                  rule.shortId(),
                                  currentColumnIndex)
                    if countOfNotMatchedSymbols > 0:
                        highlightedSegments.append((countOfNotMatchedSymbols, self.format))
                        textTypeMap += [self.textType for i in range(countOfNotMatchedSymbols)]
                        countOfNotMatchedSymbols = 0

                    if ruleTryMatchResult.rule.context is not None:
                        newContextStack = ruleTryMatchResult.rule.context.getNextContextStack(contextStack,
                                                                                              ruleTryMatchResult.data)
                    else:
                        newContextStack = contextStack

                    format = ruleTryMatchResult.rule.format if ruleTryMatchResult.rule.attribute else newContextStack.currentContext().format
                    textType = ruleTryMatchResult.rule.textType or newContextStack.currentContext().textType

                    highlightedSegments.append((ruleTryMatchResult.length,
                                                format))
                    textTypeMap += textType * ruleTryMatchResult.length

                    currentColumnIndex += ruleTryMatchResult.length

                    if newContextStack != contextStack:
                        lineContinue = isinstance(ruleTryMatchResult.rule, LineContinue)

                        return currentColumnIndex - startColumnIndex, newContextStack, highlightedSegments, textTypeMap, lineContinue

                    break  # for loop
            else:  # no matched rules
                if self.fallthroughContext is not None:
                    newContextStack = self.fallthroughContext.getNextContextStack(contextStack)
                    if newContextStack != contextStack:
                        if countOfNotMatchedSymbols > 0:
                            highlightedSegments.append((countOfNotMatchedSymbols, self.format))
                            textTypeMap += [self.textType for i in range(countOfNotMatchedSymbols)]
                        return (currentColumnIndex - startColumnIndex, newContextStack, highlightedSegments, textTypeMap, False)

                currentColumnIndex += 1
                countOfNotMatchedSymbols += 1

        if countOfNotMatchedSymbols > 0:
            highlightedSegments.append((countOfNotMatchedSymbols, self.format))
            textTypeMap += [self.textType for i in range(countOfNotMatchedSymbols)]

        lineContinue = ruleTryMatchResult is not None and \
                       isinstance(ruleTryMatchResult.rule, LineContinue)

        return currentColumnIndex - startColumnIndex, contextStack, highlightedSegments, textTypeMap, lineContinue


class Parser:
    """Parser implementation

        syntax                  Syntax instance

        attributeToFormatMap    Map "attribute" : TextFormat

        deliminatorSet          Set of deliminator characters
        lists                   Keyword lists as dictionary "list name" : "list value"
        keywordsCaseSensitive   If true, keywords are not case sensitive

        contexts                Context list as dictionary "context name" : context
        defaultContext          Default context object
    """
    def __init__(self, syntax, deliminatorSetAsString, lists, keywordsCaseSensitive, debugOutputEnabled):
        self.syntax = syntax
        self.deliminatorSet = set(deliminatorSetAsString)
        self.lists = lists
        self.keywordsCaseSensitive = keywordsCaseSensitive
        # debugOutputEnabled is used only by cParser

    def setContexts(self, contexts, defaultContext):
        self.contexts = contexts
        self.defaultContext = defaultContext
        self._defaultContextStack = ContextStack([self.defaultContext], [None])

    def __str__(self):
        """Serialize.
        For debug logs
        """
        res = 'Parser\n'
        for name, value in vars(self).items():
            if not name.startswith('_') and \
               not name in ('defaultContext', 'deliminatorSet', 'contexts', 'lists', 'syntax') and \
               not value is None:
                res += '\t%s: %s\n' % (name, value)

        res += '\tDefault context: %s\n' % self.defaultContext.name

        for listName, listValue in self.lists.items():
            res += '\tList %s: %s\n' % (listName, listValue)


        for context in self.contexts.values():
            res += str(context)

        return res

    def highlightBlock(self, text, prevContextStack):
        """Parse block and return ParseBlockFullResult

        return (lineData, highlightedSegments)
          where lineData is (contextStack, textTypeMap)
            where textTypeMap is a string of textType characters
        """
        if prevContextStack is not None:
            contextStack = prevContextStack
        else:
            contextStack = self._defaultContextStack

        highlightedSegments = []
        lineContinue = False
        currentColumnIndex = 0
        textTypeMap = []

        if len(text) > 0:
            while currentColumnIndex < len(text):
                _logger.debug('In context %s', contextStack.currentContext().name)

                length, newContextStack, segments, textTypeMapPart, lineContinue = \
                    contextStack.currentContext().parseBlock(contextStack, currentColumnIndex, text)

                highlightedSegments += segments
                contextStack = newContextStack
                textTypeMap += textTypeMapPart
                currentColumnIndex += length

            if not lineContinue:
                while contextStack.currentContext().lineEndContext is not None:
                    oldStack = contextStack
                    contextStack = contextStack.currentContext().lineEndContext.getNextContextStack(contextStack)
                    if oldStack == contextStack:  # avoid infinite while loop if nothing to switch
                        break

                # this code is not tested, because lineBeginContext is not defined by any xml file
                if contextStack.currentContext().lineBeginContext is not None:
                    contextStack = contextStack.currentContext().lineBeginContext.getNextContextStack(contextStack)
        elif contextStack.currentContext().lineEmptyContext is not None:
            contextStack = contextStack.currentContext().lineEmptyContext.getNextContextStack(contextStack)

        lineData = (contextStack, textTypeMap)
        return lineData, highlightedSegments

    def parseBlock(self, text, prevContextStack):
        return self.highlightBlock(text, prevContextStack)[0]
