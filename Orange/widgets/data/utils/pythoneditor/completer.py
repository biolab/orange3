"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Autocompletion widget and logic
"""

import re
import time

from PyQt5.QtCore import pyqtSignal, QAbstractItemModel, QEvent, QModelIndex, QObject, QSize, Qt, QTimer
from PyQt5.QtWidgets import QListView
from PyQt5.QtGui import QCursor

from qutepart.htmldelegate import HTMLDelegate


_wordPattern = "\w+"
_wordRegExp = re.compile(_wordPattern)
_wordAtEndRegExp = re.compile(_wordPattern + '$')
_wordAtStartRegExp = re.compile('^' + _wordPattern)


# Maximum count of words, for which completion will be shown. Ignored, if completion invoked manually.
MAX_VISIBLE_WORD_COUNT = 256


class _GlobalUpdateWordSetTimer:
    """Timer updates word set, when editor is idle. (5 sec. after last change)
    Timer is global, for avoid situation, when all instances
    update set simultaneously
    """
    _IDLE_TIMEOUT_MS = 1000

    def __init__(self):
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._onTimer)
        self._scheduledMethods = []

    def schedule(self, method):
        if not method in self._scheduledMethods:
            self._scheduledMethods.append(method)
        self._timer.start(self._IDLE_TIMEOUT_MS)

    def cancel(self, method):
        """Cancel scheduled method
        Safe method, may be called with not-scheduled method"""
        if method in self._scheduledMethods:
            self._scheduledMethods.remove(method)

        if not self._scheduledMethods:
            self._timer.stop()

    def _onTimer(self):
        method = self._scheduledMethods.pop()
        method()
        if self._scheduledMethods:
            self._timer.start(self._IDLE_TIMEOUT_MS)


class _CompletionModel(QAbstractItemModel):
    """QAbstractItemModel implementation for a list of completion variants

    words attribute contains all words
    canCompleteText attribute contains text, which may be inserted with tab
    """
    def __init__(self, wordSet):
        QAbstractItemModel.__init__(self)

        self._wordSet = wordSet

    def setData(self, wordBeforeCursor, wholeWord):
        """Set model information
        """
        self._typedText = wordBeforeCursor
        self.words = self._makeListOfCompletions(wordBeforeCursor, wholeWord)
        commonStart = self._commonWordStart(self.words)
        self.canCompleteText = commonStart[len(wordBeforeCursor):]

        self.layoutChanged.emit()

    def hasWords(self):
        return len(self.words) > 0

    def tooManyWords(self):
        return len(self.words) > MAX_VISIBLE_WORD_COUNT

    def data(self, index, role):
        """QAbstractItemModel method implementation
        """
        if role == Qt.DisplayRole and \
           index.row() < len(self.words):
            text = self.words[index.row()]
            typed = text[:len(self._typedText)]
            canComplete = text[len(self._typedText):len(self._typedText) + len(self.canCompleteText)]
            rest = text[len(self._typedText) + len(self.canCompleteText):]
            if canComplete:
                # NOTE foreground colors are hardcoded, but I can't set background color of selected item (Qt bug?)
                # might look bad on some color themes
                return '<html>' \
                            '%s' \
                            '<font color="#e80000">%s</font>' \
                            '%s' \
                        '</html>' % (typed, canComplete, rest)
            else:
                return typed + rest
        else:
            return None

    def rowCount(self, index = QModelIndex()):
        """QAbstractItemModel method implementation
        """
        return len(self.words)

    def typedText(self):
        """Get current typed text
        """
        return self._typedText

    def _commonWordStart(self, words):
        """Get common start of all words.
        i.e. for ['blablaxxx', 'blablayyy', 'blazzz'] common start is 'bla'
        """
        if not words:
            return ''

        length = 0
        firstWord = words[0]
        otherWords = words[1:]
        for index, char in enumerate(firstWord):
            if not all([word[index] == char for word in otherWords]):
                break
            length = index + 1

        return firstWord[:length]

    def _makeListOfCompletions(self, wordBeforeCursor, wholeWord):
        """Make list of completions, which shall be shown
        """
        onlySuitable = [word for word in self._wordSet \
                                if word.startswith(wordBeforeCursor) and \
                                   word != wholeWord]

        return sorted(onlySuitable)

    """Trivial QAbstractItemModel methods implementation
    """
    def flags(self, index):                                 return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    def headerData(self, index):                            return None
    def columnCount(self, index):                           return 1
    def index(self, row, column, parent = QModelIndex()):   return self.createIndex(row, column)
    def parent(self, index):                                return QModelIndex()


class _CompletionList(QListView):
    """Completion list widget
    """
    closeMe = pyqtSignal()
    itemSelected = pyqtSignal(int)
    tabPressed = pyqtSignal()

    _MAX_VISIBLE_ROWS = 20  # no any technical reason, just for better UI

    def __init__(self, qpart, model):
        QListView.__init__(self, qpart.viewport())

        # ensure good selected item background on Windows
        palette = self.palette()
        palette.setColor(palette.Inactive, palette.Highlight, palette.color(palette.Active, palette.Highlight))
        self.setPalette(palette)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setItemDelegate(HTMLDelegate(self))

        self._qpart = qpart
        self.setFont(qpart.font())

        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFocusPolicy(Qt.NoFocus)

        self.setModel(model)

        self._selectedIndex = -1

        # if cursor moved, we shall close widget, if its position (and model) hasn't been updated
        self._closeIfNotUpdatedTimer = QTimer(self)
        self._closeIfNotUpdatedTimer.setInterval(200)
        self._closeIfNotUpdatedTimer.setSingleShot(True)

        self._closeIfNotUpdatedTimer.timeout.connect(self._afterCursorPositionChanged)

        qpart.installEventFilter(self)

        qpart.cursorPositionChanged.connect(self._onCursorPositionChanged)

        self.clicked.connect(lambda index: self.itemSelected.emit(index.row()))

        self.updateGeometry()
        self.show()

        qpart.setFocus()

    def __del__(self):
        """Without this empty destructor Qt prints strange trace
            QObject::startTimer: QTimer can only be used with threads started with QThread
        when exiting
        """
        pass

    def close(self):
        """Explicitly called destructor.
        Removes widget from the qpart
        """
        self._closeIfNotUpdatedTimer.stop()
        self._qpart.removeEventFilter(self)
        self._qpart.cursorPositionChanged.disconnect(self._onCursorPositionChanged)

        QListView.close(self)

    def sizeHint(self):
        """QWidget.sizeHint implementation
        Automatically resizes the widget according to rows count

        FIXME very bad algorithm. Remove all this margins, if you can
        """
        width = max([self.fontMetrics().width(word) \
                        for word in self.model().words])
        width = width * 1.4  # FIXME bad hack. invent better formula
        width += 30  # margin

        # drawn with scrollbar without +2. I don't know why
        rowCount = min(self.model().rowCount(), self._MAX_VISIBLE_ROWS)
        height = self.sizeHintForRow(0) * (rowCount + 0.5)  # + 0.5 row margin

        return QSize(width, height)

    def minimumHeight(self):
        """QWidget.minimumSizeHint implementation
        """
        return self.sizeHintForRow(0) * 1.5  # + 0.5 row margin

    def _horizontalShift(self):
        """List should be plased such way, that typed text in the list is under
        typed text in the editor
        """
        strangeAdjustment = 2  # I don't know why. Probably, won't work on other systems and versions
        return self.fontMetrics().width(self.model().typedText()) + strangeAdjustment

    def updateGeometry(self):
        """Move widget to point under cursor
        """
        WIDGET_BORDER_MARGIN = 5
        SCROLLBAR_WIDTH = 30  # just a guess

        sizeHint = self.sizeHint()
        width = sizeHint.width()
        height = sizeHint.height()

        cursorRect = self._qpart.cursorRect()
        parentSize = self.parentWidget().size()

        spaceBelow = parentSize.height() - cursorRect.bottom() - WIDGET_BORDER_MARGIN
        spaceAbove = cursorRect.top() - WIDGET_BORDER_MARGIN

        if height <= spaceBelow or \
           spaceBelow > spaceAbove:
            yPos = cursorRect.bottom()
            if height > spaceBelow and \
               spaceBelow > self.minimumHeight():
                height = spaceBelow
                width = width + SCROLLBAR_WIDTH
        else:
            if height > spaceAbove and \
               spaceAbove > self.minimumHeight():
                height = spaceAbove
                width = width + SCROLLBAR_WIDTH
            yPos = max(3, cursorRect.top() - height)

        xPos = cursorRect.right() - self._horizontalShift()

        if xPos + width + WIDGET_BORDER_MARGIN > parentSize.width():
            xPos = max(3, parentSize.width() - WIDGET_BORDER_MARGIN - width)

        self.setGeometry(xPos, yPos, width, height)
        self._closeIfNotUpdatedTimer.stop()

    def _onCursorPositionChanged(self):
        """Cursor position changed. Schedule closing.
        Timer will be stopped, if widget position is being updated
        """
        self._closeIfNotUpdatedTimer.start()

    def _afterCursorPositionChanged(self):
        """Widget position hasn't been updated after cursor position change, close widget
        """
        self.closeMe.emit()

    def eventFilter(self, object, event):
        """Catch events from qpart
        Move selection, select item, or close themselves
        """
        if event.type() == QEvent.KeyPress and event.modifiers() == Qt.NoModifier:
            if event.key() == Qt.Key_Escape:
                self.closeMe.emit()
                return True
            elif event.key() == Qt.Key_Down:
                if self._selectedIndex + 1 < self.model().rowCount():
                    self._selectItem(self._selectedIndex + 1)
                return True
            elif event.key() == Qt.Key_Up:
                if self._selectedIndex - 1 >= 0:
                    self._selectItem(self._selectedIndex - 1)
                return True
            elif event.key() in (Qt.Key_Enter, Qt.Key_Return):
                if self._selectedIndex != -1:
                    self.itemSelected.emit(self._selectedIndex)
                    return True
            elif event.key() == Qt.Key_Tab:
                self.tabPressed.emit()
                return True
        elif event.type() == QEvent.FocusOut:
            self.closeMe.emit()

        return False

    def _selectItem(self, index):
        """Select item in the list
        """
        self._selectedIndex = index
        self.setCurrentIndex(self.model().createIndex(index, 0))


class Completer(QObject):
    """Object listens Qutepart widget events, computes and shows autocompletion lists
    """
    _globalUpdateWordSetTimer = _GlobalUpdateWordSetTimer()

    _WORD_SET_UPDATE_MAX_TIME_SEC = 0.4

    def __init__(self, qpart):
        QObject.__init__(self, qpart)

        self._qpart = qpart
        self._widget = None
        self._completionOpenedManually = False

        self._keywords = set()
        self._customCompletions = set()
        self._wordSet = None

        qpart.textChanged.connect(self._onTextChanged)
        qpart.document().modificationChanged.connect(self._onModificationChanged)

    def terminate(self):
        """Object deleted. Cancel timer
        """
        self._globalUpdateWordSetTimer.cancel(self._updateWordSet)

    def setKeywords(self, keywords):
        self._keywords = keywords
        self._updateWordSet()

    def setCustomCompletions(self, wordSet):
        self._customCompletions = wordSet

    def isVisible(self):
        return self._widget is not None

    def _onTextChanged(self):
        """Text in the qpart changed. Update word set"""
        self._globalUpdateWordSetTimer.schedule(self._updateWordSet)

    def _onModificationChanged(self, modified):
        if not modified:
            self._closeCompletion()

    def _updateWordSet(self):
        """Make a set of words, which shall be completed, from text
        """
        self._wordSet = set(self._keywords) | set(self._customCompletions)

        start = time.time()

        for line in self._qpart.lines:
            for match in _wordRegExp.findall(line):
                self._wordSet.add(match)
            if time.time() - start > self._WORD_SET_UPDATE_MAX_TIME_SEC:
                """It is better to have incomplete word set, than to freeze the GUI"""
                break

    def invokeCompletion(self):
        """Invoke completion manually"""
        if self.invokeCompletionIfAvailable(requestedByUser=True):
            self._completionOpenedManually = True


    def _shouldShowModel(self, model, forceShow):
        if not model.hasWords():
            return False

        return forceShow or \
               (not model.tooManyWords())

    def _createWidget(self, model):
        self._widget = _CompletionList(self._qpart, model)
        self._widget.closeMe.connect(self._closeCompletion)
        self._widget.itemSelected.connect(self._onCompletionListItemSelected)
        self._widget.tabPressed.connect(self._onCompletionListTabPressed)

    def invokeCompletionIfAvailable(self, requestedByUser=False):
        """Invoke completion, if available. Called after text has been typed in qpart
        Returns True, if invoked
        """
        if self._qpart.completionEnabled and self._wordSet is not None:
            wordBeforeCursor = self._wordBeforeCursor()
            wholeWord = wordBeforeCursor + self._wordAfterCursor()

            forceShow = requestedByUser or self._completionOpenedManually
            if wordBeforeCursor:
                if len(wordBeforeCursor) >= self._qpart.completionThreshold or forceShow:
                    if self._widget is None:
                        model = _CompletionModel(self._wordSet)
                        model.setData(wordBeforeCursor, wholeWord)
                        if self._shouldShowModel(model, forceShow):
                            self._createWidget(model)
                            return True
                    else:
                        self._widget.model().setData(wordBeforeCursor, wholeWord)
                        if self._shouldShowModel(self._widget.model(), forceShow):
                            self._widget.updateGeometry()

                            return True

        self._closeCompletion()
        return False

    def _closeCompletion(self):
        """Close completion, if visible.
        Delete widget
        """
        if self._widget is not None:
            self._widget.close()
            self._widget = None
            self._completionOpenedManually = False

    def _wordBeforeCursor(self):
        """Get word, which is located before cursor
        """
        cursor = self._qpart.textCursor()
        textBeforeCursor = cursor.block().text()[:cursor.positionInBlock()]
        match = _wordAtEndRegExp.search(textBeforeCursor)
        if match:
            return match.group(0)
        else:
            return ''

    def _wordAfterCursor(self):
        """Get word, which is located before cursor
        """
        cursor = self._qpart.textCursor()
        textAfterCursor = cursor.block().text()[cursor.positionInBlock():]
        match = _wordAtStartRegExp.search(textAfterCursor)
        if match:
            return match.group(0)
        else:
            return ''

    def _onCompletionListItemSelected(self, index):
        """Item selected. Insert completion to editor
        """
        model = self._widget.model()
        selectedWord = model.words[index]
        textToInsert = selectedWord[len(model.typedText()):]
        self._qpart.textCursor().insertText(textToInsert)
        self._closeCompletion()

    def _onCompletionListTabPressed(self):
        """Tab pressed on completion list
        Insert completable text, if available
        """
        canCompleteText = self._widget.model().canCompleteText
        if canCompleteText:
            self._qpart.textCursor().insertText(canCompleteText)
            self.invokeCompletionIfAvailable()
