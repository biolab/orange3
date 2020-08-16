"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Bookmarks functionality implementation"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QKeySequence, QTextCursor

import qutepart


class Bookmarks:
    """Bookmarks functionality implementation, grouped in one class
    """
    def __init__(self, qpart, markArea):
        self._qpart = qpart
        self._markArea = markArea
        qpart.toggleBookmarkAction = self._createAction(qpart, "emblem-favorite", "Toogle bookmark", 'Ctrl+B',
                                                        self._onToggleBookmark)
        qpart.prevBookmarkAction = self._createAction(qpart, "go-up", "Previous bookmark", 'Alt+PgUp',
                                                      self._onPrevBookmark)
        qpart.nextBookmarkAction = self._createAction(qpart, "go-down", "Next bookmark", 'Alt+PgDown',
                                                      self._onNextBookmark)

        markArea.blockClicked.connect(self._toggleBookmark)

    def _createAction(self, widget, iconFileName, text, shortcut, slot):
        """Create QAction with given parameters and add to the widget
        """
        icon = qutepart.getIcon(iconFileName)
        action = QAction(icon, text, widget)
        action.setShortcut(QKeySequence(shortcut))
        action.setShortcutContext(Qt.WidgetShortcut)
        action.triggered.connect(slot)

        widget.addAction(action)

        return action

    def removeActions(self):
        self._qpart.removeAction(self._qpart.toggleBookmarkAction)
        self._qpart.toggleBookmarkAction = None
        self._qpart.removeAction(self._qpart.prevBookmarkAction)
        self._qpart.prevBookmarkAction = None
        self._qpart.removeAction(self._qpart.nextBookmarkAction)
        self._qpart.nextBookmarkAction = None

    def clear(self, startBlock, endBlock):
        """Clear bookmarks on block range including start and end
        """
        for block in qutepart.iterateBlocksFrom(startBlock):
            self._setBlockMarked(block, False)
            if block == endBlock:
                break

    def isBlockMarked(self, block):
        """Check if block is bookmarked
        """
        return self._markArea.isBlockMarked(block)

    def _setBlockMarked(self, block, marked):
        """Set block bookmarked
        """
        self._markArea.setBlockValue(block, 1 if marked else 0)

    def _toggleBookmark(self, block):
        self._markArea.toggleBlockMark(block)
        self._markArea.update()

    def _onToggleBookmark(self):
        """Toogle Bookmark action triggered
        """
        self._toggleBookmark(self._qpart.textCursor().block())

    def _onPrevBookmark(self):
        """Previous Bookmark action triggered. Move cursor
        """
        for block in qutepart.iterateBlocksBackFrom(self._qpart.textCursor().block().previous()):
            if self.isBlockMarked(block):
                self._qpart.setTextCursor(QTextCursor(block))
                return

    def _onNextBookmark(self):
        """Previous Bookmark action triggered. Move cursor
        """
        for block in qutepart.iterateBlocksFrom(self._qpart.textCursor().block().next()):
            if self.isBlockMarked(block):
                self._qpart.setTextCursor(QTextCursor(block))
                return
