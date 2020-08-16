"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
from PyQt5.QtGui import QTextCursor

# Lines class.
# list-like object for access text document lines


def _iterateBlocksFrom(block):
    while block.isValid():
        yield block
        block = block.next()


def _atomicModification(func):
    """Decorator
    Make document modification atomic
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        with self._qpart:  # pylint: disable=protected-access
            func(*args, **kwargs)
    return wrapper


class Lines:
    """list-like object for access text document lines
    """
    def __init__(self, qpart):
        self._qpart = qpart
        self._doc = qpart.document()

    def setDocument(self, document):
        self._doc = document

    def _toList(self):
        """Convert to Python list
        """
        return [block.text() \
                    for block in _iterateBlocksFrom(self._doc.firstBlock())]

    def __str__(self):
        """Serialize
        """
        return str(self._toList())

    def __len__(self):
        """Get lines count
        """
        return self._doc.blockCount()

    def _checkAndConvertIndex(self, index):
        """Check integer index, convert from less than zero notation
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= self._doc.blockCount():
            raise IndexError('Invalid block index', index)
        return index

    def __getitem__(self, index):
        """Get item by index
        """
        def _getTextByIndex(blockIndex):
            return self._doc.findBlockByNumber(blockIndex).text()

        if isinstance(index, int):
            index = self._checkAndConvertIndex(index)
            return _getTextByIndex(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self._doc.blockCount())
            return [_getTextByIndex(blockIndex) \
                        for blockIndex in range(start, stop, step)]

    @_atomicModification
    def __setitem__(self, index, value):
        """Set item by index
        """
        def _setBlockText(blockIndex, text):
            cursor = QTextCursor(self._doc.findBlockByNumber(blockIndex))
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.insertText(text)

        if isinstance(index, int):
            index = self._checkAndConvertIndex(index)
            _setBlockText(index, value)
        elif isinstance(index, slice):
            # List of indexes is reversed for make sure
            # not processed indexes are not shifted during document modification
            start, stop, step = index.indices(self._doc.blockCount())
            if step > 0:
                start, stop, step = stop - 1, start - 1, step * -1

            blockIndexes = list(range(start, stop, step))

            if len(blockIndexes) != len(value):
                raise ValueError('Attempt to replace %d lines with %d lines' %
                                 (len(blockIndexes), len(value)))

            for blockIndex, text in zip(blockIndexes, value[::-1]):
                _setBlockText(blockIndex, text)

    @_atomicModification
    def __delitem__(self, index):
        """Delete item by index
        """
        def _removeBlock(blockIndex):
            block = self._doc.findBlockByNumber(blockIndex)
            if block.next().isValid():  # not the last
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            elif block.previous().isValid():  # the last, not the first
                cursor = QTextCursor(block.previous())
                cursor.movePosition(QTextCursor.EndOfBlock)
                cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            else:  # only one block
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()

        if isinstance(index, int):
            index = self._checkAndConvertIndex(index)
            _removeBlock(index)
        elif isinstance(index, slice):
            # List of indexes is reversed for make sure
            # not processed indexes are not shifted during document modification
            start, stop, step = index.indices(self._doc.blockCount())
            if step > 0:
                start, stop, step = stop - 1, start - 1, step * -1

            for blockIndex in range(start, stop, step):
                _removeBlock(blockIndex)

    class _Iterator:
        """Blocks iterator. Returns text
        """
        def __init__(self, block):
            self._block = block

        def __iter__(self):
            return self

        def __next__(self):
            if self._block.isValid():
                self._block, result = self._block.next(), self._block.text()
                return result
            else:
                raise StopIteration()

    def __iter__(self):
        """Return iterator object
        """
        return self._Iterator(self._doc.firstBlock())

    @_atomicModification
    def append(self, text):
        """Append line to the end
        """
        cursor = QTextCursor(self._doc)
        cursor.movePosition(QTextCursor.End)
        cursor.insertBlock()
        cursor.insertText(text)

    @_atomicModification
    def insert(self, index, text):
        """Insert line to the document
        """
        if index < 0 or index > self._doc.blockCount():
            raise IndexError('Invalid block index', index)

        if index == 0:  # first
            cursor = QTextCursor(self._doc.firstBlock())
            cursor.insertText(text)
            cursor.insertBlock()
        elif index != self._doc.blockCount():  # not the last
            cursor = QTextCursor(self._doc.findBlockByNumber(index).previous())
            cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.insertBlock()
            cursor.insertText(text)
        else:  # last append to the end
            self.append(text)
