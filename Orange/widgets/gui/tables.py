from AnyQt.QtWidgets import (
    QTableView, QHeaderView, QStyledItemDelegate, QMenu, QAction
)
from AnyQt.QtGui import QContextMenuEvent
from AnyQt.QtCore import QObject, Qt

__all__ = ["TableView", "VisibleHeaderSectionContextEventFilter"]


class TableView(QTableView):
    """An auxilliary table view for use with PyTableModel in control areas"""
    def __init__(self, parent=None, **kwargs):
        kwargs = dict(
            dict(showGrid=False,
                 sortingEnabled=True,
                 cornerButtonEnabled=False,
                 alternatingRowColors=True,
                 selectionBehavior=self.SelectRows,
                 selectionMode=self.ExtendedSelection,
                 horizontalScrollMode=self.ScrollPerPixel,
                 verticalScrollMode=self.ScrollPerPixel,
                 editTriggers=self.DoubleClicked | self.EditKeyPressed),
            **kwargs)
        super().__init__(parent, **kwargs)
        h = self.horizontalHeader()
        h.setCascadingSectionResizes(True)
        h.setMinimumSectionSize(-1)
        h.setStretchLastSection(True)
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        v = self.verticalHeader()
        v.setVisible(False)
        v.setSectionResizeMode(QHeaderView.ResizeToContents)

    class BoldFontDelegate(QStyledItemDelegate):
        """Paints the text of associated cells in bold font.

        Can be used e.g. with QTableView.setItemDelegateForColumn() to make
        certain table columns bold, or if callback is provided, the item's
        model index is passed to it, and the item is made bold only if the
        callback returns true.

        Parameters
        ----------
        parent: QObject
            The parent QObject.
        callback: callable
            Accepts model index and returns True if the item is to be
            rendered in bold font.
        """
        def __init__(self, parent=None, callback=None):
            super().__init__(parent)
            self._callback = callback

        def paint(self, painter, option, index):
            """Paint item text in bold font"""
            if not callable(self._callback) or self._callback(index):
                option.font.setWeight(option.font.Bold)
            super().paint(painter, option, index)

        def sizeHint(self, option, index):
            """Ensure item size accounts for bold font width"""
            if not callable(self._callback) or self._callback(index):
                option.font.setWeight(option.font.Bold)
            return super().sizeHint(option, index)


class VisibleHeaderSectionContextEventFilter(QObject):
    def __init__(self, parent, itemView=None):
        super().__init__(parent)
        self.itemView = itemView

    def eventFilter(self, view, event):
        if not isinstance(event, QContextMenuEvent):
            return False

        model = view.model()
        headers = [(view.isSectionHidden(i),
                    model.headerData(i, view.orientation(), Qt.DisplayRole))
                   for i in range(view.count())]
        menu = QMenu("Visible headers", view)

        for i, (checked, name) in enumerate(headers):
            action = QAction(name, menu)
            action.setCheckable(True)
            action.setChecked(not checked)
            menu.addAction(action)

            def toogleHidden(visible, section=i):
                view.setSectionHidden(section, not visible)
                if not visible:
                    return
                if self.itemView:
                    self.itemView.resizeColumnToContents(section)
                else:
                    view.resizeSection(section,
                                       max(view.sectionSizeHint(section), 10))

            action.toggled.connect(toogleHidden)
        menu.exec_(event.globalPos())
        return True
