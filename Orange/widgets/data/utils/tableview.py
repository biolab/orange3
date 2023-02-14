import sys

from AnyQt.QtCore import Qt, QObject, QEvent, QSize, QAbstractProxyModel
from AnyQt.QtGui import QPainter
from AnyQt.QtWidgets import (
    QStyle, QWidget, QStyleOptionHeader, QAbstractButton
)

from Orange.widgets.data.utils.models import RichTableModel
from Orange.widgets.utils.tableview import TableView


class DataTableView(TableView):
    """
    A TableView with settable corner text.
    """
    class __CornerPainter(QObject):
        def drawCorner(self, button: QWidget):
            opt = QStyleOptionHeader()
            view = self.parent()
            assert isinstance(view, DataTableView)
            header = view.horizontalHeader()
            opt.initFrom(header)
            state = QStyle.State_None
            if button.isEnabled():
                state |= QStyle.State_Enabled
            if button.isActiveWindow():
                state |= QStyle.State_Active
            if button.isDown():
                state |= QStyle.State_Sunken
            opt.state = state
            opt.rect = button.rect()
            opt.text = button.text()
            opt.position = QStyleOptionHeader.OnlyOneSection
            style = header.style()
            painter = QPainter(button)
            style.drawControl(QStyle.CE_Header, opt, painter, header)

        def eventFilter(self, receiver: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Paint:
                self.drawCorner(receiver)
                return True
            return super().eventFilter(receiver, event)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cornerText = ""
        self.__cornerButton = btn = self.findChild(QAbstractButton)
        self.__cornerButtonFilter = DataTableView.__CornerPainter(self)
        btn.installEventFilter(self.__cornerButtonFilter)
        btn.disconnect()
        btn.clicked.connect(self.cornerButtonClicked)
        if sys.platform == "darwin":
            btn.setAttribute(Qt.WA_MacSmallSize)

    def setCornerText(self, text: str) -> None:
        """Set the corner text."""
        self.__cornerButton.setText(text)
        self.__cornerText = text
        self.__cornerButton.update()
        assert self.__cornerButton is self.findChild(QAbstractButton)
        opt = QStyleOptionHeader()
        opt.initFrom(self.__cornerButton)
        opt.text = text
        s = self.__cornerButton.style().sizeFromContents(
            QStyle.CT_HeaderSection, opt, QSize(), self.__cornerButton
        )
        if s.isValid():
            self.verticalHeader().setMinimumWidth(s.width())

    def cornerText(self):
        """Return the corner text."""
        return self.__cornerText

    def cornerButtonClicked(self):
        model = self.model()
        selection = self.selectionModel()
        selection = selection.selection()
        if len(selection) == 1:
            srange = selection[0]
            if srange.top() == 0 and srange.left() == 0 \
                    and srange.right() == model.columnCount() - 1 \
                    and srange.bottom() == model.rowCount() - 1:
                self.clearSelection()
            else:
                self.selectAll()
        else:
            self.selectAll()


class RichTableView(DataTableView):
    """
    The preferred table view for RichTableModel.

    Handles the display of variable's labels keys in top left corner.
    """
    def setModel(self, model: RichTableModel):
        current = self.model()
        if current is not None:
            current.headerDataChanged.disconnect(self.__headerDataChanged)
        super().setModel(model)
        if model is not None:
            model.headerDataChanged.connect(self.__headerDataChanged)
            self.__headerDataChanged(Qt.Horizontal)

    def __headerDataChanged(
            self,
            orientation: Qt.Orientation,
    ) -> None:
        if orientation == Qt.Horizontal:
            model = self.model()
            while isinstance(model, QAbstractProxyModel):
                model = model.sourceModel()
            if model.richHeaderFlags() & RichTableModel.Labels:
                items = model.headerData(
                    0, Qt.Horizontal, RichTableModel.LabelsItemsRole
                )
                text = "\n"
                text += "\n".join(key for key, _ in items)
            else:
                text = ""
            self.setCornerText(text)
