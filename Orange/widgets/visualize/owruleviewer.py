import numpy as np

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import PyTableModel

from PyQt4.QtCore import Qt, QSize, pyqtSignal, QRectF, QLineF, QLocale
from PyQt4.QtGui import (QApplication, QLabel, QTableView, QStandardItem,
                         QStandardItemModel, QSortFilterProxyModel,
                         QMainWindow, QMouseEvent, QGraphicsView,
                         QPen, QBrush, QColor, QAbstractItemView,
                         QItemDelegate, QStyledItemDelegate, QPainter,
                         QFontMetrics, QStyle)


class Rule:  # temp - DELETE
    curr_class_dist = [0, 28, 0]

    def __str__(self):
        return "IF diau g>=0.172 AND Elu 210>=-0.069 THEN function=Resp"
rules = []


class OWRuleViewer(widget.OWWidget):
    name = "Rule Viewer"
    description = "Review rules induced from data."
    icon = ""
    priority = 18

    inputs = None
    outputs = None

    want_basic_layout = True
    want_main_area = True
    want_control_area = False

    UserAdviceMessages = [
        #  widget.Message()
    ]

    headers = ["IF cond", "", "THEN class", "quality", "class dist"]

    def __init__(self):
        raise NotImplementedError
        self.rule_list = None

        self.tableModel = PyTableModel(sequence=[], parent=self, editable=True)
        self.tableModel.setHorizontalHeaderLabels(self.headers)
        self.tableModel.dataChanged.connect(self.on_changed)

        self.tableView = gui.TableView
        self.tableView.setModel(self.tableModel)
        layout().addWidget(self.tableView)

        # out of date
        self.tableView = QTableView()
        self.tableView.setShowGrid(False)
        self.tableView.setSortingEnabled(True)
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setSelectionBehavior(QTableView.SelectRows)

        self.tableView.setItemDelegateForColumn(
            0, MultiLineStringItemDelegate(self))

        self.tableView.setItemDelegateForColumn(
            4, DistributionItemDelegate(self))

        self.tableView.setModel(self.tableModel)

        self.tableView.selectionModel().selectionChanged.connect(
            self.selection_changed)

        # self.updateVisibleColumns()

        # table = self.table = QTableView(
        #     self,
        #     showGrid=False,
        #     sortingEnabled=True,
        #     alternatingRowColors=True,
        #     selectionBehavior=QTableView.SelectRows,
        #     selectionMode=QTableView.ExtendedSelection,
        #     horizontalScrollMode=QTableView.ScrollPerPixel,
        #     verticalScrollMode=QTableView.ScrollPerPixel,
        #     editTriggers=QTableView.NoEditTriggers)

        # table.setItemDelegateForColumn(0, MultiLineStringItemDelegate(self))
        # # table.setItemDelegate(MultiLineStringItemDelegate(self))
        # table.verticalHeader().setVisible(False)
        # table.verticalHeader().setDefaultSectionSize(
        #     table.verticalHeader().minimumSectionSize())
        # table.horizontalHeader().setStretchLastSection(True)
        # table.setModel(QStandardItemModel(table))
        # table.setSelectionBehavior(QTableView.SelectRows)

        # self.view.selectionModel().selectionChanged.connect(self.selection_changed)
        # table.selectionChanged = QAbstractItemView.ExtendedSelection
        # table.selectionModel().selectionChanged.connect(self.selection_changed)

        self.mainArea.layout().addWidget(self.tableView)
        # self.set_table()

    def set_table(self):
        model = QStandardItemModel(self.table)
        for col, (label, tooltip) in enumerate(
                        [("IF cond", None),
                         ("", None),
                         ("THEN class", None),
                         ("Quality", None)]):
                    item = QStandardItem(label)
                    item.setToolTip(tooltip)
                    model.setHorizontalHeaderItem(col, item)

        NumericItem = self.NumericItem
        StandardItem = self.StandardItem

        for rule in rules:
            left_item = StandardItem("diau g>=0.172 AND\nElu 210>=-0.069", 2)
            left_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            middle_item = StandardItem('→')
            middle_item.setTextAlignment(Qt.AlignCenter)

            model.appendRow(
                [left_item,
                 middle_item,
                 StandardItem("survived=yes", 1),
                 NumericItem(0.92)]
            )

        table = self.table
        table.setHidden(True)
        table.setSortingEnabled(False)
        table.setModel(model)
        # for i in range(model.columnCount()):
        #     table.resizeColumnToContents(i)
        #     table.setColumnWidth(i, table.columnWidth(i) + 15)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setSortingEnabled(True)
        table.setHidden(False)

    def selection_changed(self, selected, deselected):
        pass

    class StandardItem(QStandardItem):
        def __init__(self, text):
            super().__init__(text)

    class NumericItem(StandardItem):
        def __init__(self, data):
            super().__init__('{:.2f}'.format(data), data)

        def __lt__(self, other):
            return self.data() < other.data()

    class ProxyModel(QSortFilterProxyModel):
        def filterAcceptsRow(self, row, parent):
            pass


class DistributionItemDelegate(QStyledItemDelegate):
    # taken from http://bit.ly/2b0NBqU

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_schema = None

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(int(size.height() * 1.3))
        size.setWidth(size.width() + 10)
        return size

    def paint(self, painter, option, index):
        dist = index.data(Qt.DisplayRole).toPyObject()
        num_covered = sum(dist)

        painter.save()
        self.drawBackground(painter, option, index)
        rect = option.rect
        # value = index.data(self.BarRole)
        value = dist
        if value is not None:
            pw = 3
            hmargin = 5
            x = rect.left() + hmargin
            width = rect.width() - 2 * hmargin
            vmargin = 1
            textoffset = pw + vmargin * 2
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            baseline = rect.bottom() - textoffset / 2
            fact = width / np.sum(value)
            for prop, color in zip(value, self.color_schema):
                if prop == 0:
                    continue
                painter.setPen(QPen(QBrush(color), pw))
                to_x = x + prop * fact
                line = QLineF(x, baseline, to_x, baseline)
                painter.drawLine(line)
                x = to_x
            painter.restore()
            text_rect = rect.adjusted(0, 0, 0, -textoffset)
        else:
            text_rect = rect
        text = str(index.data(Qt.DisplayRole))
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()


class MultiLineStringItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        metrics = QFontMetrics(option.font)
        text = index.data(Qt.DisplayRole)
        size = metrics.size(0, text)
        return QSize(size.width() + 8, size.height() + 8)  # 4 pixel margin

    def paint(self, painter, option, index):
        text = self.displayText(index.data(Qt.DisplayRole), QLocale())
        painter.save()

        QApplication.style().drawPrimitive(QStyle.PE_PanelItemViewRow,
                                           option, painter)
        QApplication.style().drawPrimitive(QStyle.PE_PanelItemViewItem,
                                           option, painter)

        temp = 4
        rect = option.rect.adjusted(temp, temp, -temp, -temp)

        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
            #        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color))

        painter.drawText(rect, option.displayAlignment, text)
        painter.restore()


if __name__ == "__main__":
    raise NotImplementedError

    from PyQt4.QtGui import QApplication

    a = QApplication([])
    ow = OWRuleViewer()

    # ow.set_data(data)
    #
    # ow.show()
    # a.exec()
