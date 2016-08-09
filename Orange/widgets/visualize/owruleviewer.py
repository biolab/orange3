import numpy as np

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.classification.rules import _RuleClassifier

from PyQt4 import QtGui

from PyQt4.QtCore import Qt, QSize, pyqtSignal, QRectF, QLineF, QLocale, QRect
from PyQt4.QtGui import (QApplication, QLabel, QTableView, QStandardItem,
                         QStandardItemModel, QSortFilterProxyModel, QAction,
                         QMainWindow, QMouseEvent, QGraphicsView, QPainter,
                         QPen, QBrush, QColor, QAbstractItemView, QStyle,
                         QItemDelegate, QStyledItemDelegate, QHeaderView,
                         QFontMetrics, QTextOption)


class OWRuleViewer(widget.OWWidget):
    name = "Rule Viewer"
    description = "Review rules induced from data."
    icon = ""
    priority = 18

    inputs = [("Classifier", _RuleClassifier, 'set_classifier')]

    # ascii operators
    OPERATORS = {
        '==': '=',
        '!=': '≠',
        '<=': '≤',
        '>=': '≥'
    }

    want_basic_layout = True
    want_main_area = True
    want_control_area = False

    presentation = None
    compact_view = False

    def __init__(self):
        self.classifier = None

        self.model = PyTableModel(parent=self, editable=False)
        self.model.setHorizontalHeaderLabels(
            ["IF conditions", "", "THEN class",
             "Distribution", "Rule quality", "Rule length"])

        self.view = gui.TableView(self)
        self.view.verticalHeader().setVisible(True)
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.setModel(self.model)

        self.bold_item_delegate = BoldMiddleAlignedFontDelegate(self)
        self.dist_item_delegate = DistributionItemDelegate(self)
        self.middle_item_delegate = MiddleAlignedFontDelegate(self)

        self.view.setItemDelegateForColumn(1, self.bold_item_delegate)
        self.view.setItemDelegateForColumn(2, self.middle_item_delegate)
        self.view.setItemDelegateForColumn(3, self.dist_item_delegate)

        self.mainArea.layout().addWidget(self.view)
        gui.checkBox(widget=self.mainArea, master=self, value="compact_view",
                     label="Compact view", callback=self.on_update)

    def set_classifier(self, classifier):
        self.classifier = classifier
        self.model.clear()
        self.on_update()

    def on_update(self):
        if self.classifier is not None and hasattr(self.classifier, 'rule_list'):
            self.model.setVerticalHeaderLabels(
                [str(i) for i in range(len(self.classifier.rule_list))])

            self.dist_item_delegate.color_schema = \
                [QColor(*c) for c in self.classifier.domain.class_var.colors]

            attributes = self.classifier.domain.attributes
            class_var = self.classifier.domain.class_var

            presentation = [
                ("TRUE" if not rule.selectors
                 else (" AND " if self.compact_view else " AND\n").join(
                    [attributes[s.column].name +
                     self.OPERATORS[s.op] +
                     (str(attributes[s.column].values[int(s.value)])
                      if attributes[s.column].is_discrete
                      else str(s.value)) for s in rule.selectors]),
                 '→', class_var.name + "=" + class_var.values[rule.prediction],
                 rule.curr_class_dist.tolist(), rule.quality, rule.length)
                for rule in self.classifier.rule_list]

            self.model.wrap(presentation)
            self.view.resizeColumnsToContents()
            self.view.resizeRowsToContents()

            if self.compact_view:
                self.view.setWordWrap(False)
                self.view.horizontalHeader().setResizeMode(
                    0, QHeaderView.Interactive)

                if self.view.horizontalHeader().sectionSize(0) > 250:
                    self.view.horizontalHeader().resizeSection(0, 240)
            else:
                self.view.setWordWrap(True)
                self.view.horizontalHeader().setResizeMode(
                    QHeaderView.ResizeToContents)

    def copy_to_clipboard(self):
        if self.view is not None and self.view.selectionmodel().hasSelection():
            highlighted_rows_indices = set(index.row() for index in
                                           self.view.selectedIndexes())

            actual_rows_indices = sorted([self.model.headerData(i, Qt.Vertical)
                                          for i in highlighted_rows_indices])

            output = "\n".join([self.classifier.rule_list[int(i)].__str__()
                                for i in actual_rows_indices])
            print(output)
            QApplication.clipboard().setText(output)


class DistributionItemDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_schema = None

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        # size.setHeight(int(size.height() * 1.3))
        size.setWidth(size.width() + 10)
        return size

    def paint(self, painter, option, index):
        curr_class_dist = np.array(index.data(Qt.EditRole), dtype=float)
        curr_class_dist /= curr_class_dist.sum()
        painter.save()
        self.drawBackground(painter, option, index)
        rect = option.rect

        if curr_class_dist.sum() > 0:
            pw = 3
            hmargin = 5
            x = rect.left() + hmargin
            width = rect.width() - 2 * hmargin
            vmargin = 1
            textoffset = pw + vmargin * 2
            painter.save()
            # baseline = rect.center().y() + textoffset * 2.5
            baseline = rect.bottom() - textoffset / 2

            text = str(index.data(Qt.DisplayRole))
            option.displayAlignment = Qt.AlignCenter
            text_rect = rect.adjusted(0, 0, 0, -textoffset*0)
            self.drawDisplay(painter, option, text_rect, text)

            painter.setRenderHint(QPainter.Antialiasing)
            for prop, color in zip(curr_class_dist, self.color_schema):
                if prop == 0:
                    continue
                painter.setPen(QPen(QBrush(color), pw))
                to_x = x + prop * width
                line = QLineF(x, baseline, to_x, baseline)
                painter.drawLine(line)
                x = to_x
            painter.restore()

        painter.restore()


class BoldMiddleAlignedFontDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self._callback = callback

    def paint(self, painter, option, index):
        if not callable(self._callback) or self._callback(index):
            option.font.setWeight(option.font.Bold)
            option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        if not callable(self._callback) or self._callback(index):
            option.font.setWeight(option.font.Bold)
        return super().sizeHint(option, index)


class MiddleAlignedFontDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self._callback = callback

    def paint(self, painter, option, index):
        if not callable(self._callback) or self._callback(index):
            option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        return super().sizeHint(option, index)


if __name__ == "__main__":
    from PyQt4.QtGui import QApplication

    a = QApplication([])
    ow = OWRuleViewer()

    ow.show()
    a.exec()
