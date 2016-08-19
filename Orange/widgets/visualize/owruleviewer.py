import numpy as np

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.classification.rules import _RuleClassifier

from PyQt4.QtCore import Qt, QLineF
from PyQt4.QtGui import (QSortFilterProxyModel, QPainter, QPen, QBrush, QColor,
                         QItemDelegate, QStyledItemDelegate, QHeaderView,
                         QPushButton, QApplication)


class OWRuleViewer(widget.OWWidget):
    name = "Rule Viewer"
    description = "Review rules induced from data."
    icon = "icons/CN2RuleViewer.svg"
    priority = 18

    inputs = [("Data", Table, 'set_data'),
              ("Classifier", _RuleClassifier, 'set_classifier')]

    data_output_identifier = "Filtered data"
    outputs = [(data_output_identifier, Table)]

    compact_view = settings.Setting(False)

    want_basic_layout = True
    want_main_area = True
    want_control_area = False

    # ascii operators
    OPERATORS = {
        '==': '=',
        '!=': '≠',
        '<=': '≤',
        '>=': '≥'
    }

    def __init__(self):
        self.data = None
        self.classifier = None
        self.presentation = None
        self.selected = None

        self.model = CustomRuleViewerPyTableModel(parent=self, editable=False)
        self.model.setHorizontalHeaderLabels(
            ["IF conditions", "", "THEN class", "Distribution",
             "Probabilities", "Quality", "Length"])

        self.proxy_model = QSortFilterProxyModel(parent=self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortRole(self.model.SortRole)

        self.view = gui.TableView(self)
        self.view.setModel(self.proxy_model)
        self.view.verticalHeader().setVisible(True)
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.selectionModel().selectionChanged.connect(self.commit)

        self.bold_item_delegate = BoldMiddleAlignedFontDelegate(self)
        self.dist_item_delegate = DistributionItemDelegate(self)
        self.middle_item_delegate = MiddleAlignedFontDelegate(self)

        self.view.setItemDelegateForColumn(1, self.bold_item_delegate)
        self.view.setItemDelegateForColumn(2, self.middle_item_delegate)
        self.view.setItemDelegateForColumn(3, self.dist_item_delegate)
        self.view.setItemDelegateForColumn(4, self.middle_item_delegate)

        self.mainArea.layout().setContentsMargins(0, 0, 0, 0)
        self.mainArea.layout().addWidget(self.view)

        bottom_box = gui.hBox(widget=self.mainArea, box=None,
                              margin=0, spacing=0)

        original_order_button = QPushButton(
            "Restore original order", autoDefault=False)
        original_order_button.setFixedWidth(180)
        bottom_box.layout().addWidget(original_order_button)
        original_order_button.clicked.connect(self.restore_original_order)

        gui.separator(bottom_box, width=5, height=0)
        gui.checkBox(widget=bottom_box, master=self, value="compact_view",
                     label="Compact view", callback=self.on_update)

        self.report_button.setFixedWidth(180)
        bottom_box.layout().addWidget(self.report_button)

    def set_data(self, data):
        self.data = data
        self.commit()

    def set_classifier(self, classifier):
        self.classifier = classifier
        self.presentation = None
        self.selected = None
        self.model.domain = None
        self.model.clear()

        if classifier is not None and hasattr(classifier, "rule_list"):
            self.model.domain = self.classifier.domain
            self.model.setVerticalHeaderLabels(
                list(range(len(classifier.rule_list))))

            self.dist_item_delegate.color_schema = \
                [QColor(*c) for c in classifier.domain.class_var.colors]

            attributes = classifier.domain.attributes
            class_var = classifier.domain.class_var

            self.presentation = [
                ["TRUE" if not rule.selectors else " AND ".join(
                    [attributes[s.column].name + self.OPERATORS[s.op] +
                     (str(attributes[s.column].values[int(s.value)])
                      if attributes[s.column].is_discrete
                      else str(s.value)) for s in rule.selectors]),
                 '→', class_var.name + "=" + class_var.values[rule.prediction],
                 [float('{:.1f}'.format(x)) for x in rule.curr_class_dist]
                 if rule.curr_class_dist.dtype == float
                 else rule.curr_class_dist.tolist(),
                 [float('{:.3f}'.format(x)) for x in rule.probabilities],
                 rule.quality, rule.length] for rule in classifier.rule_list]

        self.on_update()
        self.commit()

    def on_update(self):
        if self.presentation is not None:
            self._update_presentation()
            self._save_selected()

            self.model.wrap(self.presentation)
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

            self._restore_selected()

    def _update_presentation(self):
        assert self.presentation is not None
        to_replace = ((" AND\n", " AND ")
                      if self.compact_view
                      else (" AND ", " AND\n"))

        for single_rule_str_list in self.presentation:
            antecedent = single_rule_str_list[0]
            single_rule_str_list[0] = antecedent.replace(*to_replace)

    def _save_selected(self, actual=False):
        self.selected = None
        selection_model = self.view.selectionModel()
        if selection_model.hasSelection():
            selection = (selection_model.selection() if not actual
                         else self.proxy_model.mapSelectionToSource(
                                selection_model.selection()))

            self.selected = sorted(set(index.row() for index
                                       in selection.indexes()))

    def _restore_selected(self):
        if self.selected is not None:
            selection_model = self.view.selectionModel()
            for row in self.selected:
                selection_model.select(self.proxy_model.index(row, 0),
                                       selection_model.Select |
                                       selection_model.Rows)

    def restore_original_order(self):
        self.proxy_model.sort(-1)

    def copy_to_clipboard(self):
        self._save_selected(actual=True)
        if self.selected is not None:
            output = "\n".join([self.classifier.rule_list[i].__str__()
                                for i in self.selected])
            QApplication.clipboard().setText(output)

    def commit(self):
        data_output = None
        self._save_selected(actual=True)

        if (self.selected is not None and
                self.data is not None and
                self.classifier is not None and
                self.data.domain.attributes ==
                self.classifier.original_domain.attributes):

            status = np.ones(self.data.X.shape[0], dtype=bool)
            for i in self.selected:
                rule = self.classifier.rule_list[i]
                status &= rule.evaluate_data(self.data.X)

            data_output = self.data.from_table_rows(
                self.data, status.nonzero()[0])

        self.send(OWRuleViewer.data_output_identifier, data_output)

    def send_report(self):
        if self.classifier is not None:
            self.report_domain("Data domain", self.classifier.original_domain)
            self.report_items("Rule induction algorithm", self.classifier.params)
            self.report_table("Induced rules", self.view)


class CustomRuleViewerPyTableModel(PyTableModel):
    SortRole = Qt.UserRole + 1

    def data(self, index, role=Qt.DisplayRole):
        column = index.column()
        # tooltip: conditions column, each selector in its own line
        if role == Qt.ToolTipRole and column == 0:
            value = super().data(index, role)
            value = value.replace(" AND ", " AND\n")
        # tooltip: distribution column
        elif role == Qt.ToolTipRole and column == 3:
            value = super().data(index, Qt.EditRole)
            value = self.domain.class_var.name + "\n" + "\n".join(
                (str(curr_class) + ": " + str(value[i]) for i, curr_class
                 in enumerate(self.domain.class_var.values)))
        # tooltip: probabilities column
        elif role == Qt.ToolTipRole and column == 4:
            value = super().data(index, Qt.EditRole)
            value = self.domain.class_var.name + "\n" + "\n".join(
                (str(curr_class) + ": " + '{:.1f}'.format(value[i]*100) + "%"
                 for i, curr_class in enumerate(self.domain.class_var.values)))
        # display: probabilities column
        elif role == Qt.DisplayRole and column == 4:
            value = super().data(index, Qt.EditRole)
            value = " : ".join('{:.2f}'.format(x) for x in value)
        # summation to sort the distribution column (total coverage)
        elif role == self.SortRole:
            value = (sum(super().data(index, Qt.EditRole)) if column == 3
                     else super().data(index, Qt.DisplayRole))
        else:
            value = super().data(index, role)
        return value


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
