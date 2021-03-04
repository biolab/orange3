import numpy as np

from AnyQt.QtCore import (
    Qt, QLineF, QSize, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from AnyQt.QtGui import QPainter, QPen, QBrush, QColor
from AnyQt.QtWidgets import (
    QItemDelegate, QHeaderView, QPushButton, QApplication
)

from Orange.data import Table
from Orange.classification.rules import _RuleClassifier
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


class OWRuleViewer(widget.OWWidget):
    name = "CN2 Rule Viewer"
    description = "Review rules induced from data."
    icon = "icons/CN2RuleViewer.svg"
    priority = 1140
    keywords = []

    class Inputs:
        data = Input("Data", Table)
        classifier = Input("Classifier", _RuleClassifier)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    compact_view = settings.Setting(False)

    want_basic_layout = True
    want_main_area = False
    want_control_area = True

    def __init__(self):
        super().__init__()

        self.data = None
        self.classifier = None
        self.selected = None

        self.model = CustomRuleViewerTableModel(parent=self)
        self.model.set_horizontal_header_labels(
            ["IF conditions", "", "THEN class", "Distribution",
             "Probabilities [%]", "Quality", "Length"])

        self.proxy_model = QSortFilterProxyModel(parent=self)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortRole(self.model.SortRole)

        self.view = gui.TableView(self, wordWrap=False)
        self.view.setModel(self.proxy_model)
        self.view.verticalHeader().setVisible(True)
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.selectionModel().selectionChanged.connect(self.commit)

        self.dist_item_delegate = DistributionItemDelegate(self)
        self.view.setItemDelegateForColumn(3, self.dist_item_delegate)

        self.controlArea.layout().addWidget(self.view)

        gui.checkBox(widget=self.buttonsArea, master=self, value="compact_view",
                     label="Compact view", callback=self.on_update)
        gui.rubber(self.buttonsArea)

        original_order_button = gui.button(
            self.buttonsArea, self,
            "Restore original order",
            autoDefault=False,
            callback=self.restore_original_order,
            attribute=Qt.WA_LayoutUsesWidgetRect,
        )
        original_order_button.clicked.connect(self.restore_original_order)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.commit()

    @Inputs.classifier
    def set_classifier(self, classifier):
        self.classifier = classifier
        self.selected = None
        self.model.clear()

        if classifier is not None and hasattr(classifier, "rule_list"):
            self.model.set_vertical_header_labels(
                list(range(len(classifier.rule_list))))

            self.dist_item_delegate.color_schema = \
                [QColor(*c) for c in classifier.domain.class_var.colors]

            self.model.wrap(self.classifier.domain, self.classifier.rule_list)

        self.on_update()
        self.commit()

    def on_update(self):
        self._save_selected()

        self.model.set_compact_view(self.compact_view)
        if self.compact_view:
            self.view.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Interactive)  # QHeaderView.Stretch
        else:
            self.view.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents)
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()

        self._restore_selected()

    def _save_selected(self, actual=False):
        self.selected = None
        selection_model = self.view.selectionModel()
        if selection_model.hasSelection():
            if not actual:
                selection = selection_model.selection()
            else:
                selection = self.proxy_model.mapSelectionToSource(
                    selection_model.selection())

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
            output = "\n".join([str(self.classifier.rule_list[i])
                                for i in self.selected])
            QApplication.clipboard().setText(output)

    def commit(self):
        data_output = None
        self._save_selected(actual=True)
        selected_indices = []

        data = self.data or self.classifier and self.classifier.instances
        if (self.selected is not None and
                data is not None and
                self.classifier is not None and
                data.domain.attributes ==
                self.classifier.original_domain.attributes):

            status = np.ones(data.X.shape[0], dtype=bool)
            for i in self.selected:
                rule = self.classifier.rule_list[i]
                status &= rule.evaluate_data(data.X)

            selected_indices = status.nonzero()[0]
            data_output = data.from_table_rows(data, selected_indices) \
                if len(selected_indices) else None

        self.Outputs.selected_data.send(data_output)
        self.Outputs.annotated_data.send(create_annotated_table(data, selected_indices))

    def send_report(self):
        if self.classifier is not None:
            self.report_domain("Data domain", self.classifier.original_domain)
            self.report_items("Rule induction algorithm", self.classifier.params)
            self.report_table("Induced rules", self.view)

    def sizeHint(self):
        return QSize(800, 450)


class CustomRuleViewerTableModel(QAbstractTableModel):
    SortRole = Qt.UserRole + 1

    # ascii operators
    OPERATORS = {
        '==': '=',
        '!=': '≠',
        '<=': '≤',
        '>=': '≥'
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._domain = None
        self._rule_list = []
        self._compact_view = False
        self._headers = {}

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            headers = self._headers.get(orientation)
            return (headers[section] if headers and section < len(headers)
                    else str(section))
        return None

    def set_horizontal_header_labels(self, labels):
        self._headers[Qt.Horizontal] = labels

    def set_vertical_header_labels(self, labels):
        self._headers[Qt.Vertical] = labels

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers[Qt.Horizontal])

    def wrap(self, domain, rule_list):
        self.beginResetModel()
        self._domain = domain
        self._rule_list = rule_list
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._domain = None
        self._rule_list = []
        self.endResetModel()

    def set_compact_view(self, compact_view):
        self.beginResetModel()
        self._compact_view = compact_view
        self.endResetModel()

    def data(self, index, role=Qt.DisplayRole):
        if self._domain is None or not index.isValid():
            return None

        def _display_role():
            if column == 0:
                delim = " AND " if self._compact_view else " AND\n"
                return "TRUE" if not rule.selectors else delim.join(
                    [attributes[s.column].name + self.OPERATORS[s.op] +
                     (attributes[s.column].values[int(s.value)]
                      if attributes[s.column].is_discrete
                      else str(s.value)) for s in rule.selectors])
            if column == 1:
                return '→'
            if column == 2:
                return class_var.name + "=" + class_var.values[rule.prediction]
            if column == 3:
                # type(curr_class_dist) = ndarray
                return ([float(format(x, '.1f')) for x in rule.curr_class_dist]
                        if rule.curr_class_dist.dtype == float
                        else rule.curr_class_dist.tolist())
            if column == 4:
                return " : ".join(str(int(round(100 * x)))
                                  for x in rule.probabilities)
            if column == 5:
                value = rule.quality
                absval = abs(value)
                strlen = len(str(int(absval)))
                return '{:.{}{}}'.format(value,
                                         2 if absval < .001 else
                                         3 if strlen < 2 else
                                         1 if strlen < 5 else
                                         0 if strlen < 6 else
                                         3,
                                         'f' if (absval == 0 or
                                                 absval >= .001 and
                                                 strlen < 6)
                                         else 'e')
            if column == 6:
                return rule.length

            return None

        def _tooltip_role():
            if column == 0:
                return _display_role().replace(" AND ", " AND\n")
            if column == 1:
                return None
            if column == 3:
                # list of int, float
                curr_class_dist = _display_role()
                return class_var.name + "\n" + "\n".join(
                    (str(curr_class) + ": " + str(curr_class_dist[i])
                     for i, curr_class in enumerate(class_var.values)))
            if column == 4:
                return class_var.name + "\n" + "\n".join(
                    str(curr_class) + ": " +
                    '{:.1f}'.format(rule.probabilities[i] * 100) + "%"
                    for i, curr_class in enumerate(class_var.values))
            return _display_role()

        def _sort_role():
            if column == 0:
                return rule.length
            if column == 3:
                # return int, not np.int!
                return int(sum(rule.curr_class_dist))
            return _display_role()

        attributes = self._domain.attributes
        class_var = self._domain.class_var
        rule = self._rule_list[index.row()]
        column = index.column()

        if role == Qt.DisplayRole:
            return _display_role()
        if role == Qt.ToolTipRole:
            return _tooltip_role()
        if role == self.SortRole:
            return _sort_role()
        if role == Qt.TextAlignmentRole:
            return (Qt.AlignVCenter | [(Qt.AlignRight if self._compact_view
                                        else Qt.AlignLeft), Qt.AlignCenter,
                                       Qt.AlignLeft, Qt.AlignCenter,
                                       Qt.AlignCenter, Qt.AlignRight,
                                       Qt.AlignRight][column])

    def __len__(self):
        return len(self._rule_list)

    def __bool__(self):
        return len(self) != 0

    def __iter__(self):
        return iter(self._rule_list)

    def __getitem__(self, item):
        return self._rule_list[item]


class DistributionItemDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_schema = None

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setWidth(size.width() + 10)
        return size

    def paint(self, painter, option, index):
        curr_class_dist = np.array(index.data(Qt.DisplayRole), dtype=float)
        curr_class_dist /= sum(curr_class_dist)
        painter.save()
        self.drawBackground(painter, option, index)
        rect = option.rect

        if sum(curr_class_dist) > 0:
            pw = 3
            hmargin = 5
            x = rect.left() + hmargin
            width = rect.width() - 2 * hmargin
            vmargin = 1
            textoffset = pw + vmargin * 2
            painter.save()
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


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import CN2Learner
    data = Table("iris")
    learner = CN2Learner()
    model = learner(data)
    model.instances = data
    WidgetPreview(OWRuleViewer).run(model)
