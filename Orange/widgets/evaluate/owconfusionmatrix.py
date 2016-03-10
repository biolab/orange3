import unicodedata

from PyQt4.QtGui import (
    QGridLayout, QTableView, QStandardItemModel, QStandardItem,
    QItemSelectionModel, QItemSelection, QFont, QHeaderView, QBrush, QColor,
    QStyledItemDelegate)
from PyQt4.QtCore import Qt, QSize

import numpy
import sklearn.metrics as skl_metrics
from math import isnan, isinf

import Orange
from Orange.widgets import widget, settings, gui


def confusion_matrix(res, index):
    return skl_metrics.confusion_matrix(
        res.actual, res.predicted[index])


BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)


class BorderedItemDelegate(QStyledItemDelegate):
    def __init__(self, color=Qt.black):
        super().__init__()
        self.color = color

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
        borders = index.data(BorderRole)
        if borders:
            color = index.data(BorderColorRole) or self.color
            painter.save()
            painter.setPen(color)
            r = option.rect
            for side, p1, p2 in (("t", r.topLeft(), r.topRight()),
                                 ("r", r.topRight(), r.bottomRight()),
                                 ("b", r.bottomLeft(), r.bottomRight()),
                                 ("l", r.topLeft(), r.bottomLeft())):
                if side in borders:
                    painter.drawLine(p1, p2)
            painter.restore()


class OWConfusionMatrix(widget.OWWidget):
    name = "Confusion Matrix"
    description = "Display confusion matrix constructed from results " \
                  "of evaluation of classifiers."
    icon = "icons/ConfusionMatrix.svg"
    priority = 1001

    inputs = [("Evaluation Results", Orange.evaluation.Results, "set_results")]
    outputs = [("Selected Data", Orange.data.Table)]

    quantities = ["Number of instances",
                  "Proportion of predicted",
                  "Proportion of actual"]

    selected_learner = settings.Setting([])
    selected_quantity = settings.Setting(0)
    append_predictions = settings.Setting(True)
    append_probabilities = settings.Setting(False)
    autocommit = settings.Setting(True)

    UserAdviceMessages = [
        widget.Message(
                "Clicking on cells or in headers outputs the corresponding "
                "data instances",
                "click_cell")]

    def __init__(self):
        super().__init__()

        self.data = None
        self.results = None
        self.learners = []
        self.headers = []

        box = gui.widgetBox(self.controlArea, "Learners")

        self.learners_box = gui.listBox(
            box, self, "selected_learner", "learners",
            callback=self._learner_changed
        )
        box = gui.widgetBox(self.controlArea, "Show")

        gui.comboBox(box, self, "selected_quantity", items=self.quantities,
                     callback=self._update)

        box = gui.widgetBox(self.controlArea, "Select")

        gui.button(box, self, "Correct",
                   callback=self.select_correct, autoDefault=False)
        gui.button(box, self, "Misclassified",
                   callback=self.select_wrong, autoDefault=False)
        gui.button(box, self, "None",
                   callback=self.select_none, autoDefault=False)

        self.outputbox = box = gui.widgetBox(self.controlArea, "Output")
        gui.checkBox(box, self, "append_predictions",
                     "Predictions", callback=self._invalidate)
        gui.checkBox(box, self, "append_probabilities",
                     "Probabilities",
                     callback=self._invalidate)

        gui.auto_commit(self.controlArea, self, "autocommit",
                        "Send Data", "Auto send is on")

        grid = QGridLayout()

        self.tablemodel = QStandardItemModel(self)
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setModel(self.tablemodel)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        view.horizontalHeader().setMinimumSectionSize(60)
        view.selectionModel().selectionChanged.connect(self._invalidate)
        view.setShowGrid(False)
        view.setItemDelegate(BorderedItemDelegate(Qt.white))
        view.clicked.connect(self.cell_clicked)
        grid.addWidget(view, 0, 0)
        self.mainArea.layout().addLayout(grid)

    def sizeHint(self):
        return QSize(750, 490)

    def _item(self, i, j):
        return self.tablemodel.item(i, j) or QStandardItem()

    def _set_item(self, i, j, item):
        self.tablemodel.setItem(i, j, item)

    def set_results(self, results):
        """Set the input results."""

        self.clear()
        self.warning([0, 1])

        data = None
        if results is not None:
            if results.data is not None:
                data = results.data

        if data is not None and not data.domain.has_discrete_class:
            data = None
            results = None
            self.warning(
                0, "Confusion Matrix cannot be used for regression results.")

        self.results = results
        self.data = data

        if data is not None:
            class_values = data.domain.class_var.values
        elif results is not None:
            raise NotImplementedError

        if results is None:
            self.report_button.setDisabled(True)
        else:
            self.report_button.setDisabled(False)

            nmodels, ntests = results.predicted.shape
            self.headers = class_values + \
                           [unicodedata.lookup("N-ARY SUMMATION")]

            # NOTE: The 'learner_names' is set in 'Test Learners' widget.
            if hasattr(results, "learner_names"):
                self.learners = results.learner_names
            else:
                self.learners = ["Learner #%i" % (i + 1)
                                 for i in range(nmodels)]

            item = self._item(0, 2)
            item.setData("Predicted", Qt.DisplayRole)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(Qt.NoItemFlags)

            self._set_item(0, 2, item)
            item = self._item(2, 0)
            item.setData("Actual", Qt.DisplayRole)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setFlags(Qt.NoItemFlags)
            self.tableview.setItemDelegateForColumn(
                0, gui.VerticalItemDelegate())
            self._set_item(2, 0, item)
            self.tableview.setSpan(0, 2, 1, len(class_values))
            self.tableview.setSpan(2, 0, len(class_values), 1)

            font = self.tablemodel.invisibleRootItem().font()
            bold_font = QFont(font)
            bold_font.setBold(True)

            for i in (0, 1):
                for j in (0, 1):
                    item = self._item(i, j)
                    item.setFlags(Qt.NoItemFlags)
                    self._set_item(i, j, item)

            for p, label in enumerate(self.headers):
                for i, j in ((1, p + 2), (p + 2, 1)):
                    item = self._item(i, j)
                    item.setData(label, Qt.DisplayRole)
                    item.setFont(bold_font)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled)
                    if p < len(self.headers) - 1:
                        item.setData("br"[j == 1], BorderRole)
                        item.setData(QColor(192, 192, 192), BorderColorRole)
                    self._set_item(i, j, item)

            hor_header = self.tableview.horizontalHeader()
            if len(' '.join(self.headers)) < 120:
                hor_header.setResizeMode(QHeaderView.ResizeToContents)
            else:
                hor_header.setDefaultSectionSize(60)
            self.tablemodel.setRowCount(len(class_values) + 3)
            self.tablemodel.setColumnCount(len(class_values) + 3)
            self.selected_learner = [0]
            self._update()

    def clear(self):
        self.results = None
        self.data = None
        self.tablemodel.clear()
        self.headers = []
        # Clear learners last. This action will invoke `_learner_changed`
        # method
        self.learners = []

    def select_correct(self):
        selection = QItemSelection()
        n = self.tablemodel.rowCount()
        for i in range(2, n):
            index = self.tablemodel.index(i, i)
            selection.select(index, index)

        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    def select_wrong(self):
        selection = QItemSelection()
        n = self.tablemodel.rowCount()

        for i in range(2, n):
            for j in range(i + 1, n):
                index = self.tablemodel.index(i, j)
                selection.select(index, index)
                index = self.tablemodel.index(j, i)
                selection.select(index, index)

        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    def select_none(self):
        self.tableview.selectionModel().clear()

    def cell_clicked(self, model_index):
        i, j = model_index.row(), model_index.column()
        if not i or not j:
            return
        n = self.tablemodel.rowCount()
        index = self.tablemodel.index
        selection = None
        if i == j == 1 or i == j == n - 1:
            selection = QItemSelection(index(2, 2), index(n - 1, n - 1))
        elif i in (1, n - 1):
            selection = QItemSelection(index(2, j), index(n - 1, j))
        elif j in (1, n - 1):
            selection = QItemSelection(index(i, 2), index(i, n - 1))

        if selection is not None:
            self.tableview.selectionModel().select(
                selection, QItemSelectionModel.ClearAndSelect
            )

    def commit(self):
        if self.results is not None and self.data is not None \
                and self.selected_learner:
            indices = self.tableview.selectedIndexes()
            indices = {(ind.row() - 2, ind.column() - 2) for ind in indices}
            actual = self.results.actual
            selected_learner = self.selected_learner[0]
            learner_name = self.learners[selected_learner]
            predicted = self.results.predicted[selected_learner]
            selected = [i for i, t in enumerate(zip(actual, predicted))
                        if t in indices]
            row_indices = self.results.row_indices[selected]

            extra = []
            class_var = self.data.domain.class_var
            metas = self.data.domain.metas

            if self.append_predictions:
                predicted = numpy.array(predicted[selected], dtype=object)
                extra.append(predicted.reshape(-1, 1))
                var = Orange.data.DiscreteVariable(
                    "{}({})".format(class_var.name, learner_name),
                    class_var.values
                )
                metas = metas + (var,)

            if self.append_probabilities and \
                    self.results.probabilities is not None:
                probs = self.results.probabilities[selected_learner, selected]
                extra.append(numpy.array(probs, dtype=object))
                pvars = [Orange.data.ContinuousVariable("p({})".format(value))
                         for value in class_var.values]
                metas = metas + tuple(pvars)

            X = self.data.X[row_indices]
            Y = self.data.Y[row_indices]
            M = self.data.metas[row_indices]
            row_ids = self.data.ids[row_indices]

            M = numpy.hstack((M,) + tuple(extra))
            domain = Orange.data.Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                metas
            )
            data = Orange.data.Table.from_numpy(domain, X, Y, M)
            data.ids = row_ids
            data.name = learner_name

        else:
            data = None

        self.send("Selected Data", data)

    def _invalidate(self):
        self.commit()

    def _learner_changed(self):
        # The selected learner has changed
        indices = self.tableview.selectedIndexes()
        self._update()
        selection = QItemSelection()
        for sel in indices:
            selection.select(sel, sel)
        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )
        self.commit()

    def _update(self):
        def isinvalid(x):
            return isnan(x) or isinf(x)

        # Update the displayed confusion matrix
        if self.results is not None and self.selected_learner:
            index = self.selected_learner[0]
            cmatrix = confusion_matrix(self.results, index)
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            total = rowsum.sum()
            n = len(cmatrix)
            diag = numpy.diag_indices(n)

            colors = cmatrix.astype(numpy.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(numpy.int)
                formatstr = "{}"
                div = numpy.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, numpy.newaxis]
                    div = colors.max(axis=1)[:, numpy.newaxis]
                formatstr = "{:2.1f} %"
            div[div == 0] = 1
            colors /= div
            colors[diag] = normalized[diag] / normalized[diag].max()

            hue = [0, 240]
            for i in range(n):
                for j in range(n):
                    val = normalized[i, j]
                    col_val = colors[i, j]
                    item = self._item(i + 2, j + 2)
                    item.setData(
                        "NA" if isinvalid(val) else formatstr.format(val),
                        Qt.DisplayRole)
                    bkcolor = QColor.fromHsl(
                        hue[i == j], 160,
                        255 if isinvalid(col_val) else int(255 - 30 * col_val))
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                            self.headers[i], self.headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)

            model = self.tablemodel
            font = model.invisibleRootItem().font()
            bold_font = QFont(font)
            bold_font.setBold(True)

            def sum_item(value, border=""):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                item.setData(border, BorderRole)
                item.setData(QColor(192, 192, 192), BorderColorRole)
                return item

            N = len(colsum)
            for i in range(N):
                model.setItem(N + 2, i + 2, sum_item(int(colsum[i]), "t"))
                model.setItem(i + 2, N + 2, sum_item(int(rowsum[i]), "l"))

            model.setItem(N + 2, N + 2, sum_item(int(total)))

    def send_report(self):
        if self.results is not None and self.selected_learner:
            index = self.selected_learner[0]
            self.report_table(
                "Confusion matrix for {} (showing {})".
                format(self.learners[index],
                       self.quantities[self.selected_quantity].lower()),
                self.tablemodel,2,2)

if __name__ == "__main__":
    from PyQt4.QtGui import QApplication

    app = QApplication([])
    w = OWConfusionMatrix()
    w.show()
    data = Orange.data.Table("iris")
    res = Orange.evaluation.CrossValidation(
        data, [Orange.classification.TreeLearner(),
               Orange.classification.MajorityLearner()],
        store_data=True)
    w.set_results(res)
    app.exec_()
