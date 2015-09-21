import unicodedata

from PyQt4.QtGui import (
    QGridLayout, QLabel, QTableView, QStandardItemModel, QStandardItem,
    QItemSelectionModel, QItemSelection, QFont, QHeaderView
)
from PyQt4.QtCore import Qt

import numpy
import sklearn.metrics as skl_metrics

import Orange
from Orange.widgets import widget, settings, gui


def confusion_matrix(res, index):
    return skl_metrics.confusion_matrix(
        res.actual, res.predicted[index])


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

    def __init__(self, parent=None):
        super().__init__(parent)

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
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Predicted"), 0, 1, Qt.AlignCenter)
        grid.addWidget(VerticalLabel("Actual Class"), 1, 0, Qt.AlignCenter)

        self.tablemodel = QStandardItemModel(self)
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setModel(self.tablemodel)
        view.horizontalHeader().setMinimumSectionSize(60)
        view.selectionModel().selectionChanged.connect(self._invalidate)
        grid.addWidget(view, 1, 1)
        self.mainArea.layout().addLayout(grid)

    def sizeHint(self):
        return QSize(750, 600)

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

        if results is not None:
            nmodels, ntests = results.predicted.shape
            self.headers = class_values + [unicodedata.lookup("N-ARY SUMMATION")]

            # NOTE: The 'learner_names' is set in 'Test Learners' widget.
            if hasattr(results, "learner_names"):
                self.learners = results.learner_names
            else:
                self.learners = ["L %i" % (i + 1) for i in range(nmodels)]

            self.tablemodel.setVerticalHeaderLabels(self.headers)
            self.tablemodel.setHorizontalHeaderLabels(self.headers)
            if len(' '.join(self.headers)) < 120:
                self.tableview.horizontalHeader().setResizeMode(QHeaderView.ResizeToContents)
            else:
                self.tableview.horizontalHeader().setDefaultSectionSize(60)
                self.tableview.horizontalHeader().setResizeMode(QHeaderView.Interactive)
                for i, h in enumerate(self.headers):
                    self.tablemodel.horizontalHeaderItem(i).setToolTip(h)
            self.tablemodel.setRowCount(len(class_values) + 1)
            self.tablemodel.setColumnCount(len(class_values) + 1)
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
        for i in range(n):
            index = self.tablemodel.index(i, i)
            selection.select(index, index)

        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    def select_wrong(self):
        selection = QItemSelection()
        n = self.tablemodel.rowCount()

        for i in range(n):
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

    def commit(self):
        if self.results is not None and self.data is not None \
                and self.selected_learner:
            indices = self.tableview.selectedIndexes()
            indices = {(ind.row(), ind.column()) for ind in indices}
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
        self._update()
        self._invalidate()

    def _update(self):
        # Update the displayed confusion matrix
        if self.results is not None and self.selected_learner:
            index = self.selected_learner[0]
            cmatrix = confusion_matrix(self.results, index)
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            total = rowsum.sum()

            if self.selected_quantity == 0:
                value = lambda i, j: int(cmatrix[i, j])
            elif self.selected_quantity == 1:
                value = lambda i, j: \
                    ("{:2.1f} %".format(100 * cmatrix[i, j] / colsum[i])
                     if colsum[i] else "N/A")
            elif self.selected_quantity == 2:
                value = lambda i, j: \
                    ("{:2.1f} %".format(100 * cmatrix[i, j] / rowsum[i])
                     if colsum[i] else "N/A")
            else:
                assert False

            model = self.tablemodel
            for i, row in enumerate(cmatrix):
                for j, _ in enumerate(row):
                    item = model.item(i, j)
                    if item is None:
                        item = QStandardItem()
                    item.setData(value(i, j), Qt.DisplayRole)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                        self.headers[i], self.headers[j]))
                    model.setItem(i, j, item)

            font = model.invisibleRootItem().font()
            bold_font = QFont(font)
            bold_font.setBold(True)

            def sum_item(value):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                return item

            N = len(colsum)
            for i in range(N):
                model.setItem(N, i, sum_item(int(colsum[i])))
                model.setItem(i, N, sum_item(int(rowsum[i])))

            model.setItem(N, N, sum_item(int(total)))

from PyQt4.QtGui import QSizePolicy, QFontMetrics, QPainter
from PyQt4.QtCore import QRect, QPoint, QSize


class VerticalLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setMaximumWidth(self.sizeHint().width() + 2)
        self.setMargin(4)

    def sizeHint(self):
        metrics = QFontMetrics(self.font())
        rect = metrics.boundingRect(self.text())
        size = QSize(rect.height() + self.margin(),
                     rect.width() + self.margin())
        return size

    def setGeometry(self, rect):
        super().setGeometry(rect)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.geometry()
        text_rect = QRect(0, 0, rect.width(), rect.height())

        painter.translate(text_rect.bottomLeft())
        painter.rotate(-90)
        painter.drawText(QRect(QPoint(0, 0),
                               QSize(rect.height(), rect.width())),
                         Qt.AlignCenter, self.text())
        painter.end()


if __name__ == "__main__":
    from PyQt4.QtGui import QApplication

    app = QApplication([])
    w = OWConfusionMatrix()
    w.show()
    data = Orange.data.Table("iris")
    res = Orange.evaluation.CrossValidation(
        data, [Orange.classification.ClassificationTreeLearner()],
        store_data=True)
    w.set_results(res)
    app.exec_()
