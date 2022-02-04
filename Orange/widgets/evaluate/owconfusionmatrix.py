"""Confusion matrix widget"""

from math import isnan, isinf
from itertools import chain
import unicodedata

from AnyQt.QtWidgets import QTableView, QHeaderView, QStyledItemDelegate, \
    QSizePolicy
from AnyQt.QtGui import QFont, QBrush, QColor, QStandardItemModel, QStandardItem
from AnyQt.QtCore import Qt, QSize, QItemSelectionModel, QItemSelection
import numpy as np
import sklearn.metrics as skl_metrics

import Orange
from Orange.data.util import get_unique_names
import Orange.evaluation
from Orange.widgets import widget, gui
from Orange.widgets.settings import \
    Setting, ContextSetting, ClassValuesContextHandler
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output


def confusion_matrix(res, index):
    """
    Compute confusion matrix

    Args:
        res (Orange.evaluation.Results): evaluation results
        index (int): model index

    Returns: Confusion matrix
    """
    labels = np.arange(len(res.domain.class_var.values))
    if not res.actual.size:
        # scikit-learn will not return an zero matrix
        return np.zeros((len(labels), len(labels)))
    else:
        return skl_metrics.confusion_matrix(
            res.actual, res.predicted[index], labels=labels)


BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)


class BorderedItemDelegate(QStyledItemDelegate):
    """Item delegate that paints border at the specified sides

    Data for `BorderRole` is a string containing letters t, r, b and/or l,
    which defines the sides at which the border is drawn.

    Role `BorderColorRole` sets the color for the cell. If not color is given,
    `self.color` is used as default.

    Args:
        color (QColor): default color (default default is black)
    """
    def __init__(self, color=Qt.black):
        super().__init__()
        self.color = color

    def paint(self, painter, option, index):
        """Overloads `paint` to draw borders"""
        QStyledItemDelegate.paint(self, painter, option, index)
        borders = index.data(BorderRole)
        if borders:
            color = index.data(BorderColorRole) or self.color
            painter.save()
            painter.setPen(color)
            rect = option.rect
            for side, p1, p2 in (("t", rect.topLeft(), rect.topRight()),
                                 ("r", rect.topRight(), rect.bottomRight()),
                                 ("b", rect.bottomLeft(), rect.bottomRight()),
                                 ("l", rect.topLeft(), rect.bottomLeft())):
                if side in borders:
                    painter.drawLine(p1, p2)
            painter.restore()


class OWConfusionMatrix(widget.OWWidget):
    """Confusion matrix widget"""

    name = "Confusion Matrix"
    description = "Display a confusion matrix constructed from " \
                  "the results of classifier evaluations."
    icon = "icons/ConfusionMatrix.svg"
    priority = 1001
    keywords = []

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    quantities = ["Number of instances",
                  "Proportion of predicted",
                  "Proportion of actual"]

    settings_version = 1
    settingsHandler = ClassValuesContextHandler()

    selected_learner = Setting([0], schema_only=True)
    selection = ContextSetting(set())
    selected_quantity = Setting(0)
    append_predictions = Setting(True)
    append_probabilities = Setting(False)
    autocommit = Setting(True)

    UserAdviceMessages = [
        widget.Message(
            "Clicking on cells or in headers outputs the corresponding "
            "data instances",
            "click_cell")]

    class Error(widget.OWWidget.Error):
        no_regression = Msg("Confusion Matrix cannot show regression results.")
        invalid_values = Msg("Evaluation Results input contains invalid values")
        empty_input = widget.Msg("Empty result on input. Nothing to display.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.results = None
        self.learners = []
        self.headers = []

        self.learners_box = gui.listBox(
            self.controlArea, self, "selected_learner", "learners", box='Learners',
            callback=self._learner_changed
        )

        self.outputbox = gui.vBox(self.buttonsArea)
        box = gui.hBox(self.outputbox)
        gui.checkBox(box, self, "append_predictions",
                     "Predictions", callback=self._invalidate)
        gui.checkBox(box, self, "append_probabilities",
                     "Probabilities",
                     callback=self._invalidate)

        gui.auto_apply(self.outputbox, self, "autocommit", box=False)

        box = gui.vBox(self.mainArea, box=True)

        sbox = gui.hBox(box)
        gui.rubber(sbox)
        gui.comboBox(sbox, self, "selected_quantity",
                     items=self.quantities, label="Show: ",
                     orientation=Qt.Horizontal, callback=self._update)

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
        view.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        view.clicked.connect(self.cell_clicked)
        box.layout().addWidget(view)

        selbox = gui.hBox(box)
        gui.button(selbox, self, "Select Correct",
                   callback=self.select_correct, autoDefault=False)
        gui.button(selbox, self, "Select Misclassified",
                   callback=self.select_wrong, autoDefault=False)
        gui.button(selbox, self, "Clear Selection",
                   callback=self.select_none, autoDefault=False)

    @staticmethod
    def sizeHint():
        """Initial size"""
        return QSize(750, 340)

    def _item(self, i, j):
        return self.tablemodel.item(i, j) or QStandardItem()

    def _set_item(self, i, j, item):
        self.tablemodel.setItem(i, j, item)

    def _init_table(self, nclasses):
        item = self._item(0, 2)
        item.setData("Predicted", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.NoItemFlags)

        self._set_item(0, 2, item)
        item = self._item(2, 0)
        item.setData("Actual", Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        item.setFlags(Qt.NoItemFlags)
        self.tableview.setItemDelegateForColumn(0, gui.VerticalItemDelegate())
        self._set_item(2, 0, item)
        self.tableview.setSpan(0, 2, 1, nclasses)
        self.tableview.setSpan(2, 0, nclasses, 1)

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
            hor_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        else:
            hor_header.setDefaultSectionSize(60)
        self.tablemodel.setRowCount(nclasses + 3)
        self.tablemodel.setColumnCount(nclasses + 3)

    @Inputs.evaluation_results
    def set_results(self, results):
        """Set the input results."""
        # false positive, pylint: disable=no-member
        prev_sel_learner = self.selected_learner.copy()
        self.clear()
        self.warning()
        self.closeContext()

        data = None
        if results is not None and results.data is not None:
            data = results.data[results.row_indices]

        self.Error.no_regression.clear()
        self.Error.empty_input.clear()
        if data is not None and not data.domain.has_discrete_class:
            self.Error.no_regression()
            data = results = None
        elif results is not None and not results.actual.size:
            self.Error.empty_input()
            data = results = None

        nan_values = False
        if results is not None:
            assert isinstance(results, Orange.evaluation.Results)
            if np.any(np.isnan(results.actual)) or \
                    np.any(np.isnan(results.predicted)):
                # Error out here (could filter them out with a warning
                # instead).
                nan_values = True
                results = data = None

        self.Error.invalid_values(shown=nan_values)

        self.results = results
        self.data = data

        if data is not None:
            class_values = data.domain.class_var.values
        elif results is not None:
            raise NotImplementedError

        if results is None:
            self.report_button.setDisabled(True)
            return

        self.report_button.setDisabled(False)

        nmodels = results.predicted.shape[0]
        self.headers = class_values + \
                       (unicodedata.lookup("N-ARY SUMMATION"), )

        # NOTE: The 'learner_names' is set in 'Test Learners' widget.
        self.learners = getattr(
            results, "learner_names",
            [f"Learner #{i + 1}" for i in range(nmodels)])

        self._init_table(len(class_values))
        self.openContext(data.domain.class_var)
        if not prev_sel_learner or prev_sel_learner[0] >= len(self.learners):
            if self.learners:
                self.selected_learner[:] = [0]
        else:
            self.selected_learner[:] = prev_sel_learner
        self._update()
        self._set_selection()
        self.commit.now()

    def clear(self):
        """Reset the widget, clear controls"""
        self.results = None
        self.data = None
        self.tablemodel.clear()
        self.headers = []
        # Clear learners last. This action will invoke `_learner_changed`
        self.learners = []

    def select_correct(self):
        """Select the diagonal elements of the matrix"""
        selection = QItemSelection()
        n = self.tablemodel.rowCount()
        for i in range(2, n):
            index = self.tablemodel.index(i, i)
            selection.select(index, index)
        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def select_wrong(self):
        """Select the off-diagonal elements of the matrix"""
        selection = QItemSelection()
        n = self.tablemodel.rowCount()
        for i in range(2, n):
            for j in range(i + 1, n):
                index = self.tablemodel.index(i, j)
                selection.select(index, index)
                index = self.tablemodel.index(j, i)
                selection.select(index, index)
        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def select_none(self):
        """Reset selection"""
        self.tableview.selectionModel().clear()

    def cell_clicked(self, model_index):
        """Handle cell click event"""
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
                selection, QItemSelectionModel.ClearAndSelect)

    def _prepare_data(self):
        indices = self.tableview.selectedIndexes()
        indices = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        actual = self.results.actual
        learner_name = self.learners[self.selected_learner[0]]
        predicted = self.results.predicted[self.selected_learner[0]]
        selected = [i for i, t in enumerate(zip(actual, predicted))
                    if t in indices]

        extra = []
        class_var = self.data.domain.class_var
        metas = self.data.domain.metas
        attrs = self.data.domain.attributes
        names = [var.name for var in chain(metas, [class_var], attrs)]

        if self.append_predictions:
            extra.append(predicted.reshape(-1, 1))
            proposed = "{}({})".format(class_var.name, learner_name)
            name = get_unique_names(names, proposed)
            var = Orange.data.DiscreteVariable(
                                               name,
                                               class_var.values)
            metas = metas + (var,)

        if self.append_probabilities and \
                        self.results.probabilities is not None:
            probs = self.results.probabilities[self.selected_learner[0]]
            extra.append(np.array(probs, dtype=object))
            pvars = [Orange.data.ContinuousVariable("p({})".format(value))
                     for value in class_var.values]
            metas = metas + tuple(pvars)

        domain = Orange.data.Domain(self.data.domain.attributes,
                                    self.data.domain.class_vars,
                                    metas)
        data = self.data.transform(domain)
        if extra:
            with data.unlocked(data.metas):
                data.metas[:, len(self.data.domain.metas):] = \
                    np.hstack(tuple(extra))
        data.name = learner_name

        if selected:
            annotated_data = create_annotated_table(data, selected)
            data = data[selected]
        else:
            annotated_data = create_annotated_table(data, [])
            data = None

        return data, annotated_data

    @gui.deferred
    def commit(self):
        """Output data instances corresponding to selected cells"""
        if self.results is not None and self.data is not None \
                and self.selected_learner:
            data, annotated_data = self._prepare_data()
        else:
            data = None
            annotated_data = None

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(annotated_data)

    def _invalidate(self):
        indices = self.tableview.selectedIndexes()
        self.selection = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        self.commit.deferred()

    def _set_selection(self):
        selection = QItemSelection()
        index = self.tableview.model().index
        for row, col in self.selection:
            sel = index(row + 2, col + 2)
            selection.select(sel, sel)
        self.tableview.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def _learner_changed(self):
        self._update()
        self._set_selection()
        self.commit.deferred()

    def _update(self):
        def _isinvalid(x):
            return isnan(x) or isinf(x)

        # Update the displayed confusion matrix
        if self.results is not None and self.selected_learner:
            cmatrix = confusion_matrix(self.results, self.selected_learner[0])
            colsum = cmatrix.sum(axis=0)
            rowsum = cmatrix.sum(axis=1)
            n = len(cmatrix)
            diag = np.diag_indices(n)

            colors = cmatrix.astype(np.double)
            colors[diag] = 0
            if self.selected_quantity == 0:
                normalized = cmatrix.astype(int)
                formatstr = "{}"
                div = np.array([colors.max()])
            else:
                if self.selected_quantity == 1:
                    normalized = 100 * cmatrix / colsum
                    div = colors.max(axis=0)
                else:
                    normalized = 100 * cmatrix / rowsum[:, np.newaxis]
                    div = colors.max(axis=1)[:, np.newaxis]
                formatstr = "{:2.1f} %"
            div[div == 0] = 1
            colors /= div
            maxval = normalized[diag].max()
            if maxval > 0:
                colors[diag] = normalized[diag] / maxval

            for i in range(n):
                for j in range(n):
                    val = normalized[i, j]
                    col_val = colors[i, j]
                    item = self._item(i + 2, j + 2)
                    item.setData(
                        "NA" if _isinvalid(val) else formatstr.format(val),
                        Qt.DisplayRole)
                    bkcolor = QColor.fromHsl(
                        [0, 240][i == j], 160,
                        255 if _isinvalid(col_val) else int(255 - 30 * col_val))
                    item.setData(QBrush(bkcolor), Qt.BackgroundRole)
                    # bkcolor is light-ish so use a black text
                    item.setData(QBrush(Qt.black), Qt.ForegroundRole)
                    item.setData("trbl", BorderRole)
                    item.setToolTip("actual: {}\npredicted: {}".format(
                        self.headers[i], self.headers[j]))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self._set_item(i + 2, j + 2, item)

            bold_font = self.tablemodel.invisibleRootItem().font()
            bold_font.setBold(True)

            def _sum_item(value, border=""):
                item = QStandardItem()
                item.setData(value, Qt.DisplayRole)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setFlags(Qt.ItemIsEnabled)
                item.setFont(bold_font)
                item.setData(border, BorderRole)
                item.setData(QColor(192, 192, 192), BorderColorRole)
                return item

            for i in range(n):
                self._set_item(n + 2, i + 2, _sum_item(int(colsum[i]), "t"))
                self._set_item(i + 2, n + 2, _sum_item(int(rowsum[i]), "l"))
            self._set_item(n + 2, n + 2, _sum_item(int(rowsum.sum())))

    def send_report(self):
        """Send report"""
        if self.results is not None and self.selected_learner:
            self.report_table(
                "Confusion matrix for {} (showing {})".
                format(self.learners[self.selected_learner[0]],
                       self.quantities[self.selected_quantity].lower()),
                self.tableview)

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version:
            # For some period of time the 'selected_learner' property was
            # changed from List[int] -> int
            # (commit 4e49bb3fd0e11262f3ebf4b1116a91a4b49cc982) and then back
            # again (commit 8a492d79a2e17154a0881e24a05843406c8892c0)
            if "selected_learner" in settings and \
                    isinstance(settings["selected_learner"], int):
                settings["selected_learner"] = [settings["selected_learner"]]


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.evaluate.utils import results_for_preview
    WidgetPreview(OWConfusionMatrix).run(results_for_preview("iris"))
