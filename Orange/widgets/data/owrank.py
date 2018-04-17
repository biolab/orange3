"""
Rank
====

Rank (score) features for prediction.

"""

from collections import namedtuple, OrderedDict
import logging
from functools import partial
from itertools import chain

import numpy as np
from scipy.sparse import issparse

from AnyQt.QtGui import QFontMetrics
from AnyQt.QtWidgets import (
    QTableView, QRadioButton, QButtonGroup, QGridLayout,
    QStackedWidget, QHeaderView, QCheckBox, QItemDelegate,
)
from AnyQt.QtCore import (
    Qt, QItemSelection, QItemSelectionRange, QItemSelectionModel,
)

from Orange.data import (Table, Domain, ContinuousVariable, DiscreteVariable,
                         StringVariable)
from Orange.misc.cache import memoize_method
from Orange.preprocess import score
from Orange.widgets import report
from Orange.widgets import gui
from Orange.widgets.settings import (DomainContextHandler, Setting,
                                     ContextSetting)
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg, Input, Output


log = logging.getLogger(__name__)


class ProblemType:
    CLASSIFICATION, REGRESSION, UNSUPERVISED = range(3)

    @classmethod
    def from_variable(cls, variable):
        return (cls.CLASSIFICATION if isinstance(variable, DiscreteVariable) else
                cls.REGRESSION if isinstance(variable, ContinuousVariable) else
                cls.UNSUPERVISED)

ScoreMeta = namedtuple("score_meta", ["name", "shortname", "scorer", 'problem_type', 'is_default'])

# Default scores.
CLS_SCORES = [
    ScoreMeta("Information Gain", "Info. gain",
              score.InfoGain, ProblemType.CLASSIFICATION, False),
    ScoreMeta("Information Gain Ratio", "Gain ratio",
              score.GainRatio, ProblemType.CLASSIFICATION, True),
    ScoreMeta("Gini Decrease", "Gini",
              score.Gini, ProblemType.CLASSIFICATION, True),
    ScoreMeta("ANOVA", "ANOVA",
              score.ANOVA, ProblemType.CLASSIFICATION, False),
    ScoreMeta("χ²", "χ²",
              score.Chi2, ProblemType.CLASSIFICATION, False),
    ScoreMeta("ReliefF", "ReliefF",
              score.ReliefF, ProblemType.CLASSIFICATION, False),
    ScoreMeta("FCBF", "FCBF",
              score.FCBF, ProblemType.CLASSIFICATION, False)
]
REG_SCORES = [
    ScoreMeta("Univariate Regression", "Univar. reg.",
              score.UnivariateLinearRegression, ProblemType.REGRESSION, True),
    ScoreMeta("RReliefF", "RReliefF",
              score.RReliefF, ProblemType.REGRESSION, True)
]
SCORES = CLS_SCORES + REG_SCORES


class TableView(QTableView):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent,
                         selectionBehavior=QTableView.SelectRows,
                         selectionMode=QTableView.ExtendedSelection,
                         sortingEnabled=True,
                         showGrid=True,
                         cornerButtonEnabled=False,
                         alternatingRowColors=False,
                         **kwargs)
        self.setItemDelegate(gui.ColoredBarItemDelegate(self))
        self.setItemDelegateForColumn(0, QItemDelegate())

        header = self.verticalHeader()
        header.setSectionResizeMode(header.Fixed)
        header.setFixedWidth(50)
        header.setDefaultSectionSize(22)
        header.setTextElideMode(Qt.ElideMiddle)  # Note: https://bugreports.qt.io/browse/QTBUG-62091

        header = self.horizontalHeader()
        header.setSectionResizeMode(header.Fixed)
        header.setFixedHeight(24)
        header.setDefaultSectionSize(80)
        header.setTextElideMode(Qt.ElideMiddle)

    def setVHeaderFixedWidthFromLabel(self, max_label):
        header = self.verticalHeader()
        width = QFontMetrics(header.font()).width(max_label)
        header.setFixedWidth(min(width + 40, 400))


class TableModel(PyTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extremes = {}

    def data(self, index, role=Qt.DisplayRole, _isnan=np.isnan):
        if role == gui.BarRatioRole and index.isValid():
            value = super().data(index, Qt.EditRole)
            if not isinstance(value, float):
                return None
            vmin, vmax = self._extremes.get(index.column(), (-np.inf, np.inf))
            value = (value - vmin) / ((vmax - vmin) or 1)
            return value

        if role == Qt.DisplayRole:
            role = Qt.EditRole

        value = super().data(index, role)

        # Display nothing for non-existent attr value counts in the first column
        if role == Qt.EditRole and index.column() == 0 and _isnan(value):
            return ''

        return value

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.InitialSortOrderRole:
            return Qt.DescendingOrder
        return super().headerData(section, orientation, role)

    def setExtremesFrom(self, column, values):
        """Set extremes for columnn's ratio bars from values"""
        try:
            vmin = np.nanmin(values)
            if np.isnan(vmin):
                raise TypeError
        except TypeError:
            vmin, vmax = -np.inf, np.inf
        else:
            vmax = np.nanmax(values)
        self._extremes[column] = (vmin, vmax)

    def resetSorting(self, yes_reset=False):
        """We don't want to invalidate our sort proxy model everytime we
        wrap a new list. Our proxymodel only invalidates explicitly
        (i.e. when new data is set)"""
        if yes_reset:
            super().resetSorting()

    def _argsortData(self, data, order):
        """Always sort NaNs last"""
        indices = np.argsort(data, kind='mergesort')
        if order == Qt.DescendingOrder:
            return np.roll(indices[::-1], -np.isnan(data).sum())
        return indices


class OWRank(OWWidget):
    name = "Rank"
    description = "Rank and filter data features by their relevance."
    icon = "icons/Rank.svg"
    priority = 1102

    buttons_area_orientation = Qt.Vertical

    class Inputs:
        data = Input("Data", Table)
        scorer = Input("Scorer", score.Scorer, multiple=True)

    class Outputs:
        reduced_data = Output("Reduced Data", Table, default=True)
        scores = Output("Scores", Table)

    SelectNone, SelectAll, SelectManual, SelectNBest = range(4)

    nSelected = Setting(5)
    auto_apply = Setting(True)

    sorting = Setting((0, Qt.DescendingOrder))
    selected_methods = Setting(set())

    settings_version = 2
    settingsHandler = DomainContextHandler()
    selected_rows = ContextSetting([])
    selectionMethod = ContextSetting(SelectNBest)

    class Information(OWWidget.Information):
        no_target_var = Msg("Data does not have a single target variable. "
                            "You can still connect in unsupervised scorers "
                            "such as PCA.")
        missings_imputed = Msg('Missing values will be imputed as needed.')

    class Error(OWWidget.Error):
        invalid_type = Msg("Cannot handle target variable type {}")
        inadequate_learner = Msg("Scorer {} inadequate: {}")
        no_attributes = Msg("Data does not have a single attribute.")

    def __init__(self):
        super().__init__()
        self.scorers = OrderedDict()
        self.out_domain_desc = None
        self.data = None
        self.problem_type_mode = ProblemType.CLASSIFICATION

        if not self.selected_methods:
            self.selected_methods = {method.name for method in SCORES
                                     if method.is_default}

        # GUI

        self.ranksModel = model = TableModel(parent=self)  # type: TableModel
        self.ranksView = view = TableView(self)            # type: TableView
        self.mainArea.layout().addWidget(view)
        view.setModel(model)
        view.setColumnWidth(0, 30)
        view.selectionModel().selectionChanged.connect(self.on_select)

        def _set_select_manual():
            self.setSelectionMethod(OWRank.SelectManual)

        view.pressed.connect(_set_select_manual)
        view.verticalHeader().sectionClicked.connect(_set_select_manual)
        view.horizontalHeader().sectionClicked.connect(self.headerClick)

        self.measuresStack = stacked = QStackedWidget(self)
        self.controlArea.layout().addWidget(stacked)

        for scoring_methods in (CLS_SCORES,
                                REG_SCORES,
                                []):
            box = gui.vBox(None, "Scoring Methods" if scoring_methods else None)
            stacked.addWidget(box)
            for method in scoring_methods:
                box.layout().addWidget(QCheckBox(
                    method.name, self,
                    objectName=method.shortname,  # To be easily found in tests
                    checked=method.name in self.selected_methods,
                    stateChanged=partial(self.methodSelectionChanged, method_name=method.name)))
            gui.rubber(box)

        gui.rubber(self.controlArea)
        self.switchProblemType(ProblemType.CLASSIFICATION)

        selMethBox = gui.vBox(self.controlArea, "Select Attributes", addSpace=True)

        grid = QGridLayout()
        grid.setContentsMargins(6, 0, 6, 0)
        self.selectButtons = QButtonGroup()
        self.selectButtons.buttonClicked[int].connect(self.setSelectionMethod)

        def button(text, buttonid, toolTip=None):
            b = QRadioButton(text)
            self.selectButtons.addButton(b, buttonid)
            if toolTip is not None:
                b.setToolTip(toolTip)
            return b

        b1 = button(self.tr("None"), OWRank.SelectNone)
        b2 = button(self.tr("All"), OWRank.SelectAll)
        b3 = button(self.tr("Manual"), OWRank.SelectManual)
        b4 = button(self.tr("Best ranked:"), OWRank.SelectNBest)

        s = gui.spin(selMethBox, self, "nSelected", 1, 100,
                     callback=lambda: self.setSelectionMethod(OWRank.SelectNBest))

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(b4, 3, 0)
        grid.addWidget(s, 3, 1)

        self.selectButtons.button(self.selectionMethod).setChecked(True)

        selMethBox.layout().addLayout(grid)

        gui.auto_commit(selMethBox, self, "auto_apply", "Send", box=False)

        self.resize(690, 500)

    def switchProblemType(self, index):
        """
        Switch between discrete/continuous/no_class mode
        """
        self.measuresStack.setCurrentIndex(index)
        self.problem_type_mode = index

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.selected_rows = []
        self.ranksModel.clear()
        self.ranksModel.resetSorting(True)

        self.get_method_scores.cache_clear()
        self.get_scorer_scores.cache_clear()

        self.Error.clear()
        self.Information.clear()
        self.Information.missings_imputed(
            shown=data is not None and data.has_missing())

        if data is not None and not len(data.domain.attributes):
            data = None
            self.Error.no_attributes()
        self.data = data
        self.switchProblemType(ProblemType.CLASSIFICATION)
        if self.data is not None:
            domain = self.data.domain

            if domain.has_discrete_class:
                problem_type = ProblemType.CLASSIFICATION
            elif domain.has_continuous_class:
                problem_type = ProblemType.REGRESSION
            elif not domain.class_var:
                self.Information.no_target_var()
                problem_type = ProblemType.UNSUPERVISED
            else:
                # This can happen?
                self.Error.invalid_type(type(domain.class_var).__name__)
                problem_type = None

            if problem_type is not None:
                self.switchProblemType(problem_type)

            self.ranksModel.setVerticalHeaderLabels(domain.attributes)
            self.ranksView.setVHeaderFixedWidthFromLabel(
                max((a.name for a in domain.attributes), key=len))

            self.selectionMethod = OWRank.SelectNBest

        self.openContext(data)

    def handleNewSignals(self):
        self.setStatusMessage('Running')
        self.updateScores()
        self.setStatusMessage('')
        self.on_select()

    @Inputs.scorer
    def set_learner(self, scorer, id):
        if scorer is None:
            self.scorers.pop(id, None)
        else:
            # Avoid caching a (possibly stale) previous instance of the same
            # Scorer passed via the same signal
            if id in self.scorers:
                self.get_scorer_scores.cache_clear()

            self.scorers[id] = ScoreMeta(scorer.name, scorer.name, scorer,
                                         ProblemType.from_variable(scorer.class_type),
                                         False)

    @memoize_method()
    def get_method_scores(self, method):
        estimator = method.scorer()
        data = self.data
        try:
            scores = np.asarray(estimator(data))
        except ValueError:
            log.warning("Scorer %s wasn't able to compute all scores at once",
                        method.name)
            try:
                scores = np.array([estimator(data, attr)
                                   for attr in data.domain.attributes])
            except ValueError:
                log.error(
                    "Scorer %s wasn't able to compute scores at all",
                    method.name)
                scores = np.full(len(data.domain.attributes), np.nan)
        return scores

    @memoize_method()
    def get_scorer_scores(self, scorer):
        try:
            scores = scorer.scorer.score_data(self.data).T
        except ValueError:
            log.error(
                "Scorer %s wasn't able to compute scores at all",
                scorer.name)
            scores = np.full((len(self.data.domain.attributes), 1), np.nan)

        labels = ((scorer.shortname,)
                  if scores.shape[1] == 1 else
                  tuple(scorer.shortname + '_' + str(i)
                        for i in range(1, 1 + scores.shape[1])))
        return scores, labels

    def updateScores(self):
        if self.data is None:
            self.ranksModel.clear()
            self.Outputs.scores.send(None)
            return

        methods = [method
                   for method in SCORES
                   if (method.name in self.selected_methods and
                       method.problem_type == self.problem_type_mode and
                       (not issparse(self.data.X) or
                        method.scorer.supports_sparse_data))]

        scorers = []
        self.Error.inadequate_learner.clear()
        for scorer in self.scorers.values():
            if scorer.problem_type in (self.problem_type_mode, ProblemType.UNSUPERVISED):
                scorers.append(scorer)
            else:
                self.Error.inadequate_learner(scorer.name, scorer.learner_adequacy_err_msg)

        method_scores = tuple(self.get_method_scores(method)
                              for method in methods)

        scorer_scores, scorer_labels = (), ()
        if scorers:
            scorer_scores, scorer_labels = zip(*(self.get_scorer_scores(scorer)
                                                 for scorer in scorers))
            scorer_labels = tuple(chain.from_iterable(scorer_labels))

        labels = tuple(method.shortname for method in methods) + scorer_labels
        model_array = np.column_stack(
            ([len(a.values) if a.is_discrete else np.nan
              for a in self.data.domain.attributes],) +
            (method_scores if method_scores else ()) +
            (scorer_scores if scorer_scores else ())
        )
        for column, values in enumerate(model_array.T):
            self.ranksModel.setExtremesFrom(column, values)

        self.ranksModel.wrap(model_array.tolist())
        self.ranksModel.setHorizontalHeaderLabels(('#',) + labels)
        self.ranksView.setColumnWidth(0, 40)

        # Re-apply sort
        try:
            sort_column, sort_order = self.sorting
            if sort_column < len(labels):
                # adds 1 for '#' (discrete count) column
                self.ranksModel.sort(sort_column + 1, sort_order)
                self.ranksView.horizontalHeader().setSortIndicator(sort_column + 1, sort_order)
        except ValueError:
            pass

        self.autoSelection()
        self.Outputs.scores.send(self.create_scores_table(labels))

    def on_select(self):
        # Save indices of attributes in the original, unsorted domain
        self.selected_rows = self.ranksModel.mapToSourceRows([
            i.row() for i in self.ranksView.selectionModel().selectedRows(0)])
        self.commit()

    def setSelectionMethod(self, method):
        if self.selectionMethod != method:
            self.selectionMethod = method
            self.selectButtons.button(method).setChecked(True)
        self.autoSelection()
        self.on_select()

    def autoSelection(self):
        selModel = self.ranksView.selectionModel()
        model = self.ranksModel
        rowCount = model.rowCount()
        columnCount = model.columnCount()

        if self.selectionMethod == OWRank.SelectNone:
            selection = QItemSelection()
        elif self.selectionMethod == OWRank.SelectAll:
            selection = QItemSelection(
                model.index(0, 0),
                model.index(rowCount - 1, columnCount - 1)
            )
        elif self.selectionMethod == OWRank.SelectNBest:
            nSelected = min(self.nSelected, rowCount)
            selection = QItemSelection(
                model.index(0, 0),
                model.index(nSelected - 1, columnCount - 1)
            )
        else:
            selection = QItemSelection()
            if len(self.selected_rows):
                for row in model.mapFromSourceRows(self.selected_rows):
                    selection.append(QItemSelectionRange(
                        model.index(row, 0), model.index(row, columnCount - 1)))

        selModel.select(selection, QItemSelectionModel.ClearAndSelect)

    def headerClick(self, index):
        if index >= 1 and self.selectionMethod == OWRank.SelectNBest:
            # Reselect the top ranked attributes
            self.autoSelection()

        # Store the header states
        sort_order = self.ranksModel.sortOrder()
        sort_column = self.ranksModel.sortColumn() - 1  # -1 for '#' (discrete count) column
        self.sorting = (sort_column, sort_order)

    def methodSelectionChanged(self, state, method_name):
        if state == Qt.Checked:
            self.selected_methods.add(method_name)
        elif method_name in self.selected_methods:
            self.selected_methods.remove(method_name)

        self.updateScores()

    def send_report(self):
        if not self.data:
            return
        self.report_domain("Input", self.data.domain)
        self.report_table("Ranks", self.ranksView, num_format="{:.3f}")
        if self.out_domain_desc is not None:
            self.report_items("Output", self.out_domain_desc)

    def commit(self):
        selected_attrs = []
        if self.data is not None:
            attributes = self.data.domain.attributes
            if len(attributes) == len(self.selected_rows):
                self.selectionMethod = OWRank.SelectAll
                self.selectButtons.button(self.selectionMethod).setChecked(True)
            selected_attrs = [attributes[i]
                              for i in self.selected_rows]
        if not selected_attrs:
            self.Outputs.reduced_data.send(None)
            self.out_domain_desc = None
        else:
            reduced_domain = Domain(
                selected_attrs, self.data.domain.class_var, self.data.domain.metas)
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)
            self.out_domain_desc = report.describe_domain(data.domain)

    def create_scores_table(self, labels):
        model_list = self.ranksModel.tolist()
        if not model_list or len(model_list[0]) == 1:  # Empty or just n_values column
            return None

        domain = Domain([ContinuousVariable(label) for label in labels],
                        metas=[StringVariable("Feature")])

        # Prevent np.inf scores
        finfo = np.finfo(np.float64)
        scores = np.clip(np.array(model_list)[:, 1:], finfo.min, finfo.max)

        feature_names = np.array([a.name for a in self.data.domain.attributes])
        # Reshape to 2d array as Table does not like 1d arrays
        feature_names = feature_names[:, None]

        new_table = Table(domain, scores, metas=feature_names)
        new_table.name = "Feature Scores"
        return new_table

    @classmethod
    def migrate_settings(cls, settings, version):
        # If older settings, restore sort header to default
        # Saved selected_rows will likely be incorrect
        if version is None or version < 2:
            column, order = 0, Qt.DescendingOrder
            headerState = settings.pop("headerState", None)

            # Lacking knowledge of last problemType, use discrete ranks view's ordering
            if isinstance(headerState, (tuple, list)):
                headerState = headerState[0]

            if isinstance(headerState, bytes):
                hview = QHeaderView(Qt.Horizontal)
                hview.restoreState(headerState)
                column, order = hview.sortIndicatorSection() - 1, hview.sortIndicatorOrder()
            settings["sorting"] = (column, order)

    @classmethod
    def migrate_context(cls, context, version):
        if version is None or version < 2:
            # Old selection was saved as sorted indices. New selection is original indices.
            # Since we can't devise the latter without first computing the ranks,
            # just reset the selection to avoid confusion.
            context.values['selected_rows'] = []


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    from Orange.classification import RandomForestLearner
    a = QApplication([])
    ow = OWRank()
    ow.set_learner(RandomForestLearner(), (3, 'Learner', None))
    ow.setData(Table("heart_disease.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()
