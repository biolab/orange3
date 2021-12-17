import logging
import warnings
from collections import namedtuple
from functools import partial
from itertools import chain
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple

import numpy as np
from AnyQt.QtCore import (
    QItemSelection, QItemSelectionModel, QItemSelectionRange, Qt,
    pyqtSignal as Signal
)
from AnyQt.QtWidgets import (
    QButtonGroup, QCheckBox, QGridLayout, QHeaderView, QItemDelegate,
    QRadioButton, QStackedWidget, QTableView
)
from orangewidget.settings import IncompatibleContext
from scipy.sparse import issparse

from Orange.data import (
    ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
)
from Orange.data.util import get_unique_names_duplicates
from Orange.preprocess import score
from Orange.widgets import gui, report
from Orange.widgets.settings import (
    ContextSetting, DomainContextHandler, Setting
)
from Orange.widgets.unsupervised.owdistances import InterruptException
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import AttributeList, Input, MultiInput, Output, Msg, OWWidget

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

VARNAME_COL, NVAL_COL = range(2)


class TableView(QTableView):
    manualSelection = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent,
                         selectionBehavior=QTableView.SelectRows,
                         selectionMode=QTableView.ExtendedSelection,
                         sortingEnabled=True,
                         showGrid=True,
                         cornerButtonEnabled=False,
                         alternatingRowColors=False,
                         **kwargs)
        # setItemDelegate(ForColumn) doesn't take ownership of delegates
        self._bar_delegate = gui.ColoredBarItemDelegate(self)
        self._del0, self._del1 = QItemDelegate(), QItemDelegate()
        self.setItemDelegate(self._bar_delegate)
        self.setItemDelegateForColumn(VARNAME_COL, self._del0)
        self.setItemDelegateForColumn(NVAL_COL, self._del1)

        header = self.horizontalHeader()
        header.setSectionResizeMode(header.Fixed)
        header.setFixedHeight(24)
        header.setDefaultSectionSize(80)
        header.setTextElideMode(Qt.ElideMiddle)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.manualSelection.emit()


class TableModel(PyTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extremes = {}

    def data(self, index, role=Qt.DisplayRole):
        if role == gui.BarRatioRole and index.isValid():
            value = super().data(index, Qt.EditRole)
            if not isinstance(value, float):
                return None
            vmin, vmax = self._extremes.get(index.column(), (-np.inf, np.inf))
            value = (value - vmin) / ((vmax - vmin) or 1)
            return value

        if role == Qt.DisplayRole and index.column() != VARNAME_COL:
            role = Qt.EditRole

        value = super().data(index, role)

        # Display nothing for non-existent attr value counts in column 1
        if role == Qt.EditRole \
                and index.column() == NVAL_COL and np.isnan(value):
            return ''

        return value

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.InitialSortOrderRole:
            return Qt.DescendingOrder if section > 0 else Qt.AscendingOrder
        return super().headerData(section, orientation, role)

    def setExtremesFrom(self, column, values):
        """Set extremes for columnn's ratio bars from values"""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
                vmin = np.nanmin(values)
            if np.isnan(vmin):
                raise TypeError
        except TypeError:
            vmin, vmax = -np.inf, np.inf
        else:
            vmax = np.nanmax(values)
        self._extremes[column] = (vmin, vmax)

    def resetSorting(self, yes_reset=False):
        # pylint: disable=arguments-differ
        """We don't want to invalidate our sort proxy model everytime we
        wrap a new list. Our proxymodel only invalidates explicitly
        (i.e. when new data is set)"""
        if yes_reset:
            super().resetSorting()

    def _argsortData(self, data, order):
        if data.dtype not in (float, int):
            data = np.char.lower(data)
        indices = np.argsort(data, kind='mergesort')
        if order == Qt.DescendingOrder:
            indices = indices[::-1]
            if data.dtype == float:
                # Always sort NaNs last
                return np.roll(indices, -np.isnan(data).sum())
        return indices


class Results(SimpleNamespace):
    method_scores: Tuple[ScoreMeta, np.ndarray] = None
    scorer_scores: Tuple[ScoreMeta, Tuple[np.ndarray, List[str]]] = None


def get_method_scores(data: Table, method: ScoreMeta) -> np.ndarray:
    estimator = method.scorer()
    # The widget handles infs and nans.
    # Any errors in scorers need to be detected elsewhere.
    with np.errstate(all="ignore"):
        try:
            scores = np.asarray(estimator(data))
        except ValueError:
            try:
                scores = np.array(
                    [estimator(data, attr) for attr in data.domain.attributes]
                )
            except ValueError:
                log.error("%s doesn't work on this data", method.name)
                scores = np.full(len(data.domain.attributes), np.nan)
            else:
                log.warning(
                    "%s had to be computed separately for each " "variable",
                    method.name,
                )
        return scores


def get_scorer_scores(
    data: Table, scorer: ScoreMeta
) -> Tuple[np.ndarray, Tuple[str]]:
    try:
        scores = scorer.scorer.score_data(data).T
    except (ValueError, TypeError):
        log.error("%s doesn't work on this data", scorer.name)
        scores = np.full((len(data.domain.attributes), 1), np.nan)

    labels = (
        (scorer.shortname,)
        if scores.shape[1] == 1
        else tuple(
            scorer.shortname + "_" + str(i)
            for i in range(1, 1 + scores.shape[1])
        )
    )
    return scores, labels


def run(
    data: Table,
    methods: List[ScoreMeta],
    scorers: List[ScoreMeta],
    state: TaskState,
) -> Results:
    progress_steps = iter(np.linspace(0, 100, len(methods) + len(scorers)))

    def call_with_cb(get_scores: Callable, method: ScoreMeta):
        scores = get_scores(data, method)
        state.set_progress_value(next(progress_steps))
        if state.is_interruption_requested():
            raise InterruptException
        return scores

    method_scores = tuple(
        (method, call_with_cb(get_method_scores, method)) for method in methods
    )
    scorer_scores = tuple(
        (scorer, call_with_cb(get_scorer_scores, scorer)) for scorer in scorers
    )
    return Results(method_scores=method_scores, scorer_scores=scorer_scores)


class OWRank(OWWidget, ConcurrentWidgetMixin):
    name = "Rank"
    description = "Rank and filter data features by their relevance."
    icon = "icons/Rank.svg"
    priority = 1102
    keywords = []

    buttons_area_orientation = Qt.Vertical

    class Inputs:
        data = Input("Data", Table)
        scorer = MultiInput("Scorer", score.Scorer, filter_none=True)

    class Outputs:
        reduced_data = Output("Reduced Data", Table, default=True)
        scores = Output("Scores", Table)
        features = Output("Features", AttributeList, dynamic=False)

    SelectNone, SelectAll, SelectManual, SelectNBest = range(4)

    nSelected = ContextSetting(5)
    auto_apply = Setting(True)

    sorting = Setting((0, Qt.DescendingOrder))
    selected_methods = Setting(set())

    settings_version = 3
    settingsHandler = DomainContextHandler()
    selected_attrs = ContextSetting([], schema_only=True)
    selectionMethod = ContextSetting(SelectNBest)

    class Information(OWWidget.Information):
        no_target_var = Msg("Data does not have a (single) target variable.")
        missings_imputed = Msg('Missing values will be imputed as needed.')

    class Error(OWWidget.Error):
        invalid_type = Msg("Cannot handle target variable type {}")
        inadequate_learner = Msg("Scorer {} inadequate: {}")
        no_attributes = Msg("Data does not have a single attribute.")

    class Warning(OWWidget.Warning):
        renamed_variables = Msg(
            "Variables with duplicated names have been renamed.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.scorers: List[ScoreMeta] = []
        self.out_domain_desc = None
        self.data = None
        self.problem_type_mode = ProblemType.CLASSIFICATION

        # results caches
        self.scorers_results = {}
        self.methods_results = {}

        if not self.selected_methods:
            self.selected_methods = {method.name for method in SCORES
                                     if method.is_default}

        # GUI
        self.ranksModel = model = TableModel(parent=self)  # type: TableModel
        self.ranksView = view = TableView(self)            # type: TableView
        self.mainArea.layout().addWidget(view)
        view.setModel(model)
        view.setColumnWidth(NVAL_COL, 30)
        view.selectionModel().selectionChanged.connect(self.on_select)

        def _set_select_manual():
            self.setSelectionMethod(OWRank.SelectManual)

        view.manualSelection.connect(_set_select_manual)
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

        selMethBox = gui.vBox(self.buttonsArea, "Select Attributes")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
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

        s = gui.spin(selMethBox, self, "nSelected", 1, 999,
                     callback=lambda: self.setSelectionMethod(OWRank.SelectNBest),
                     addToLayout=False)

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(b4, 3, 0)
        grid.addWidget(s, 3, 1)

        self.selectButtons.button(self.selectionMethod).setChecked(True)

        selMethBox.layout().addLayout(grid)

        gui.auto_send(self.buttonsArea, self, "auto_apply")

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
        self.selected_attrs = []
        self.ranksModel.clear()
        self.ranksModel.resetSorting(True)

        self.scorers_results = {}
        self.methods_results = {}
        self.cancel()

        self.Error.clear()
        self.Information.clear()
        self.Information.missings_imputed(
            shown=data is not None and data.has_missing())

        if data is not None and not data.domain.attributes:
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

            self.selectionMethod = OWRank.SelectNBest

        self.openContext(data)
        self.selectButtons.button(self.selectionMethod).setChecked(True)

    def handleNewSignals(self):
        self.setStatusMessage('Running')
        self.update_scores()
        self.setStatusMessage('')
        self.on_select()

    @Inputs.scorer
    def set_learner(self, index, scorer):
        self.scorers[index] = ScoreMeta(
            scorer.name, scorer.name, scorer,
            ProblemType.from_variable(scorer.class_type),
            False
        )
        self.scorers_results = {}

    @Inputs.scorer.insert
    def insert_learner(self, index: int, scorer):
        self.scorers.insert(index, ScoreMeta(
            scorer.name, scorer.name, scorer,
            ProblemType.from_variable(scorer.class_type),
            False
        ))
        self.scorers_results = {}

    @Inputs.scorer.remove
    def remove_learner(self, index):
        self.scorers.pop(index)
        self.scorers_results = {}

    def _get_methods(self):
        return [
            method
            for method in SCORES
            if (
                method.name in self.selected_methods
                and method.problem_type == self.problem_type_mode
                and (
                    not issparse(self.data.X)
                    or method.scorer.supports_sparse_data
                )
            )
        ]

    def _get_scorers(self):
        scorers = []
        for scorer in self.scorers:
            if scorer.problem_type in (
                self.problem_type_mode,
                ProblemType.UNSUPERVISED,
            ):
                scorers.append(scorer)
            else:
                self.Error.inadequate_learner(
                    scorer.name, scorer.learner_adequacy_err_msg
                )
        return scorers

    def update_scores(self):
        if self.data is None:
            self.ranksModel.clear()
            self.Outputs.scores.send(None)
            return

        self.Error.inadequate_learner.clear()

        scorers = [
            s for s in self._get_scorers() if s not in self.scorers_results
        ]
        methods = [
            m for m in self._get_methods() if m not in self.methods_results
        ]
        self.start(run, self.data, methods, scorers)

    def on_done(self, result: Results) -> None:
        self.methods_results.update(result.method_scores)
        self.scorers_results.update(result.scorer_scores)

        methods = self._get_methods()
        method_labels = tuple(m.shortname for m in methods)
        method_scores = tuple(self.methods_results[m] for m in methods)

        scores = [self.scorers_results[s] for s in self._get_scorers()]
        scorer_scores, scorer_labels = zip(*scores) if scores else ((), ())

        labels = method_labels + tuple(chain.from_iterable(scorer_labels))
        model_array = np.column_stack(
            (list(self.data.domain.attributes), )
            + (
                [float(len(a.values)) if a.is_discrete else np.nan
                 for a in self.data.domain.attributes],
            )
            + method_scores
            + scorer_scores
        )
        for column, values in enumerate(model_array.T[2:].astype(float),
                                        start=2):
            self.ranksModel.setExtremesFrom(column, values)

        self.ranksModel.wrap(model_array.tolist())
        self.ranksModel.setHorizontalHeaderLabels(('', '#',) + labels)
        self.ranksView.setColumnWidth(NVAL_COL, 40)
        self.ranksView.resizeColumnToContents(VARNAME_COL)

        # Re-apply sort
        try:
            sort_column, sort_order = self.sorting
            if sort_column < len(labels):
                # adds 2 to skip the first two columns
                self.ranksModel.sort(sort_column + 2, sort_order)
                self.ranksView.horizontalHeader().setSortIndicator(
                    sort_column + 2, sort_order
                )
        except ValueError:
            pass

        self.autoSelection()
        self.Outputs.scores.send(self.create_scores_table(labels))

    def on_exception(self, ex: Exception) -> None:
        raise ex

    def on_partial_result(self, result: Any) -> None:
        pass

    def on_select(self):
        # Save indices of attributes in the original, unsorted domain
        selected_rows = self.ranksView.selectionModel().selectedRows(0)
        row_indices = [i.row() for i in selected_rows]
        attr_indices = self.ranksModel.mapToSourceRows(row_indices)
        self.selected_attrs = [self.data.domain[idx] for idx in attr_indices]
        self.commit.deferred()

    def setSelectionMethod(self, method):
        self.selectionMethod = method
        self.selectButtons.button(method).setChecked(True)
        self.autoSelection()

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
            if self.selected_attrs is not None:
                attr_indices = [self.data.domain.attributes.index(var)
                                for var in self.selected_attrs]
                for row in model.mapFromSourceRows(attr_indices):
                    selection.append(QItemSelectionRange(
                        model.index(row, 0), model.index(row, columnCount - 1)))

        selModel.select(selection, QItemSelectionModel.ClearAndSelect)

    def headerClick(self, index):
        if index >= 2 and self.selectionMethod == OWRank.SelectNBest:
            # Reselect the top ranked attributes
            self.autoSelection()

        # Store the header states
        sort_order = self.ranksModel.sortOrder()
        sort_column = self.ranksModel.sortColumn() - 2  # -2 for '#' (discrete count) column
        self.sorting = (sort_column, sort_order)

    def methodSelectionChanged(self, state, method_name):
        if state == Qt.Checked:
            self.selected_methods.add(method_name)
        elif method_name in self.selected_methods:
            self.selected_methods.remove(method_name)

        self.update_scores()

    def send_report(self):
        if not self.data:
            return
        self.report_domain("Input", self.data.domain)
        self.report_table("Ranks", self.ranksView, num_format="{:.3f}")
        if self.out_domain_desc is not None:
            self.report_items("Output", self.out_domain_desc)

    @gui.deferred
    def commit(self):
        if not self.selected_attrs:
            self.Outputs.reduced_data.send(None)
            self.Outputs.features.send(None)
            self.out_domain_desc = None
        else:
            reduced_domain = Domain(
                self.selected_attrs, self.data.domain.class_var,
                self.data.domain.metas)
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)
            self.Outputs.features.send(AttributeList(self.selected_attrs))
            self.out_domain_desc = report.describe_domain(data.domain)

    def create_scores_table(self, labels):
        self.Warning.renamed_variables.clear()
        model_list = self.ranksModel.tolist()
        if not model_list or len(model_list[0]) == 2:  # Empty or just first two columns
            return None
        unique, renamed = get_unique_names_duplicates(labels + ('Feature',),
                                                      return_duplicated=True)
        if renamed:
            self.Warning.renamed_variables(', '.join(renamed))

        domain = Domain([ContinuousVariable(label) for label in unique[:-1]],
                        metas=[StringVariable(unique[-1])])

        # Prevent np.inf scores
        finfo = np.finfo(np.float64)
        scores = np.clip(np.array(model_list)[:, 2:], finfo.min, finfo.max)

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
        if version is None or version < 3:
            # Selections were stored as indices, so these contexts matched
            # any domain. The only safe thing to do is to remove them.
            raise IncompatibleContext


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import RandomForestLearner
    previewer = WidgetPreview(OWRank)
    previewer.run(Table("heart_disease.tab"), no_exit=True)
    previewer.send_signals(
        set_learner=(RandomForestLearner(), (3, 'Learner', None)))
    previewer.run()

    # pylint: disable=pointless-string-statement
    """
    WidgetPreview(OWRank).run(
        set_learner=(RandomForestLearner(), (3, 'Learner', None)),
        set_data=Table("heart_disease.tab"))
    """
