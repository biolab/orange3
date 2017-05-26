"""
Rank
====

Rank (score) features for prediction.

"""

from collections import namedtuple

import numpy as np
from scipy.sparse import issparse

from AnyQt.QtWidgets import (
    QTableView, QRadioButton, QButtonGroup, QGridLayout, QSizePolicy,
    QStackedLayout, QStackedWidget, QWidget
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtCore import (
    Qt, QItemSelection, QItemSelectionRange, QItemSelectionModel,
    QSortFilterProxyModel
)

from Orange.base import Learner
from Orange.data import (Table, Domain, ContinuousVariable, DiscreteVariable,
                         StringVariable)
from Orange.preprocess import score
from Orange.canvas import report
from Orange.widgets import gui
from Orange.widgets.settings import (DomainContextHandler, Setting,
                                     ContextSetting)
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg, Input, Output


def table(shape, fill=None):
    """ Return a 2D table with shape filed with ``fill``
    """
    return [[fill for j in range(shape[1])] for i in range(shape[0])]


ScoreMeta = namedtuple("score_meta", ["name", "shortname", "score"])

# Default scores.
SCORES = [ScoreMeta("Information Gain", "Inf. gain", score.InfoGain),
          ScoreMeta("Gain Ratio", "Gain Ratio", score.GainRatio),
          ScoreMeta("Gini Decrease", "Gini", score.Gini),
          ScoreMeta("ANOVA", "ANOVA", score.ANOVA),
          ScoreMeta("Chi2", "Chi2", score.Chi2),
          ScoreMeta("ReliefF", "ReliefF", score.ReliefF),
          ScoreMeta("FCBF", "FCBF", score.FCBF),
          ScoreMeta("Univariate Linear Regression", "Univar. Lin. Reg.",
                    score.UnivariateLinearRegression),
          ScoreMeta("RReliefF", "RReliefF", score.RReliefF)]


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

    cls_default_selected = Setting({"Gain Ratio", "Gini Decrease"})
    reg_default_selected = Setting({"Univariate Linear Regression", "RReliefF"})
    selectMethod = Setting(SelectNBest)
    nSelected = Setting(5)
    auto_apply = Setting(True)

    # Header state for discrete/continuous/no_class scores
    headerState = Setting([None, None, None])

    settings_version = 1
    settingsHandler = DomainContextHandler()
    selected_rows = ContextSetting([])

    gain = inf_gain = gini = anova = chi2 = ulr = relief = rrelief = fcbc = True
    _score_vars = ["gain", "inf_gain", "gini", "anova", "chi2", "relief",
                   "fcbc", "ulr", "rrelief"]

    class Warning(OWWidget.Warning):
        no_target_var = Msg("Data does not have a target variable")

    class Error(OWWidget.Error):
        invalid_type = Msg("Cannot handle target variable type {}")
        inadequate_learner = Msg("{}")

    def __init__(self):
        super().__init__()
        self.measure_scores = None
        self.update_scores = True
        self.usefulAttributes = []
        self.learners = {}
        self.labels = []
        self.out_domain_desc = None

        self.all_measures = SCORES

        self.selectedMeasures = dict([(m.name, True) for m
                                      in self.all_measures])
        # Discrete (0) or continuous (1) class mode
        self.rankMode = 0

        self.data = None

        self.discMeasures = [m for m in self.all_measures if
                             issubclass(DiscreteVariable, m.score.class_type)]
        self.contMeasures = [m for m in self.all_measures if
                             issubclass(ContinuousVariable, m.score.class_type)]

        self.score_checks = []
        self.cls_scoring_box = gui.vBox(None, "Scoring for Classification")
        self.reg_scoring_box = gui.vBox(None, "Scoring for Regression")
        boxes = [self.cls_scoring_box] * 7 + [self.reg_scoring_box] * 2
        for _score, var, box in zip(SCORES, self._score_vars, boxes):
            check = gui.checkBox(
                box, self, var, label=_score.name,
                callback=lambda val=_score: self.measuresSelectionChanged(val))
            self.score_checks.append(check)

        self.score_stack = QStackedWidget(self)
        self.score_stack.addWidget(self.cls_scoring_box)
        self.score_stack.addWidget(self.reg_scoring_box)
        self.score_stack.addWidget(QWidget())
        self.controlArea.layout().addWidget(self.score_stack)

        gui.rubber(self.controlArea)

        selMethBox = gui.vBox(
            self.controlArea, "Select Attributes", addSpace=True)

        grid = QGridLayout()
        grid.setContentsMargins(6, 0, 6, 0)
        self.selectButtons = QButtonGroup()
        self.selectButtons.buttonClicked[int].connect(self.setSelectMethod)

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
                     callback=self.nSelectedChanged)

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(b4, 3, 0)
        grid.addWidget(s, 3, 1)

        self.selectButtons.button(self.selectMethod).setChecked(True)

        selMethBox.layout().addLayout(grid)

        gui.auto_commit(selMethBox, self, "auto_apply", "Send", box=False)

        # Discrete, continuous and no_class table views are stacked
        self.ranksViewStack = QStackedLayout()
        self.mainArea.layout().addLayout(self.ranksViewStack)

        self.discRanksView = QTableView()
        self.ranksViewStack.addWidget(self.discRanksView)
        self.discRanksView.setSelectionBehavior(QTableView.SelectRows)
        self.discRanksView.setSelectionMode(QTableView.MultiSelection)
        self.discRanksView.setSortingEnabled(True)

        self.discRanksLabels = ["#"] + [m.shortname for m in self.discMeasures]
        self.discRanksModel = QStandardItemModel(self)
        self.discRanksModel.setHorizontalHeaderLabels(self.discRanksLabels)

        self.discRanksProxyModel = MySortProxyModel(self)
        self.discRanksProxyModel.setSourceModel(self.discRanksModel)
        self.discRanksView.setModel(self.discRanksProxyModel)

        self.discRanksView.setColumnWidth(0, 20)
        self.discRanksView.selectionModel().selectionChanged.connect(
            self.commit
        )
        self.discRanksView.pressed.connect(self.onSelectItem)
        self.discRanksView.horizontalHeader().sectionClicked.connect(
            self.headerClick
        )
        self.discRanksView.verticalHeader().sectionClicked.connect(
            self.onSelectItem
        )

        if self.headerState[0] is not None:
            self.discRanksView.horizontalHeader().restoreState(
                self.headerState[0])

        self.contRanksView = QTableView()
        self.ranksViewStack.addWidget(self.contRanksView)
        self.contRanksView.setSelectionBehavior(QTableView.SelectRows)
        self.contRanksView.setSelectionMode(QTableView.MultiSelection)
        self.contRanksView.setSortingEnabled(True)

        self.contRanksLabels = ["#"] + [m.shortname for m in self.contMeasures]
        self.contRanksModel = QStandardItemModel(self)
        self.contRanksModel.setHorizontalHeaderLabels(self.contRanksLabels)

        self.contRanksProxyModel = MySortProxyModel(self)
        self.contRanksProxyModel.setSourceModel(self.contRanksModel)
        self.contRanksView.setModel(self.contRanksProxyModel)

        self.contRanksView.setColumnWidth(0, 20)
        self.contRanksView.selectionModel().selectionChanged.connect(
            self.commit
        )
        self.contRanksView.pressed.connect(self.onSelectItem)
        self.contRanksView.horizontalHeader().sectionClicked.connect(
            self.headerClick
        )
        self.contRanksView.verticalHeader().sectionClicked.connect(
            self.onSelectItem
        )

        if self.headerState[1] is not None:
            self.contRanksView.horizontalHeader().restoreState(
                self.headerState[1])

        self.noClassRanksView = QTableView()
        self.ranksViewStack.addWidget(self.noClassRanksView)
        self.noClassRanksView.setSelectionBehavior(QTableView.SelectRows)
        self.noClassRanksView.setSelectionMode(QTableView.MultiSelection)
        self.noClassRanksView.setSortingEnabled(True)

        self.noClassRanksLabels = ["#"]
        self.noClassRanksModel = QStandardItemModel(self)
        self.noClassRanksModel.setHorizontalHeaderLabels(self.noClassRanksLabels)

        self.noClassRanksProxyModel = MySortProxyModel(self)
        self.noClassRanksProxyModel.setSourceModel(self.noClassRanksModel)
        self.noClassRanksView.setModel(self.noClassRanksProxyModel)

        self.noClassRanksView.setColumnWidth(0, 20)
        self.noClassRanksView.selectionModel().selectionChanged.connect(
            self.commit
        )
        self.noClassRanksView.pressed.connect(self.onSelectItem)
        self.noClassRanksView.horizontalHeader().sectionClicked.connect(
            self.headerClick
        )
        self.noClassRanksView.verticalHeader().sectionClicked.connect(
            self.onSelectItem
        )

        if self.headerState[2] is not None:
            self.noClassRanksView.horizontalHeader().restoreState(
                self.headerState[2])

        # Switch the current view to Discrete
        self.switchRanksMode(0)
        self.resetInternals()
        self.updateDelegates()
        self.updateVisibleScoreColumns()

        self.resize(690, 500)

        self.measure_scores = table((len(self.measures), 0), None)

    def switchRanksMode(self, index):
        """
        Switch between discrete/continuous/no_class mode
        """
        self.rankMode = index
        self.ranksViewStack.setCurrentIndex(index)

        if index == 0:
            self.ranksView = self.discRanksView
            self.ranksModel = self.discRanksModel
            self.ranksProxyModel = self.discRanksProxyModel
            self.measures = self.discMeasures
            self.selected_checks = self.cls_default_selected
            self.reg_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)
            self.cls_scoring_box.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Expanding)
        elif index == 1:
            self.ranksView = self.contRanksView
            self.ranksModel = self.contRanksModel
            self.ranksProxyModel = self.contRanksProxyModel
            self.measures = self.contMeasures
            self.selected_checks = self.reg_default_selected
            self.cls_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)
            self.reg_scoring_box.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Expanding)
        else:
            self.ranksView = self.noClassRanksView
            self.ranksModel = self.noClassRanksModel
            self.ranksProxyModel = self.noClassRanksProxyModel
            self.measures = []
            self.selected_checks = set()
            self.reg_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)
            self.cls_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)

        shape = (len(self.measures) + len(self.learners), 0)
        self.measure_scores = table(shape, None)
        self.update_scores = False
        for check, score in zip(self.score_checks, SCORES):
            check.setChecked(score.name in self.selected_checks)
        self.update_scores = True
        self.score_stack.setCurrentIndex(index)
        self.updateVisibleScoreColumns()

    @check_sql_input
    @Inputs.data
    def setData(self, data):
        self.closeContext()
        self.clear_messages()
        self.resetInternals()

        self.data = data
        self.switchRanksMode(0)
        if self.data is not None:
            domain = self.data.domain
            attrs = domain.attributes
            self.usefulAttributes = [attr for attr in attrs
                                     if attr.is_discrete or attr.is_continuous]

            if domain.has_continuous_class:
                self.switchRanksMode(1)
            elif not domain.class_var:
                self.Warning.no_target_var()
                self.switchRanksMode(2)
            elif not domain.has_discrete_class:
                self.Error.invalid_type(type(domain.class_var).__name__)

            if issparse(self.data.X):   # keep only measures supporting sparse data
                self.measures = [m for m in self.measures
                                 if m.score.supports_sparse_data]

            self.ranksModel.setRowCount(len(attrs))
            for i, a in enumerate(attrs):
                if a.is_discrete:
                    v = len(a.values)
                else:
                    v = "C"
                item = ScoreValueItem()
                item.setData(v, Qt.DisplayRole)
                self.ranksModel.setItem(i, 0, item)
                item = QStandardItem(a.name)
                item.setData(gui.attributeIconDict[a], Qt.DecorationRole)
                self.ranksModel.setVerticalHeaderItem(i, item)

            shape = (len(self.measures) + len(self.learners), len(attrs))
            self.measure_scores = table(shape, None)
            self.updateScores()
        else:
            self.Outputs.scores.send(None)

        self.selected_rows = []
        self.openContext(data)
        self.selectMethodChanged()
        self.commit()

    def get_selection(self):
        selection = self.ranksView.selectionModel().selection()
        return list(set(ind.row() for ind in selection.indexes()))

    @Inputs.scorer
    def set_learner(self, learner, lid=None):
        if learner is None and lid is not None:
            del self.learners[lid]
        elif learner is not None:
            self.learners[lid] = ScoreMeta(
                learner.name,
                learner.name,
                learner
            )
        attrs_len = 0 if not self.data else len(self.data.domain.attributes)
        shape = (len(self.learners), attrs_len)
        self.measure_scores = self.measure_scores[:len(self.measures)]
        self.measure_scores += table(shape, None)
        self.contRanksModel.setHorizontalHeaderLabels(self.contRanksLabels)
        self.discRanksModel.setHorizontalHeaderLabels(self.discRanksLabels)
        self.noClassRanksModel.setHorizontalHeaderLabels(
            self.noClassRanksLabels)
        measures_mask = [False] * len(self.measures)
        measures_mask += [True for _ in self.learners]
        self.updateScores(measures_mask)
        self.commit()

    def updateScores(self, measuresMask=None):
        """
        Update the current computed scores.

        If `measuresMask` is given it must be an list of bool values
        indicating what measures should be recomputed.

        """
        if not self.data:
            return
        if self.data.has_missing():
            self.information("Missing values have been imputed.")

        measures = self.measures + [v for k, v in self.learners.items()]
        if measuresMask is None:
            # Update all selected measures
            measuresMask = [self.selectedMeasures.get(m.name)
                            for m in self.measures]
            measuresMask = measuresMask + [v.name for k, v in
                                           self.learners.items()]

        data = self.data
        learner_col = len(self.measures)
        if len(measuresMask) <= len(self.measures) or \
                measuresMask[len(self.measures)]:
            self.labels = []
            self.Error.inadequate_learner.clear()

        self.setStatusMessage("Running")
        with self.progressBar():
            n_measure_update = len([x for x in measuresMask if x is not False])
            count = 0
            for index, (meas, mask) in enumerate(zip(measures, measuresMask)):
                if not mask:
                    continue
                self.progressBarSet(90 * count / n_measure_update)
                count += 1
                if index < len(self.measures):
                    estimator = meas.score()
                    try:
                        self.measure_scores[index] = estimator(data)
                    except ValueError:
                        self.measure_scores[index] = []
                        for attr in data.domain.attributes:
                            try:
                                self.measure_scores[index].append(
                                    estimator(data, attr))
                            except ValueError:
                                self.measure_scores[index].append(None)
                else:
                    learner = meas.score
                    if isinstance(learner, Learner) and \
                            not learner.check_learner_adequacy(self.data.domain):
                        self.Error.inadequate_learner(
                            learner.learner_adequacy_err_msg)
                        scores = table((1, len(data.domain.attributes)))
                    else:
                        scores = meas.score.score_data(data)
                    for i, row in enumerate(scores):
                        self.labels.append(meas.shortname + str(i + 1))
                        if len(self.measure_scores) > learner_col:
                            self.measure_scores[learner_col] = row
                        else:
                            self.measure_scores.append(row)
                        learner_col += 1
            self.progressBarSet(90)
        self.contRanksModel.setHorizontalHeaderLabels(
            self.contRanksLabels + self.labels
        )
        self.discRanksModel.setHorizontalHeaderLabels(
            self.discRanksLabels + self.labels
        )
        self.noClassRanksModel.setHorizontalHeaderLabels(
            self.noClassRanksLabels + self.labels
        )
        self.updateRankModel(measuresMask)
        self.ranksProxyModel.invalidate()
        self.selectMethodChanged()
        self.Outputs.scores.send(self.create_scores_table(self.labels))
        self.setStatusMessage("")

    def updateRankModel(self, measuresMask):
        """
        Update the rankModel.
        """
        values = []
        diff = len(self.measure_scores) - len(measuresMask)
        if len(measuresMask):
            measuresMask += [measuresMask[-1]] * diff
        for i in range(self.ranksModel.columnCount() - 1,
                       len(self.measure_scores), -1):
            self.ranksModel.removeColumn(i)

        for i, (scores, m) in enumerate(zip(self.measure_scores, measuresMask)):
            if not m and self.ranksModel.item(0, i + 1):
                values.append([])
                continue
            values_one = []
            for j, _score in enumerate(scores):
                values_one.append(_score)
                item = self.ranksModel.item(j, i + 1)
                if not item:
                    item = ScoreValueItem()
                    self.ranksModel.setItem(j, i + 1, item)
                item.setData(_score, Qt.DisplayRole)
            values.append(values_one)
        for i, (vals, m) in enumerate(zip(values, measuresMask)):
            if not m:
                continue
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                vmin, vmax = min(valid_vals), max(valid_vals)
                for j, v in enumerate(vals):
                    if v is not None:
                        # Set the bar ratio role for i-th measure.
                        ratio = float((v - vmin) / ((vmax - vmin) or 1))
                        item = self.ranksModel.item(j, i + 1)
                        item.setData(ratio, gui.BarRatioRole)

        self.ranksView.setColumnWidth(0, 20)
        self.ranksView.resizeRowsToContents()

    def resetInternals(self):
        self.data = None
        self.usefulAttributes = []
        self.ranksModel.setRowCount(0)

    def onSelectItem(self, index):
        """
        Called when the user selects/unselects an item in the table view.
        """
        self.selectMethod = OWRank.SelectManual  # Manual
        self.selectButtons.button(self.selectMethod).setChecked(True)
        self.commit()

    def setSelectMethod(self, method):
        if self.selectMethod != method:
            self.selectMethod = method
            self.selectButtons.button(method).setChecked(True)
            self.selectMethodChanged()

    def selectMethodChanged(self):
        self.autoSelection()
        self.ranksView.setFocus()

    def nSelectedChanged(self):
        self.selectMethod = OWRank.SelectNBest
        self.selectButtons.button(self.selectMethod).setChecked(True)
        self.selectMethodChanged()

    def autoSelection(self):
        selModel = self.ranksView.selectionModel()
        rowCount = self.ranksModel.rowCount()
        columnCount = self.ranksModel.columnCount()
        model = self.ranksProxyModel

        if self.selectMethod == OWRank.SelectNone:
            selection = QItemSelection()
        elif self.selectMethod == OWRank.SelectAll:
            selection = QItemSelection(
                model.index(0, 0),
                model.index(rowCount - 1, columnCount - 1)
            )
        elif self.selectMethod == OWRank.SelectNBest:
            nSelected = min(self.nSelected, rowCount)
            selection = QItemSelection(
                model.index(0, 0),
                model.index(nSelected - 1, columnCount - 1)
            )
        else:
            selection = QItemSelection()
            if len(self.selected_rows):
                selection = QItemSelection()
                for row in self.selected_rows:
                    selection.append(QItemSelectionRange(
                        model.index(row, 0), model.index(row, columnCount - 1)))

        selModel.select(selection, QItemSelectionModel.ClearAndSelect)

    def headerClick(self, index):
        if index >= 1 and self.selectMethod == OWRank.SelectNBest:
            # Reselect the top ranked attributes
            self.autoSelection()

        # Store the header states
        disc = bytes(self.discRanksView.horizontalHeader().saveState())
        cont = bytes(self.contRanksView.horizontalHeader().saveState())
        no_class = bytes(self.noClassRanksView.horizontalHeader().saveState())
        self.headerState = [disc, cont, no_class]

    def measuresSelectionChanged(self, measure):
        """Measure selection has changed. Update column visibility.
        """
        checked = self.selectedMeasures[measure.name]
        self.selectedMeasures[measure.name] = not checked
        if not checked:
            self.selected_checks.add(measure.name)
        elif measure.name in self.selected_checks:
            self.selected_checks.remove(measure.name)
        measures_mask = [False] * len(self.measures)
        measures_mask += [False for _ in self.learners]
        # Update scores for shown column if they are not yet computed.
        if measure in self.measures and self.measure_scores:
            index = self.measures.index(measure)
            if all(s is None for s in self.measure_scores[index]):
                measures_mask[index] = True
        if self.update_scores:
            self.updateScores(measures_mask)
        self.updateVisibleScoreColumns()

    def updateVisibleScoreColumns(self):
        """
        Update the visible columns of the scores view.
        """
        for i, measure in enumerate(self.measures):
            shown = self.selectedMeasures.get(measure.name)
            self.ranksView.setColumnHidden(i + 1, not shown)
            self.ranksView.setColumnWidth(i + 1, 100)

        index = self.ranksView.horizontalHeader().sortIndicatorSection()
        if self.ranksView.isColumnHidden(index):
            self.headerState[self.rankMode] = None

        if self.headerState[self.rankMode] is None:
            def get_sort_by_col(measures, selected_measures):
                cols = [i + 1 for i, m in enumerate(measures) if
                        m.name in selected_measures]
                return cols[0] if cols else len(measures) + 1

            col = get_sort_by_col(self.measures, self.selected_checks)
            self.ranksView.sortByColumn(col, Qt.DescendingOrder)
            self.autoSelection()

    def updateDelegates(self):
        self.contRanksView.setItemDelegate(gui.ColoredBarItemDelegate(self))
        self.discRanksView.setItemDelegate(gui.ColoredBarItemDelegate(self))
        self.noClassRanksView.setItemDelegate(gui.ColoredBarItemDelegate(self))

    def send_report(self):
        if not self.data:
            return
        self.report_domain("Input", self.data.domain)
        self.report_table("Ranks", self.ranksView, num_format="{:.3f}")
        if self.out_domain_desc is not None:
            self.report_items("Output", self.out_domain_desc)

    def commit(self):
        self.selected_rows = self.get_selection()
        if self.data and len(self.data.domain.attributes) == len(
                self.selected_rows):
            self.selectMethod = OWRank.SelectAll
            self.selectButtons.button(self.selectMethod).setChecked(True)
        selected = self.selectedAttrs()
        if not self.data or not selected:
            self.Outputs.reduced_data.send(None)
            self.out_domain_desc = None
        else:
            reduced_domain = Domain(
                selected, self.data.domain.class_var, self.data.domain.metas)
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)
            self.out_domain_desc = report.describe_domain(data.domain)

    def selectedAttrs(self):
        if self.data:
            inds = self.ranksView.selectionModel().selectedRows(0)
            source = self.ranksProxyModel.mapToSource
            inds = map(source, inds)
            inds = [ind.row() for ind in inds]
            return [self.data.domain.attributes[i] for i in inds]
        else:
            return []

    def create_scores_table(self, labels):
        indices = [i for i, m in enumerate(self.measures)
                   if self.selectedMeasures.get(m.name, False)]
        measures = [s.name for s in self.measures if
                    self.selectedMeasures.get(s.name, False)]
        measures += [label for label in labels]
        if not measures:
            return None
        features = [ContinuousVariable(s) for s in measures]
        metas = [StringVariable("Feature name")]
        domain = Domain(features, metas=metas)

        scores = np.nan_to_num(np.array([row for i, row in enumerate(self.measure_scores)
                           if i in indices or i >= len(self.measures)], dtype=np.float64).T)
        feature_names = np.array([a.name for a in self.data.domain.attributes])
        # Reshape to 2d array as Table does not like 1d arrays
        feature_names = feature_names[:, None]

        new_table = Table(domain, scores, metas=feature_names)
        new_table.name = "Feature Scores"
        return new_table

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version:
            # Before fc5caa1e1d716607f1f5c4e0b0be265c23280fa0
            # headerState had length 2
            headerState = settings.get("headerState", None)
            if headerState is not None and \
                    isinstance(headerState, tuple) and \
                    len(headerState) < 3:
                headerState = (list(headerState) + [None] * 3)[:3]
                settings["headerState"] = headerState


class ScoreValueItem(QStandardItem):
    """A StandardItem subclass for python objects.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def __lt__(self, other):
        model = self.model()
        if model is not None:
            role = model.sortRole()
        else:
            role = Qt.DisplayRole
        my = self.data(role)
        other = other.data(role)
        if my is None:
            return True
        return my < other


class MySortProxyModel(QSortFilterProxyModel):

    @staticmethod
    def comparable(val):
        return val is not None and \
               (isinstance(val, str) or float('-inf') < val < float('inf'))

    def lessThan(self, left, right):
        role = self.sortRole()
        left_data = left.data(role)
        if not self.comparable(left_data):
            left_data = float('-inf')
        right_data = right.data(role)
        if not self.comparable(right_data):
            right_data = float('-inf')
        try:
            return left_data < right_data
        except TypeError:
            return str(left_data) < str(right_data)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    from Orange.classification import RandomForestLearner
    a = QApplication([])
    ow = OWRank()
    ow.setData(Table("heart_disease.tab"))
    ow.set_learner(RandomForestLearner(), (3, 'Learner', None))
    ow.commit()
    ow.show()
    a.exec_()
    ow.saveSettings()
