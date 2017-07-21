"""
Rank
====

Rank (score) features for prediction.

"""

from collections import namedtuple

import numpy as np
from scipy.sparse import issparse

from AnyQt.QtGui import QFontMetrics
from AnyQt.QtWidgets import (
    QTableView, QRadioButton, QButtonGroup, QGridLayout, QSizePolicy,
    QStackedLayout, QStackedWidget, QWidget, QHeaderView,
)
from AnyQt.QtCore import (
    Qt, QItemSelection, QItemSelectionRange, QItemSelectionModel,
    QSize,
)

from Orange.base import Learner
from Orange.data import (Table, Domain, ContinuousVariable, DiscreteVariable,
                         StringVariable)
from Orange.preprocess import score
from Orange.canvas import report
from Orange.widgets import gui
from Orange.widgets.settings import (DomainContextHandler, Setting,
                                     ContextSetting)
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg, Input, Output


def table(shape, fill=None):
    """ Return a 2D table with shape filed with ``fill``
    """
    return np.full(shape, fill).tolist()


ScoreMeta = namedtuple("score_meta", ["name", "shortname", "score"])

# Default scores.
SCORES = [ScoreMeta("Information Gain", "Info. gain", score.InfoGain),
          ScoreMeta("Information Gain Ratio", "Gain ratio", score.GainRatio),
          ScoreMeta("Gini Decrease", "Gini", score.Gini),
          ScoreMeta("ANOVA", "ANOVA", score.ANOVA),
          ScoreMeta("χ²", "χ²", score.Chi2),
          ScoreMeta("ReliefF", "ReliefF", score.ReliefF),
          ScoreMeta("FCBF", "FCBF", score.FCBF),
          ScoreMeta("Univariate Regression", "Univar. reg.",
                    score.UnivariateLinearRegression),
          ScoreMeta("RReliefF", "RReliefF", score.RReliefF)]


class TableView(QTableView):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent,
                         selectionBehavior=QTableView.SelectRows,
                         selectionMode=QTableView.ExtendedSelection,
                         sortingEnabled=True,
                         showGrid=False,
                         cornerButtonEnabled=False,
                         alternatingRowColors=True,
                         **kwargs)

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

    def data(self, index, role=Qt.DisplayRole):
        if role == gui.BarRatioRole and index.isValid():
            value = super().data(index, Qt.EditRole)
            if not isinstance(value, float):
                return None
            vmin, vmax = self._extremes.get(index.column(), (-np.inf, np.inf))
            value = (value - vmin) / ((vmax - vmin) or 1)
            return value

        if role == Qt.DisplayRole:
            role = Qt.EditRole

        return super().data(index, role)

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
        self.learners = {}
        self.labels = []
        self.out_domain_desc = None

        self.selectedMeasures = dict([(m.name, True) for m in SCORES])
        # Discrete (0) or continuous (1) class mode
        self.rankMode = 0
        self.ranksModel = None  # type: TableModel
        self.ranksView = None  # type: TableView

        self.data = None

        self.discMeasures = [m for m in SCORES if
                             issubclass(DiscreteVariable, m.score.class_type)]
        self.contMeasures = [m for m in SCORES if
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

        # Discrete, continuous and no_class table views and models

        self.discRanksLabels = ["#"] + [m.shortname for m in self.discMeasures]
        self.discRanksModel = TableModel(parent=self)
        self.discRanksModel.setHorizontalHeaderLabels(self.discRanksLabels)

        self.discRanksView = TableView(self)
        self.discRanksView.setModel(self.discRanksModel)

        self.contRanksLabels = ["#"] + [m.shortname for m in self.contMeasures]
        self.contRanksModel = TableModel(parent=self)
        self.contRanksModel.setHorizontalHeaderLabels(self.contRanksLabels)

        self.contRanksView = TableView(self)
        self.contRanksView.setModel(self.contRanksModel)

        self.noClassRanksLabels = ["#"]
        self.noClassRanksModel = TableModel(parent=self)
        self.noClassRanksModel.setHorizontalHeaderLabels(self.noClassRanksLabels)

        self.noClassRanksView = TableView()
        self.noClassRanksView.setModel(self.noClassRanksModel)

        for i, view in enumerate((self.discRanksView,
                                  self.contRanksView,
                                  self.noClassRanksView)):
            view.setColumnWidth(0, 30)
            view.selectionModel().selectionChanged.connect(self.commit)
            view.pressed.connect(self.onSelectItem)
            view.horizontalHeader().sectionClicked.connect(self.headerClick)
            view.verticalHeader().sectionClicked.connect(self.onSelectItem)

            if self.headerState[i] is not None:
                view.horizontalHeader().restoreState(self.headerState[i])

        # Discrete, continuous and no_class table views are stacked
        self.ranksViewStack = QStackedLayout()
        self.ranksViewStack.addWidget(self.discRanksView)
        self.ranksViewStack.addWidget(self.contRanksView)
        self.ranksViewStack.addWidget(self.noClassRanksView)
        self.mainArea.layout().addLayout(self.ranksViewStack)

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
            self.measures = self.discMeasures
            self.selected_checks = self.cls_default_selected
            self.reg_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)
            self.cls_scoring_box.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Expanding)
        elif index == 1:
            self.ranksView = self.contRanksView
            self.ranksModel = self.contRanksModel
            self.measures = self.contMeasures
            self.selected_checks = self.reg_default_selected
            self.cls_scoring_box.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Ignored)
            self.reg_scoring_box.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Expanding)
        else:
            self.ranksView = self.noClassRanksView
            self.ranksModel = self.noClassRanksModel
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

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.closeContext()
        self.selected_rows = []
        self.clear_messages()
        self.resetInternals()

        self.data = data
        self.switchRanksMode(0)
        if self.data is not None:
            domain = self.data.domain
            attrs = [attr for attr in domain.attributes if attr.is_primitive()]

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

            model_list = [
                [len(a.values) if a.is_discrete else '']
                for a in attrs
            ]
            self.ranksModel.wrap(model_list)
            self.ranksModel.setVerticalHeaderLabels(attrs)

            max_label = max((a.name for a in attrs), key=len)
            self.contRanksView.setVHeaderFixedWidthFromLabel(max_label)
            self.discRanksView.setVHeaderFixedWidthFromLabel(max_label)
            self.noClassRanksView.setVHeaderFixedWidthFromLabel(max_label)

            shape = (len(self.measures) + len(self.learners), len(attrs))
            self.measure_scores = table(shape, None)
            self.updateScores()
        else:
            self.Outputs.scores.send(None)

        self.selected_rows = []
        self.openContext(data)
        self.selectMethodChanged()
        self.commit()

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
        self.noClassRanksModel.setHorizontalHeaderLabels(self.noClassRanksLabels)
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
            n_measure_update = sum(measuresMask)
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
                                self.measure_scores[index].append(np.nan)
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
        self.contRanksModel.setHorizontalHeaderLabels(self.contRanksLabels + self.labels)
        self.discRanksModel.setHorizontalHeaderLabels(self.discRanksLabels + self.labels)
        self.noClassRanksModel.setHorizontalHeaderLabels(self.noClassRanksLabels + self.labels)
        self.updateRankModel(measuresMask)
        self.ranksModel.invalidate()
        self.autoSelection()
        self.Outputs.scores.send(self.create_scores_table(self.labels))
        self.setStatusMessage("")

    def updateRankModel(self, measuresMask):
        """
        Update the rankModel.
        """
        model = self.ranksModel

        values = []
        diff = len(self.measure_scores) - len(measuresMask)
        if len(measuresMask):
            measuresMask += [measuresMask[-1]] * diff

        table = [[row[0]] for row in self.ranksModel.tolist()]
        model.wrap(table)

        for i, (scores, m) in enumerate(zip(self.measure_scores, measuresMask)):
            for j, _score in enumerate(scores):
                table[j].append(_score)
            values.append(scores)

        for i, (vals, m) in enumerate(zip(values, measuresMask)):
            if not any(vals):
                continue
            model.setExtremesFrom(i + 1, vals)

        self.ranksView.setColumnWidth(0, 30)

    def resetInternals(self):
        self.data = None
        self.ranksModel.clear()

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
        model = self.ranksModel
        rowCount = model.rowCount()
        columnCount = model.columnCount()

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
                for row in self.ranksModel.mapFromSource(self.selected_rows):
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

        header = self.ranksView.horizontalHeader()
        index = header.sortIndicatorSection()
        if self.ranksView.isColumnHidden(index):
            self.headerState[self.rankMode] = None
        # else:
        #     self.ranksView.sortByColumn(index, header.sortIndicatorOrder())
        #     self.autoSelection()

        if self.headerState[self.rankMode] is None:
            index = 1 + next((i for i, m in enumerate(self.measures)
                              if m.name in self.selected_checks), len(self.measures))
            self.ranksView.sortByColumn(index, Qt.DescendingOrder)
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
        self.selected_rows = self.ranksModel.mapToSource([
            i.row() for i in self.ranksView.selectionModel().selectedRows(0)])
        if self.data and len(self.data.domain.attributes) == len(self.selected_rows):
            self.selectMethod = OWRank.SelectAll
            self.selectButtons.button(self.selectMethod).setChecked(True)

        selected_attrs = []
        if self.data:
            selected_attrs = [self.data.domain.attributes[i]
                              for i in self.selected_rows]

        if not self.data or not selected_attrs:
            self.Outputs.reduced_data.send(None)
            self.out_domain_desc = None
        else:
            reduced_domain = Domain(
                selected_attrs, self.data.domain.class_var, self.data.domain.metas)
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)
            self.out_domain_desc = report.describe_domain(data.domain)

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
