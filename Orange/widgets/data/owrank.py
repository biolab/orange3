"""
Rank
====

Rank (score) features for prediction.

"""

from collections import namedtuple

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import Orange
from Orange.preprocess import score
import Orange.preprocess.discretize
from Orange.widgets import widget, settings, gui


def table(shape, fill=None):
    """ Return a 2D table with shape filed with ``fill``
    """
    return [[fill for j in range(shape[1])] for i in range(shape[0])]


_score_meta = namedtuple(
    "_score_meta",
    ["name",
     "shortname",
     "score",
     "supports_regression",
     "supports_classification",
     "handles_discrete",
     "handles_continuous"]
)


class score_meta(_score_meta):
    # Add sensible defaults to __new__
    def __new__(cls, name, shortname, score,
                supports_regression=True, supports_classification=True,
                handles_continuous=True, handles_discrete=True):
        return _score_meta.__new__(
            cls, name, shortname, score,
            supports_regression, supports_classification,
            handles_discrete, handles_continuous
        )

# Default scores.
SCORES = [
    score_meta(
        "Information Gain", "Inf. gain", score.InfoGain,
        supports_regression=False,
        supports_classification=True,
        handles_continuous=False,
        handles_discrete=True),
    score_meta(
        "Gain Ratio", "Gain Ratio", score.GainRatio,
        supports_regression=False,
        handles_continuous=False,
        handles_discrete=True),
    score_meta(
        "Gini Gain", "Gini", score.Gini,
        supports_regression=False,
        supports_classification=True,
        handles_continuous=False),
]

_DEFAULT_SELECTED = set(m.name for m in SCORES)


class OWRank(widget.OWWidget):
    name = "Rank"
    description = "Rank data features by their correlation to " \
                  "the class variable."
    icon = "icons/Rank.svg"
    priority = 1102

    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Reduced Data", Orange.data.Table)]

    SelectNone, SelectAll, SelectManual, SelectNBest = range(4)

    selectMethod = settings.Setting(SelectNBest)
    nSelected = settings.Setting(5)
    auto_apply = settings.Setting(True)

    # Header state for discrete/continuous scores
    headerState = settings.Setting((None, None))

    def __init__(self):
        super().__init__()

        self.all_measures = SCORES

        self.selectedMeasures = dict(
            [(name, True) for name in _DEFAULT_SELECTED] +
            [(m.name, False)
             for m in self.all_measures[len(_DEFAULT_SELECTED):]]
        )
        # Discrete (0) or continuous (1) class mode
        self.rankMode = 0

        self.data = None

        self.discMeasures = [m for m in self.all_measures
                             if m.supports_classification]
        self.contMeasures = [m for m in self.all_measures
                             if m.supports_regression]

        selMethBox = gui.widgetBox(
            self.controlArea, "Select attributes", addSpace=True)

        grid = QtGui.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.selectButtons = QtGui.QButtonGroup()
        self.selectButtons.buttonClicked[int].connect(self.setSelectMethod)

        def button(text, buttonid, toolTip=None):
            b = QtGui.QRadioButton(text)
            self.selectButtons.addButton(b, buttonid)
            if toolTip is not None:
                b.setToolTip(toolTip)
            return b

        b1 = button(self.tr("None"), OWRank.SelectNone)
        b2 = button(self.tr("All"), OWRank.SelectAll)
        b3 = button(self.tr("Manual"), OWRank.SelectManual)
        b4 = button(self.tr("Best ranked"), OWRank.SelectNBest)

        s = gui.spin(selMethBox, self, "nSelected", 1, 100,
                     callback=self.nSelectedChanged)

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(b4, 3, 0)
        grid.addWidget(s, 3, 1)

        self.selectButtons.button(self.selectMethod).setChecked(True)

        selMethBox.layout().addLayout(grid)

        gui.auto_commit(self.controlArea, self, "auto_apply", "Commit",
                        checkbox_label="Commit on any change")

        gui.rubber(self.controlArea)

        # Discrete and continuous table views are stacked
        self.ranksViewStack = QtGui.QStackedLayout()
        self.mainArea.layout().addLayout(self.ranksViewStack)

        self.discRanksView = QtGui.QTableView()
        self.ranksViewStack.addWidget(self.discRanksView)
        self.discRanksView.setSelectionBehavior(QtGui.QTableView.SelectRows)
        self.discRanksView.setSelectionMode(QtGui.QTableView.MultiSelection)
        self.discRanksView.setSortingEnabled(True)

        self.discRanksModel = QtGui.QStandardItemModel(self)
        self.discRanksModel.setHorizontalHeaderLabels(
            ["#"] + [m.shortname for m in self.discMeasures]
        )

        self.discRanksProxyModel = MySortProxyModel(self)
        self.discRanksProxyModel.setSourceModel(self.discRanksModel)
        self.discRanksView.setModel(self.discRanksProxyModel)

        self.discRanksView.setColumnWidth(0, 20)
        self.discRanksView.sortByColumn(1, Qt.DescendingOrder)
        self.discRanksView.selectionModel().selectionChanged.connect(
            self.onSelectionChanged
        )
        self.discRanksView.pressed.connect(self.onSelectItem)
        self.discRanksView.horizontalHeader().sectionClicked.connect(
            self.headerClick
        )

        if self.headerState[0] is not None:
            self.discRanksView.horizontalHeader().restoreState(
            self.headerState[0]
        )

        self.contRanksView = QtGui.QTableView()
        self.ranksViewStack.addWidget(self.contRanksView)
        self.contRanksView.setSelectionBehavior(QtGui.QTableView.SelectRows)
        self.contRanksView.setSelectionMode(QtGui.QTableView.MultiSelection)
        self.contRanksView.setSortingEnabled(True)

        self.contRanksModel = QtGui.QStandardItemModel(self)
        self.contRanksModel.setHorizontalHeaderLabels(
            ["#"] + [m.shortname for m in self.contMeasures]
        )

        self.contRanksProxyModel = MySortProxyModel(self)
        self.contRanksProxyModel.setSourceModel(self.contRanksModel)
        self.contRanksView.setModel(self.contRanksProxyModel)

        self.discRanksView.setColumnWidth(0, 20)
        self.contRanksView.sortByColumn(1, Qt.DescendingOrder)
        self.contRanksView.selectionModel().selectionChanged.connect(
            self.onSelectionChanged
        )
        self.contRanksView.pressed.connect(self.onSelectItem)
        self.contRanksView.horizontalHeader().sectionClicked.connect(
            self.headerClick
        )
        if self.headerState[1] is not None:
            self.contRanksView.horizontalHeader().restoreState(
            self.headerState[1]
        )

        # Switch the current view to Discrete
        self.switchRanksMode(0)
        self.resetInternals()
        self.updateDelegates()
        self.updateVisibleScoreColumns()

        self.resize(690, 500)

        self.measure_scores = table((len(self.measures), 0), None)

    def switchRanksMode(self, index):
        """
        Switch between discrete/continuous mode
        """
        self.rankMode = index
        self.ranksViewStack.setCurrentIndex(index)

        if index == 0:
            self.ranksView = self.discRanksView
            self.ranksModel = self.discRanksModel
            self.ranksProxyModel = self.discRanksProxyModel
            self.measures = self.discMeasures
        else:
            self.ranksView = self.contRanksView
            self.ranksModel = self.contRanksModel
            self.ranksProxyModel = self.contRanksProxyModel
            self.measures = self.contMeasures

        self.updateVisibleScoreColumns()

    def setData(self, data):
        self.error()
        self.resetInternals()

        if data is not None and not data.domain.class_var:
            data = None
            self.error(100, "")

        self.data = data
        if self.data is not None:
            attrs = self.data.domain.attributes
            self.usefulAttributes = [attr for attr in attrs
                                     if attr.is_discrete or attr.is_continuous]

            if self.data.domain.has_continuous_class:
                self.switchRanksMode(1)
            elif self.data.domain.has_discrete_class:
                self.switchRanksMode(0)
            else:
                # String or other.
                self.error(0, "Cannot handle class variable type %r" %
                           type(self.data.domain.class_var).__name__)

            self.ranksModel.setRowCount(len(attrs))
            for i, a in enumerate(attrs):
                if a.is_discrete:
                    v = len(a.values)
                else:
                    v = "C"
                item = ScoreValueItem()
                item.setData(v, Qt.DisplayRole)
                self.ranksModel.setItem(i, 0, item)
                item = QtGui.QStandardItem(a.name)
                item.setData(gui.attributeIconDict[a], Qt.DecorationRole)
                self.ranksModel.setVerticalHeaderItem(i, item)

            self.measure_scores = table((len(self.measures),
                                         len(attrs)), None)
            self.updateScores()

        self.unconditional_commit()

    def updateScores(self, measuresMask=None):
        """
        Update the current computed scores.

        If `measuresMask` is given it must be an list of bool values
        indicating what measures should be recomputed.

        """
        if not self.data:
            return

        measures = self.measures
        # Invalidate all warnings
        self.warning(range(max(len(self.discMeasures),
                               len(self.contMeasures))))

        if measuresMask is None:
            # Update all selected measures
            measuresMask = [self.selectedMeasures.get(m.name)
                            for m in measures]

        data = self.data

        for index, (meas, mask) in enumerate(zip(measures, measuresMask)):
            if not mask:
                continue
            estimator = meas.score()

            if not meas.handles_continuous:
                data = self.getDiscretizedData()
                attr_map = data.attrDict
                data = self.data
            else:
                attr_map, data = {}, self.data

            attr_scores = []
            for attr in data.domain.attributes:
                attr = attr_map.get(attr, attr)
                s = None
                if attr is not None:
                    try:
                        s = float(estimator(data, attr))
                    except Exception as ex:
                        self.warning(index, "Error evaluating %r: %r" %
                                     (meas.name, str(ex)))
                attr_scores.append(s)
            self.measure_scores[index] = attr_scores

        self.updateRankModel(measuresMask)
        self.ranksProxyModel.invalidate()

        if self.selectMethod in [0, 2]:
            self.autoSelection()

    def updateRankModel(self, measuresMask=None):
        """
        Update the rankModel.
        """
        values = []
        for i, scores in enumerate(self.measure_scores):
            values_one = []
            for j, score in enumerate(scores):
                values_one.append(score)
                item = self.ranksModel.item(j, i + 1)
                if not item:
                    item = ScoreValueItem()
                    self.ranksModel.setItem(j, i + 1, item)
                item.setData(score, Qt.DisplayRole)
            values.append(values_one)

        for i, vals in enumerate(values):
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
        self.discretizedData = None
        self.usefulAttributes = []
        self.ranksModel.setRowCount(0)

    def onSelectionChanged(self, *args):
        """
        Called when the ranks view selection changes.
        """
        self.data_changed()

    def onSelectItem(self, index):
        """
        Called when the user selects/unselects an item in the table view.
        """
        self.selectMethod = OWRank.SelectManual  # Manual
        self.selectButtons.button(self.selectMethod).setChecked(True)
        self.data_changed()

    def setSelectMethod(self, method):
        if self.selectMethod != method:
            self.selectMethod = method
            self.selectButtons.button(method).setChecked(True)
            self.selectMethodChanged()

    def selectMethodChanged(self):
        if self.selectMethod in [OWRank.SelectNone, OWRank.SelectAll,
                                 OWRank.SelectNBest]:
            self.autoSelection()

    def nSelectedChanged(self):
        self.selectMethod = OWRank.SelectNBest
        self.selectButtons.button(self.selectMethod).setChecked(True)
        self.selectMethodChanged()

    def getDiscretizedData(self):
        if not self.discretizedData:
            discretizer = Orange.preprocess.discretize.EqualFreq(n=4)
            contAttrs = [attr for attr in self.data.domain.attributes
                         if attr.is_continuous]
            at = []
            attrDict = {}
            for attri in contAttrs:
                try:
                    nattr = discretizer(attri, self.data)
                    at.append(nattr)
                    attrDict[attri] = nattr
                except:
                    pass
            domain = Orange.data.Domain(at, self.data.domain.class_var)
            self.discretizedData = Orange.data.Table(domain, self.data)
            self.discretizedData.attrDict = attrDict
        return self.discretizedData

    def autoSelection(self):
        selModel = self.ranksView.selectionModel()
        rowCount = self.ranksModel.rowCount()
        columnCount = self.ranksModel.columnCount()
        model = self.ranksProxyModel

        if self.selectMethod == OWRank.SelectNone:
            selection = QtGui.QItemSelection()
        elif self.selectMethod == OWRank.SelectAll:
            selection = QtGui.QItemSelection(
                model.index(0, 0),
                model.index(rowCount - 1, columnCount - 1)
            )
            selModel.select(selection,
                            QtGui.QItemSelectionModel.ClearAndSelect)
        elif self.selectMethod == OWRank.SelectNBest:
            nSelected = min(self.nSelected, rowCount)
            selection = QtGui.QItemSelection(
                model.index(0, 0),
                model.index(nSelected - 1, columnCount - 1)
            )
        else:
            selection = QtGui.QItemSelection()

        selModel.select(selection, QtGui.QItemSelectionModel.ClearAndSelect)

    def headerClick(self, index):
        if index >= 1 and self.selectMethod == OWRank.SelectNBest:
            # Reselect the top ranked attributes
            self.autoSelection()

        # Store the header states
        disc = bytes(self.discRanksView.horizontalHeader().saveState())
        cont = bytes(self.contRanksView.horizontalHeader().saveState())
        self.headerState = (disc, cont)

    def measuresSelectionChanged(self, measure=None):
        """Measure selection has changed. Update column visibility.
        """
        if measure is None:
            # Update all scores
            measuresMask = None
        else:
            # Update scores for shown column if they are not yet computed.
            shown = self.selectedMeasures.get(measure.name, False)
            index = self.measures.index(measure)
            if all(s is None for s in self.measure_scores[index]) and shown:
                measuresMask = [m == measure for m in self.measures]
            else:
                measuresMask = [False] * len(self.measures)
        self.updateScores(measuresMask)

        self.updateVisibleScoreColumns()

    def updateVisibleScoreColumns(self):
        """
        Update the visible columns of the scores view.
        """
        for i, measure in enumerate(self.measures):
            shown = self.selectedMeasures.get(measure.name)
            self.ranksView.setColumnHidden(i + 1, not shown)

    def updateDelegates(self):
        self.contRanksView.setItemDelegate(
            gui.ColoredBarItemDelegate(self)
        )

        self.discRanksView.setItemDelegate(
            gui.ColoredBarItemDelegate(self)
        )

    def sendReport(self):
        self.reportData(self.data)
        self.reportRaw(gui.reportTable(self.ranksView))

    def data_changed(self):
        self.commit()

    def commit(self):
        selected = self.selectedAttrs()
        if not self.data or not selected:
            self.send("Reduced Data", None)
        else:
            domain = Orange.data.Domain(selected, self.data.domain.class_var,
                                        metas=self.data.domain.metas)
            data = Orange.data.Table(domain, self.data)
            self.send("Reduced Data", data)

    def selectedAttrs(self):
        if self.data:
            inds = self.ranksView.selectionModel().selectedRows(0)
            source = self.ranksProxyModel.mapToSource
            inds = map(source, inds)
            inds = [ind.row() for ind in inds]
            return [self.data.domain.attributes[i] for i in inds]
        else:
            return []


class ScoreValueItem(QtGui.QStandardItem):
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


class MySortProxyModel(QtGui.QSortFilterProxyModel):
    def lessThan(self, left, right):
        role = self.sortRole()
        left_data = left.data(role)
        right_data = right.data(role)
        try:
            return left_data < right_data
        except TypeError:
            return left < right


if __name__ == "__main__":
    a = QtGui.QApplication([])
    ow = OWRank()
    ow.setData(Orange.data.Table("wine.tab"))
    ow.setData(Orange.data.Table("zoo.tab"))
#     ow.setData(Orange.data.Table("servo.tab"))
#     ow.setData(Orange.data.Table("iris.tab"))
#     ow.setData(orange.ExampleTable("auto-mpg.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()
