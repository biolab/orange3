"""
Predictions widget

"""

from collections import OrderedDict, namedtuple

import numpy
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

import Orange.data
import Orange.classification
from Orange.classification import Model
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting

from Orange.widgets.data.owtable import ExampleTableModel


# Input slot for the Predictors channel
PredictorSlot = namedtuple(
    "PredictorSlot",
    ["predictor",  # The `Model` instance
     "name",       # Predictor name
     "results"]    # Computed prediction results or None.
)


def pname(predictor):
    """Return a predictor name."""
    if hasattr(predictor, "name"):
        return predictor.name
    else:
        return type(predictor).__name__


class OWPredictions(widget.OWWidget):
    name = "Predictions"
    icon = "icons/Predictions.svg"
    description = "Displays predictions of models for a particular data set."
    inputs = [("Data", Orange.data.Table, "setData"),
              ("Predictors", Orange.classification.Model,
               "setPredictor", widget.Multiple)]
    outputs = [("Predictions", Orange.data.Table)]

    showProbabilities = Setting(True)
    showFullDataset = Setting(False)

    def __init__(self):
        super().__init__()

        # Control GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(
            box, "No data on input\nPredictors: 0\nTask: N/A"
        )
        self.infolabel.setMinimumWidth(200)

        box = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(box, self, "showProbabilities",
                     "Show predicted probabilities",
                     callback=self._updatePredictionDelegate)
        gui.checkBox(box, self, "showFullDataset", "Show full data set",
                     callback=self._updateDataView,
                     tooltip="Show the whole input data set or just the " +
                             "class column if available"
        )

        gui.rubber(self.controlArea)
        box = gui.widgetBox(self.controlArea, self.tr("Commit"))
#         cb = gui.checkBox(box, self, "autocommit", "Auto commit")
        button = gui.button(box, self, "Send Predictions",
                            callback=self.commit,
                            default=True)

        # Main GUI
        self.splitter = QtGui.QSplitter(
            orientation=Qt.Horizontal,
            childrenCollapsible=False,
            handleWidth=2,
        )
        self.tableview = QtGui.QTableView(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollMode=QtGui.QTableView.ScrollPerPixel
        )

        self.predictionsview = QtGui.QTableView(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollMode=QtGui.QTableView.ScrollPerPixel
        )
        self.predictionsview.setItemDelegate(PredictionsItemDelegate())
        self.predictionsview.verticalHeader().hide()

        table_sbar = self.tableview.verticalScrollBar()
        prediction_sbar = self.predictionsview.verticalScrollBar()

        prediction_sbar.valueChanged.connect(table_sbar.setValue)
        table_sbar.valueChanged.connect(prediction_sbar.setValue)

        self.tableview.verticalHeader().setDefaultSectionSize(22)
        self.predictionsview.verticalHeader().setDefaultSectionSize(22)
        self.tableview.verticalHeader().sectionResized.connect(
            lambda index, _, size:
                self.predictionsview.verticalHeader()
                    .resizeSection(index, size)
        )

        self.splitter.addWidget(self.tableview)
        self.splitter.addWidget(self.predictionsview)

        self.mainArea.layout().addWidget(self.splitter)

        #: input data
        self.data = None

        #: A dict mapping input ids to PredictorSlot
        self.predictors = OrderedDict()

        #: A class variable (prediction target)
        self.class_var = None

    def setData(self, data):
        """Set the input data to predict on."""
        self.data = data
        if data is None:
            self.tableview.setModel(None)
            self.predictionsview.setModel(None)
            self.invalidatePredictions()
        else:
            model = ExampleTableModel(data, None)
            self.tableview.setModel(model)
            self.invalidatePredictions()
        self._updateDataView()

    def setPredictor(self, predictor=None, id=None):
        """Set input predictor."""
        if id in self.predictors:
            self.predictors[id] = self.predictors[id]._replace(
                predictor=predictor, name=pname(predictor), results=None
            )
        else:
            self.predictors[id] = PredictorSlot(predictor, pname(predictor),
                                                None)

        if self.class_var is None and predictor is not None:
            self.class_var = predictor.domain.class_var

    def handleNewSignals(self):
        for inputid, pred in list(self.predictors.items()):
            if pred.predictor is None:
                del self.predictors[inputid]

            elif pred.results is None:
                if self.data is not None:
                    results = predict(pred.predictor, self.data)
                    self.predictors[inputid] = pred._replace(results=results)

        if not self.predictors:
            self.class_var = None

        self._updatePredictionDelegate()
        self._updatePredictionsModel()

        # Check for prediction target consistency
        target_vars = set([p.predictor.domain.class_var
                           for p in self.predictors.values()])
        if len(target_vars) > 1:
            self.warning(0, "Inconsistent class variables")
        else:
            self.warning(0)

        # Update the Info box text.
        info = []
        if self.data is not None:
            info.append("Data: {} instances.".format(len(self.data)))
        else:
            info.append("Data: N/A")

        if self.predictors:
            info.append("Predictors: {}".format(len(self.predictors)))
        else:
            info.append("Predictors: N/A")

        if self.class_var is not None:
            if is_discrete(self.class_var):
                info.append("Task: Classification")
            else:
                info.append("Task: Regression")
        else:
            info.append("Task: N/A")

        self.infolabel.setText("\n".join(info))

        self.commit()

    def invalidatePredictions(self):
        """Invalidate all prediction results."""
        for inputid, pred in list(self.predictors.items()):
            self.predictors[inputid] = pred._replace(results=None)

    def _updatePredictionsModel(self):
        """Update the prediction view model."""
        if self.data is not None:
            slots = self.predictors.values()
            results = []
            for p in slots:
                values, prob = p.results
                if is_discrete(p.predictor.domain.class_var):
                    values = [
                        Orange.data.Value(p.predictor.domain.class_var, v)
                        for v in values
                    ]
                results.append((values, prob))
            results = list(zip(*(zip(*res) for res in results)))

            headers = [p.name for p in slots]
            model = PredictionsModel(results, headers)
        else:
            model = None

        self.predictionsview.setModel(model)

    def _updateDataView(self):
        """Update data column visibility."""
        if self.data is not None:
            for i in range(len(self.data.domain.attributes)):
                self.tableview.setColumnHidden(i, not self.showFullDataset)
            if self.data.domain.class_var:
                self.tableview.setColumnHidden(
                    len(self.data.domain.attributes), False
                )

    def _updatePredictionDelegate(self):
        """Update the predicted probability visibility state"""
        delegate = PredictionsItemDelegate()
        if self.class_var is not None:
            if self.showProbabilities and is_discrete(self.class_var):
                float_fmt = "{{dist[{}]:.1f}}"
                dist_fmt = " : ".join(
                    float_fmt.format(i)
                    for i in range(len(self.class_var.values))
                )
                dist_fmt = "{dist:.1f}"
                fmt = dist_fmt + " -> {value!s}"
            else:
                fmt = "{value!s}"
            delegate.setFormat(fmt)
            self.predictionsview.setItemDelegate(delegate)

    def commit(self):
        if self.data is None or not self.predictors:
            self.send("Predictions", None)
            return

        predictor = next(iter(self.predictors.values())).predictor
        class_var = predictor.domain.class_var
        classification = is_discrete(class_var)

        newattrs = []
        newcolumns = []
        slots = list(self.predictors.values())
        if classification:
            for p in slots:
                m = [Orange.data.ContinuousVariable(
                         name="%s(%s)" % (p.name, value))
                     for value in class_var.values]
                newattrs.extend(m)
            newcolumns.extend(p.results[1] for p in slots)

            mc = [Orange.data.DiscreteVariable(
                      name=p.name, values=class_var.values)
                  for p in slots]
            newattrs.extend(mc)
            newcolumns.extend(p.results[0].reshape((-1, 1))
                              for p in slots)

        else:
            # regression
            mc = [Orange.data.ContinuousVariable(name=p.name)
                  for p in self.predictors.values()]
            newattrs.extend(mc)
            newcolumns.extend(p.results[0].reshape((-1, 1))
                              for p in slots)

        domain = Orange.data.Domain(self.data.domain.attributes,
                                    self.data.domain.class_var,
                                    metas=tuple(newattrs))

        newcolumns = [numpy.atleast_2d(cols) for cols in newcolumns]
        newcolumns = numpy.hstack(tuple(newcolumns))

        predictions = Orange.data.Table.from_numpy(
            domain, self.data.X, self.data.Y, metas=newcolumns
        )

        predictions.name = self.data.name
        self.send("Predictions", predictions)


def predict(predictor, data):
    if isinstance(predictor.domain.class_var,
                  Orange.data.DiscreteVariable):
        return predict_discrete(predictor, data)
    elif isinstance(predictor.domain.class_var,
                   Orange.data.ContinuousVariable):
        return predict_continuous(predictor, data)


def predict_discrete(predictor, data):
    return predictor(data, Model.ValueProbs)


def predict_continuous(predictor, data):
    values = predictor(data, Model.Value)
    return values, [None] * len(data)


class PredictionsItemDelegate(QtGui.QStyledItemDelegate):
    def __init__(self, parent=None, **kwargs):
        self.__fmt = "{value!s}"
        super().__init__(parent, **kwargs)

    def setFormat(self, fmt):
        if fmt != self.__fmt:
            self.__fmt = fmt
            self.sizeHintChanged.emit(QtCore.QModelIndex())

    def displayText(self, value, locale):
        try:
            value, dist = value
        except ValueError:
            return ""
        else:
            fmt = self.__fmt
            if dist is not None:
                text = fmt.format(dist=DistFormater(dist), value=value)
            else:
                text = fmt.format(value=value)
            return text

    def paint(self, painter, option, index):
        super().paint(painter, option, index)


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, table=None, headers=None, parent=None):
        super().__init__(parent)
        self._table = [[]] if table is None else table
        self._header = [None] * len(self._table) if headers is None else headers

    def _value(self, index):
        row, column = index.row(), index.column()
        return self._table[row][column]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self._value(index)
        else:
            return None

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._table)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return max([len(row) for row in self._table] or [0])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical and  role == Qt.DisplayRole:
            return str(section + 1)
        elif orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return (self._header[section] if section < len(self._header)
                    else str(section))
        else:
            return None


class PredictionsModel(TableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole:
            value = self._value(index)
            return tool_tip(value)
        else:
            return super().data(index, role)


def tool_tip(value):
    value, dist = value
    if dist is not None:
        return "{!s} -> {!s}".format(dist, value)
    else:
        return str(value)


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class DistFormater(object):
    def __init__(self, dist):
        self.dist = dist

    def __format__(self, fmt):
        return " : ".join(("%" + fmt) % v for v in self.dist)


if __name__ == "__main__":
    import Orange.classification.naive_bayes
    app = QtGui.QApplication([])
    w = OWPredictions()
    data = Orange.data.Table("iris")
    nb = Orange.classification.naive_bayes.BayesLearner()(data)
    w.setData(data)
    w.setPredictor(nb, 1)
    w.handleNewSignals()
    w.show()
    app.exec_()
    w.saveSettings()
