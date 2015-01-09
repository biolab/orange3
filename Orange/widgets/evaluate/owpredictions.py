"""
Predictions widget

"""

from collections import OrderedDict, namedtuple

import numpy
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtCore import pyqtSignal as Signal

import Orange.data
import Orange.classification
from Orange.classification import Model
from Orange.evaluation import testing

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


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
    priority = 200
    description = "Displays predictions of models for a particular data set."
    inputs = [("Data", Orange.data.Table, "setData"),
              ("Predictors", Orange.classification.Model,
               "setPredictor", widget.Multiple)]
    outputs = [("Predictions", Orange.data.Table),
               ("Evaluation Results", testing.Results)]

    show_probabilities = Setting(True)
    show_class = Setting(True)

    def __init__(self):
        super().__init__()

        # Control GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(
            box, "No data on input\nPredictors: 0\nTask: N/A"
        )
        self.infolabel.setMinimumWidth(200)

        box = gui.widgetBox(self.controlArea, "Options")
        self.checkbox_class = gui.checkBox(box, self, "show_class",
                                           "Show predicted class",
                                           callback=self.flipClass)
        self.checkbox_prob = gui.checkBox(box, self, "show_probabilities",
                                          "Show predicted probabilities",
                                          callback=self.flipProb)
        QtGui.qApp.processEvents()
        QtCore.QTimer.singleShot(0, self.fix_size)

        #: input data
        self.data = None

        #: A dict mapping input ids to PredictorSlot
        self.predictors = OrderedDict()

        #: A class variable (prediction target)
        self.class_var = None

    def fix_size(self):
        self.adjustSize()
        self.targets = "None"
        self.setFixedSize(self.size())

    def flipClass(self):
        if not self.show_class and not self.show_probabilities:
            self.checkbox_class.setChecked(True)
        self.commit()

    def flipProb(self):
        if not self.show_class and not self.show_probabilities:
            self.checkbox_prob.setChecked(True)
        self.commit()

    def setData(self, data):
        """Set the input data to predict on."""
        self.data = data
        self.invalidatePredictions()

    def setPredictor(self, predictor=None, id=None):
        """Set input predictor."""
        if id in self.predictors:
            self.predictors[id] = self.predictors[id]._replace(
                predictor=predictor, name=pname(predictor), results=None
            )
        else:
            self.predictors[id] = PredictorSlot(predictor, pname(predictor),
                                                None)

        if predictor is not None:
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
                self.checkbox_class.setEnabled(True)
                self.checkbox_prob.setEnabled(True)
            else:
                info.append("Task: Regression")
                self.checkbox_class.setEnabled(False)
                self.checkbox_prob.setEnabled(False)
        else:
            info.append("Task: N/A")

        self.infolabel.setText("\n".join(info))

        self.commit()

    def invalidatePredictions(self):
        """Invalidate all prediction results."""
        for inputid, pred in list(self.predictors.items()):
            self.predictors[inputid] = pred._replace(results=None)

    def commit(self):
        if self.data is None or not self.predictors:
            self.send("Predictions", None)
            self.send("Evaluation Results", None)
            return

        predictor = next(iter(self.predictors.values())).predictor
        class_var = predictor.domain.class_var
        classification = is_discrete(class_var)

        newattrs = []
        newcolumns = []
        slots = list(self.predictors.values())

        if classification:
            if self.show_class:
                mc = [Orange.data.DiscreteVariable(
                          name=p.name, values=class_var.values)
                      for p in slots]
                newattrs.extend(mc)
                newcolumns.extend(p.results[0].reshape((-1, 1))
                                  for p in slots)

            if self.show_probabilities:
                for p in slots:
                    m = [Orange.data.ContinuousVariable(
                             name="%s(%s)" % (p.name, value))
                         for value in class_var.values]
                    newattrs.extend(m)
                newcolumns.extend(p.results[1] for p in slots)

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

        if newcolumns:
            newcolumns = [numpy.atleast_2d(cols) for cols in newcolumns]
            newcolumns = numpy.hstack(tuple(newcolumns))
        else:
            newcolumns = None

        predictions = Orange.data.Table.from_numpy(
            domain, self.data.X, self.data.Y, metas=newcolumns
        )

        predictions.name = self.data.name

        results = None
        if self.data.domain.class_var == class_var:
            N = len(self.data)
            results = testing.Results(self.data, store_data=True)
            results.folds = None
            results.row_indices = numpy.arange(N)
            results.actual = self.data.Y.ravel()
            results.predicted = numpy.vstack(
                tuple(p.results[0] for p in slots))
            if classification:
                results.probabilities = numpy.array(
                    [p.results[1] for p in slots])
            results.fitter_names = [pname(p.predictor) for p in slots]

        self.send("Predictions", predictions)
        self.send("Evaluation Results", results)


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


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


if __name__ == "__main__":
    import Orange.classification.svm as svm
    import Orange.classification.logistic_regression as lr
    app = QtGui.QApplication([])
    w = OWPredictions()
    data = Orange.data.Table("iris")
    svm_clf = svm.SVMLearner(probability=True)(data)
    lr_clf = lr.LogisticRegressionLearner()(data)
    w.setData(data)
    w.setPredictor(svm_clf, 0)
    w.setPredictor(lr_clf, 1)
    w.handleNewSignals()
    w.show()
    app.exec_()
    w.saveSettings()
