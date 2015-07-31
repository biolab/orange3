"""
Predictions widget

"""

from collections import OrderedDict, namedtuple

import numpy
from PyQt4 import QtCore, QtGui

import Orange
from Orange.base import Model
from Orange.data import ContinuousVariable, DiscreteVariable
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
    description = "Display predictions of models for an input data set."
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Predictors", Model,
               "set_predictor", widget.Multiple)]
    outputs = [("Predictions", Orange.data.Table),
               ("Evaluation Results", Orange.evaluation.Results)]

    show_attrs = Setting(True)
    show_predictions = Setting(True)
    show_probabilities = Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(
            box, "No data on input\nPredictors: 0\nTask: N/A")
        self.infolabel.setMinimumWidth(150)

        box = gui.widgetBox(self.controlArea, "Output")
        self.checkbox_class = gui.checkBox(
            box, self, "show_attrs", "Original data", callback=self.commit)
        self.checkbox_class = gui.checkBox(
            box, self, "show_predictions", "Predictions", callback=self.commit)
        self.checkbox_prob = gui.checkBox(
            box, self, "show_probabilities", "Probabilities",
            callback=self.commit)

        #: input data
        self.data = None
        #: A dict mapping input ids to PredictorSlot
        self.predictors = OrderedDict()
        #: A class variable (prediction target)
        self.class_var = None

        # enforce fixed size but provide a sensible minimum width constraint.
        self.layout().activate()
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def set_data(self, data):
        self.data = data
        self.invalidate_predictions()

    def set_predictor(self, predictor=None, id=None):
        if id in self.predictors:
            self.predictors[id] = self.predictors[id]._replace(
                predictor=predictor, name=pname(predictor), results=None)
        else:
            self.predictors[id] = \
                PredictorSlot(predictor, pname(predictor), None)
        if predictor is not None:
            self.class_var = predictor.domain.class_var

    def handleNewSignals(self):
        self.error(0)
        for inputid, pred in list(self.predictors.items()):
            if pred.predictor is None:
                del self.predictors[inputid]
            elif pred.results is None or numpy.isnan(pred.results[0]).all():
                if self.data is not None:
                    try:
                        results = self.predict(pred.predictor, self.data)
                    except ValueError as err:
                        err_msg = '{}:\n'.format(pred.predictor.name) + str(err)
                        self.error(0, err_msg)
                        n, m = len(self.data), 1
                        if self.data.domain.has_discrete_class:
                            m = len(self.data.domain.class_var.values)
                        probabilities = numpy.full((n, m), numpy.nan)
                        results = (numpy.full(n, numpy.nan), probabilities)
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
            if self.class_var.is_discrete:
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

    def invalidate_predictions(self):
        for inputid, pred in list(self.predictors.items()):
            self.predictors[inputid] = pred._replace(results=None)

    def commit(self):
        if self.data is None or not self.predictors:
            self.send("Predictions", None)
            self.send("Evaluation Results", None)
            return

        predictor = next(iter(self.predictors.values())).predictor
        class_var = predictor.domain.class_var
        classification = class_var and class_var.is_discrete

        newattrs = []
        newcolumns = []
        slots = list(self.predictors.values())

        if classification:
            if self.show_predictions:
                mc = [DiscreteVariable(name=p.name, values=class_var.values)
                      for p in slots]
                newattrs.extend(mc)
                newcolumns.extend(p.results[0].reshape((-1, 1))
                                  for p in slots)

            if self.show_probabilities:
                for p in slots:
                    m = [ContinuousVariable(name="%s(%s)" % (p.name, value))
                         for value in class_var.values]
                    newattrs.extend(m)
                newcolumns.extend(p.results[1] for p in slots)

        else:
            # regression
            mc = [ContinuousVariable(name=p.name)
                  for p in self.predictors.values()]
            newattrs.extend(mc)
            newcolumns.extend(p.results[0].reshape((-1, 1))
                              for p in slots)

        if self.show_attrs:
            X = [self.data.X]
            attrs = list(self.data.domain.attributes) + newattrs
        else:
            X = []
            attrs = newattrs
        domain = Orange.data.Domain(attrs, self.data.domain.class_var,
                                    metas=self.data.domain.metas)

        if newcolumns:
            X.extend(numpy.atleast_2d(cols) for cols in newcolumns)
        if X:
            X = numpy.hstack(tuple(X))
        else:
            X = numpy.zeros((len(self.data), 0))

        predictions = Orange.data.Table.from_numpy(
            domain, X, self.data.Y, metas=self.data.metas)
        predictions.name = self.data.name

        results = None
        if self.data.domain.class_var == class_var:
            N = len(self.data)
            results = Orange.evaluation.Results(self.data, store_data=True)
            results.folds = None
            results.row_indices = numpy.arange(N)
            results.actual = self.data.Y.ravel()
            results.predicted = numpy.vstack(
                tuple(p.results[0] for p in slots))
            if classification:
                results.probabilities = numpy.array(
                    [p.results[1] for p in slots])
            results.learner_names = [pname(p.predictor) for p in slots]

        self.send("Predictions", predictions)
        self.send("Evaluation Results", results)

    @classmethod
    def predict(cls, predictor, data):
        class_var = predictor.domain.class_var
        if class_var:
            if class_var.is_discrete:
                return cls.predict_discrete(predictor, data)
            elif class_var.is_continuous:
                return cls.predict_continuous(predictor, data)

    @staticmethod
    def predict_discrete(predictor, data):
        return predictor(data, Model.ValueProbs)

    @staticmethod
    def predict_continuous(predictor, data):
        values = predictor(data, Model.Value)
        return values, [None] * len(data)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWPredictions()
    data = Orange.data.Table("iris")
#    svm_clf = Orange.classification.SVMLearner(probability=True)(data)
#    lr_clf = Orange.classification.LogisticRegressionLearner()(data)
    svm_clf = Orange.regression.RidgeRegressionLearner(alpha=1.0)(data)
    lr_clf = Orange.regression.LinearRegressionLearner()(data)
    w.set_data(data)
    w.set_predictor(svm_clf, 0)
    w.set_predictor(lr_clf, 1)
    w.handleNewSignals()
    w.show()
    app.exec()
    w.saveSettings()
