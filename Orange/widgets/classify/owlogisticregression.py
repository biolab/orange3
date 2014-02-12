from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

from sklearn import linear_model as lm

import Orange.data
import Orange.classification
from Orange.widgets import widget, settings, gui


class LRLearner(Orange.classification.Fitter):
    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 intercept=True, intercept_scaling=1.0):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.intercept = intercept
        self.intercept_scaling = intercept_scaling

    def fit(self, X, Y, W):
        lr = lm.LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.intercept,
            intercept_scaling=self.intercept_scaling
        )
        clsf = lr.fit(X, Y.ravel())

        return LRClassifier(clsf)


class LRClassifier(Orange.classification.Model):
    def __init__(self, skl_model):
        self._model = skl_model

    def predict(self, X):
        value = self._model.predict(X)
        prob = self._model.predict_proba(X)
        return value, prob


class OWLogisticRegression(widget.OWWidget):
    name = "Logistic Regression"
    description = ""
    icon = "icons/LogisticRegression.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Learner", LRLearner),
               ("Classifier", LRClassifier)]

    want_main_area = False

    learner_name = settings.Setting("Logistic Regression")

    penalty_type = 1
    dual = settings.Setting(False)
    C = settings.Setting(1.0)
    tol = settings.Setting(0.0001)
    intercept = True
    intercept_scaling = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, self.tr("Regularization"))

        pbox = gui.widgetBox(box, "Penalty type")
        pbox.setFlat(True)
        gui.radioButtonsInBox(
            pbox, self, "penalty_type", btnLabels=("L1", "L2"),
            orientation="horizontal"
        )

        gui.doubleSpin(box, self, "C", 0.0, 1024.0, step=0.0001,
                       label="Reg (C)")

        box = gui.widgetBox(self.controlArea, "Numerical Tolerance")
        gui.doubleSpin(box, self, "tol", 1e-7, 1e-3, 5e-7)

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self.setSizePolicy(
            QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                              QtGui.QSizePolicy.Fixed)
        )
        self.setMinimumWidth(250)

        self.apply()

    def set_data(self, data):

        self.data = data

        if data is not None:
            self.data = data

            self.apply()

    def apply(self):
        penalty = ["l1", "l2"][self.penalty_type]
        learner = LRLearner(
            penalty=penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            intercept=self.intercept,
            intercept_scaling=self.intercept_scaling)
        classifier = None

        if self.data is not None:
            classifier = learner(self.data)

        self.send("Learner", learner)
        self.send("Classifier", classifier)
