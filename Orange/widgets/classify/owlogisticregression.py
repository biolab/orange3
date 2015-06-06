from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt


import Orange.data
from Orange.classification import logistic_regression as lr
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui


class OWLogisticRegression(widget.OWWidget):
    name = "Logistic Regression"
    description = "Logistic regression classification algorithm with " \
                  "LASSO (L1) or ridge (L2) regularization."
    icon = "icons/LogisticRegression.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", lr.LogisticRegressionLearner),
               ("Classifier", lr.LogisticRegressionClassifier)]

    want_main_area = False

    learner_name = settings.Setting("Logistic Regression")

    penalty_type = settings.Setting(1)
    dual = settings.Setting(False)
    C = settings.Setting(1.0)
    tol = settings.Setting(0.0001)
    fit_intercept = True
    intercept_scaling = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, self.tr("Regularization"))
        form = QtGui.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        box.layout().addLayout(form)

        buttonbox = gui.radioButtonsInBox(
            box, self, "penalty_type", btnLabels=("L1", "L2"),
            orientation="horizontal"
        )
        form.addRow(self.tr("Penalty type:"), buttonbox)

        spin = gui.doubleSpin(box, self, "C", 0.0, 1024.0, step=0.0001)

        form.addRow("Reg (C):", spin)

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
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        penalty = ["l1", "l2"][self.penalty_type]
        learner = lr.LogisticRegressionLearner(
            penalty=penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            preprocessors=self.preprocessors
        )
        learner.name = self.learner_name
        classifier = None

        if self.data is not None:
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWLogisticRegression()
    w.set_data(Orange.data.Table("zoo"))
    w.show()
    app.exec_()
