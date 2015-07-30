from itertools import chain
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification import logistic_regression as lr
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui


class OWLogisticRegression(widget.OWWidget):
    name = "Logistic Regression"
    description = "Logistic regression classification algorithm with " \
                  "LASSO (L1) or ridge (L2) regularization."
    icon = "icons/LogisticRegression.svg"

    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", lr.LogisticRegressionLearner),
               ("Classifier", lr.LogisticRegressionClassifier)]

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("Logistic Regression")

    penalty_type = settings.Setting(1)
    C_index = settings.Setting(61)

    C_s = list(chain(range(1000, 200, -50),
                     range(200, 100, -10),
                     range(100, 20, -5),
                     range(20, 0, -1),
                     [x / 10 for x in range(9, 2, -1)],
                     [x / 100 for x in range(20, 2, -1)],
                     [x / 1000 for x in range(20, 0, -1)]))
    dual = False
    tol = 0.0001
    fit_intercept = True
    intercept_scaling = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, box=True)
        gui.comboBox(box, self, "penalty_type", label="Regularization type: ",
                     items=("Lasso (L1)", "Ridge (L2)"),
                     orientation="horizontal", addSpace=4)
        gui.widgetLabel(box, "Strength:")
        box2 = gui.widgetBox(gui.indentedBox(box), orientation="horizontal")
        gui.widgetLabel(box2, "Weak").setStyleSheet("margin-top:6px")
        gui.hSlider(box2, self, "C_index",
                    minValue=0, maxValue=len(self.C_s) - 1,
                    callback=self.set_c, createLabel=False)
        gui.widgetLabel(box2, "Strong").setStyleSheet("margin-top:6px")
        box2 = gui.widgetBox(box, orientation="horizontal")
        box2.layout().setAlignment(Qt.AlignCenter)
        self.c_label = gui.widgetLabel(box2)
        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)
        self.set_c()
        self.apply()

    def set_c(self):
        self.C = self.C_s[self.C_index]
        if self.C >= 1:
            frmt = "C={}"
        else:
            frmt = "C={:.3f}"
        self.c_label.setText(frmt.format(self.C))

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
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWLogisticRegression()
    w.set_data(Table("zoo"))
    w.show()
    app.exec_()
