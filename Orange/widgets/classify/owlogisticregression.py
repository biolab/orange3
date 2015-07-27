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

    learner_name = settings.Setting("Logistic Regression")

    penalty_type = settings.Setting(1)
    log_C = settings.Setting(0)

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
        gui.hSlider(box2, self, "log_C", minValue=-3, maxValue=3, step=0.01,
                    callback=self.set_c, intOnly=False, createLabel=False)
        gui.widgetLabel(box2, "Strong").setStyleSheet("margin-top:6px")
        box2 = gui.widgetBox(box, orientation="horizontal")
        box2.layout().setAlignment(Qt.AlignCenter)
        self.c_label = gui.widgetLabel(box2)
        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                                             QtGui.QSizePolicy.Fixed))
        self.setMinimumWidth(300)
        self.set_c()
        self.apply()

    def set_c(self):
        self.C = pow(10, -self.log_C)
        if self.C >= 100:
            self.C = (self.C // 25) * 25
        elif self.C >= 10:
            self.C = (self.C // 10) * 10
        elif self.C >= 1:
            self.C = round(self.C, 0)
        elif self.C > 0.2:
            self.C = round(self.C * 20) / 20
        else:
            self.C = round(self.C, 3)

        if self.C >= 1:
            frmt = "{}"
            self.C = int(self.C)
        elif self.C >= 0.2:
            frmt = "{:.2f}"
        else:
            frmt = "{:.3f}"
        self.c_label.setText(("C=" + frmt).format(self.C))

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
