import numpy as np
from itertools import chain
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.classification import logistic_regression as lr
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner
from Orange.widgets.utils.sql import check_sql_input


class OWLogisticRegression(OWProvidesLearner, widget.OWWidget):
    name = "Logistic Regression"
    description = "Logistic regression classification algorithm with " \
                  "LASSO (L1) or ridge (L2) regularization."
    icon = "icons/LogisticRegression.svg"

    inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
    outputs = [("Learner", lr.LogisticRegressionLearner),
               ("Classifier", lr.LogisticRegressionClassifier),
               ("Coefficients", Table)]

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

    penalty_types = ("Lasso (L1)", "Ridge (L2)")

    def __init__(self):
        super().__init__()

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, box=True)
        gui.comboBox(box, self, "penalty_type", label="Regularization type: ",
                     items=self.penalty_types,
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
        box = gui.widgetBox(self.controlArea, orientation="horizontal",
                            margin=0)
        box.layout().addWidget(self.report_button)
        gui.button(box, self, "&Apply", callback=self.apply, default=True)
        self.set_c()
        self.apply()

    def set_c(self):
        self.C = self.C_s[self.C_index]
        if self.C >= 1:
            frmt = "C={}"
        else:
            frmt = "C={:.3f}"
        self.c_label.setText(frmt.format(self.C))

    @check_sql_input
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    LEARNER = lr.LogisticRegressionLearner

    def apply(self):
        penalty = ["l1", "l2"][self.penalty_type]
        learner = self.LEARNER(
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
        coef_table = None

        if self.data is not None:
            self.error([0, 1])
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            elif len(np.unique(self.data.Y)) < 2:
                self.error(1, "Data contains only one target value.")
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name
                coef_table = create_coef_table(classifier)

        self.send("Learner", learner)
        self.send("Classifier", classifier)
        self.send("Coefficients", coef_table)

    def send_report(self):
        self.report_items((("Name", self.learner_name),))
        self.report_items("Model parameters", (
            ("Regularization", "{}, C={}".format(
                self.penalty_types[self.penalty_type], self.C_s[self.C_index])),
        ))
        if self.data:
            self.report_data("Data", self.data)


def create_coef_table(classifier):
    i = classifier.intercept
    c = classifier.coefficients
    if len(classifier.domain.class_var.values) > 2:
        values = classifier.domain.class_var.values
    else:
        values = ["coef"]
    domain = Domain([ContinuousVariable(value, number_of_decimals=7)
                     for value in values], metas=[StringVariable("name")])
    coefs = np.vstack((i.reshape(1, len(i)), c.T))
    names = [[attr.name] for attr in classifier.domain.attributes]
    names = [["intercept"]] + names
    names = np.array(names, dtype=object)
    coef_table = Table.from_numpy(domain, X=coefs, metas=names)
    coef_table.name = "coefficients"
    return coef_table


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWLogisticRegression()
    w.set_data(Table("zoo"))
    w.show()
    app.exec_()
