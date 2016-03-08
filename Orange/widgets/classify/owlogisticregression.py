import numpy as np
from itertools import chain
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.classification import logistic_regression as lr
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWLogisticRegression(OWBaseLearner):
    name = "Logistic Regression"
    description = "Logistic regression classification algorithm with " \
                  "LASSO (L1) or ridge (L2) regularization."
    icon = "icons/LogisticRegression.svg"
    priority = 60

    LEARNER = lr.LogisticRegressionLearner
    OUTPUT_MODEL_NAME = "Classifier"

    outputs = [("Coefficients", Table)]

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

    def add_main_layout(self):
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
        self.set_c()

    def set_c(self):
        self.C = self.C_s[self.C_index]
        if self.C >= 1:
            frmt = "C={}"
        else:
            frmt = "C={:.3f}"
        self.c_label.setText(frmt.format(self.C))

    def create_learner(self):
        penalty = ["l1", "l2"][self.penalty_type]
        return self.LEARNER(
            penalty=penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            preprocessors=self.preprocessors
        )

    def update_model(self):
        super().update_model()
        coef_table = None
        if self.good_data:
            coef_table = create_coef_table(self.model)
        self.send("Coefficients", coef_table)

    def get_model_parameters(self):
        return (("Regularization", "{}, C={}".format(
                self.penalty_types[self.penalty_type], self.C_s[self.C_index])),)


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
