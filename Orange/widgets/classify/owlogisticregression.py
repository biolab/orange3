import numpy as np
from PyQt4 import QtGui

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.classification.logistic_regression import LogisticRegressionLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWLogisticRegression(OWBaseLearner):
    name = "Logistic Regression"
    description = "Logistic regression classification algorithm with " \
                  "LASSO (L1) or ridge (L2) regularization."
    icon = "icons/LogisticRegression.svg"
    priority = 60

    LEARNER = LogisticRegressionLearner

    outputs = [("Coefficients", Table)]

    def update_model(self):
        super().update_model()
        coef_table = None
        if self.valid_data:
            coef_table = create_coef_table(self.model)
        self.send("Coefficients", coef_table)


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
