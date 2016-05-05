from PyQt4.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.regression.linear import (
    LassoRegressionLearner, LinearRegressionLearner,
    RidgeRegressionLearner, ElasticNetLearner)
from Orange.widgets.utils.owlearnerwidget import OWBaseMultipleLearner


class OWLinearRegression(OWBaseMultipleLearner):
    name = "Linear Regression"
    description = "A linear regression algorithm with optional L1 and L2 " \
                  "regularization."
    icon = "icons/LinearRegression.svg"
    priority = 60

    LEARNER = LinearRegressionLearner

    outputs = [("Coefficients", Table)]

    Learners = [LinearRegressionLearner(), RidgeRegressionLearner(),
                LassoRegressionLearner(), ElasticNetLearner()]

    ORIENTATION = Qt.Horizontal

    def update_model(self):
        super().update_model()
        coef_table = None
        if self.valid_data:
            domain = Domain(
                    [ContinuousVariable("coef", number_of_decimals=7)],
                    metas=[StringVariable("name")])
            coefs = [self.model.intercept] + list(self.model.coefficients)
            names = ["intercept"] + \
                    [attr.name for attr in self.model.domain.attributes]
            coef_table = Table(domain, list(zip(coefs, names)))
            coef_table.name = "coefficients"
        self.send("Coefficients", coef_table)


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWLinearRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
