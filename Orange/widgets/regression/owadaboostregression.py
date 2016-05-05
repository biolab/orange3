from Orange.regression.base_regression import LearnerRegression
from Orange.ensembles import SklAdaBoostRegressionLearner
from Orange.widgets.classify import owadaboost


class OWAdaBoostRegression(owadaboost.OWAdaBoostClassification):
    name = "AdaBoost Regression"
    description = "An AdaBoost regression algorithm."
    icon = "icons/AdaBoost.svg"
    priority = 80

    LEARNER = SklAdaBoostRegressionLearner

    inputs = [("Learner", LearnerRegression, "set_base_learner")]


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWAdaBoostRegression()
    ow.set_data(Table("housing"))
    ow.show()
    a.exec_()
    ow.saveSettings()
