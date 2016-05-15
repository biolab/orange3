# -*- coding: utf-8 -*-
from Orange.regression.linear import SGDRegressionLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWSGDRegression(OWBaseLearner):
    name = "Stochastic Gradient Descent"
    description = "Stochastic gradient descent algorithm for regression."
    icon = "icons/SGDRegression.svg"
    priority = 90
    LEARNER = SGDRegressionLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWSGDRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
