from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.widgets.classify import owrandomforest
from Orange.widgets import settings


class OWRandomForestRegression(owrandomforest.OWRandomForest):
    name = "Random Forest Regression"
    description = "Predict using an ensemble of regression trees."
    icon = "icons/RandomForest.svg"
    priority = 40

    LEARNER = RandomForestRegressionLearner


if __name__ == "__main__":
    from PyQt4 import QtGui
    from Orange.data import Table

    app = QtGui.QApplication([])
    w = OWRandomForestRegression()
    w.set_data(Table("housing"))
    w.show()
    app.exec_()
