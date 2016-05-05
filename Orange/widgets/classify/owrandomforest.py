# -*- coding: utf-8 -*-
from Orange.classification.random_forest import RandomForestLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWRandomForest(OWBaseLearner):
    name = "Random Forest Classification"
    description = "Random forest classification algorithm."
    icon = "icons/RandomForest.svg"
    priority = 40

    LEARNER = RandomForestLearner


if __name__ == "__main__":
    from PyQt4 import QtGui
    from Orange.data import Table
    app = QtGui.QApplication([])
    w = OWRandomForest()
    w.set_data(Table("iris"))
    w.show()
    app.exec_()
