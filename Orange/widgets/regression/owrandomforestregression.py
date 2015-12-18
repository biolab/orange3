from PyQt4 import QtGui
from Orange.data import Table
from Orange.regression.random_forest import (RandomForestRegressionLearner,
                                             RandomForestRegressor)
from Orange.widgets.classify.owrandomforest import OWRandomForest
from Orange.widgets import settings


class OWRandomForestRegression(OWRandomForest):
    name = "Random Forest Regression"
    description = "Random forest regression algorithm."

    LEARNER = RandomForestRegressionLearner
    outputs = [("Learner", LEARNER),
               ("Model", RandomForestRegressor)]

    learner_name = settings.Setting("RF Regression Learner")
    n_estimators = settings.Setting(10)
    max_features = settings.Setting(5)
    use_max_features = settings.Setting(False)
    random_state = settings.Setting(0)
    use_random_state = settings.Setting(False)
    max_depth = settings.Setting(3)
    use_max_depth = settings.Setting(False)
    max_leaf_nodes = settings.Setting(5)
    use_max_leaf_nodes = settings.Setting(True)
    index_output = settings.Setting(0)


del OWRandomForest

if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWRandomForestRegression()
    w.set_data(Table("housing"))
    w.show()
    app.exec_()
