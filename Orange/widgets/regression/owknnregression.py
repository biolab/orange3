from Orange.regression.knn import KNNRegressionLearner
from Orange.widgets.classify import owknn
from Orange.widgets.settings import Setting


class OWKNNRegression(owknn.OWKNNLearner):
    name = "Nearest Neighbors"
    description = "k-nearest neighbours regression algorithm."
    icon = "icons/kNearestNeighbours.svg"
    priority = 20

    LEARNER = KNNRegressionLearner
    OUTPUT_MODEL_NAME = "Predictor"

    learner_name = Setting("kNN Regression")


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWKNNRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
