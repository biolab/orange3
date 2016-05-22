from Orange.widgets.classify import owknn
from Orange.regression.knn import KNNRegressionLearner


class OWKNNRegression(owknn.OWKNNLearner):
    name = "Nearest Neighbors"
    description = "Predict according to the nearest training instances."
    icon = "icons/kNearestNeighbours.svg"
    priority = 20

    LEARNER = KNNRegressionLearner


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
