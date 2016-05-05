from Orange.classification import KNNLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWKNNLearner(OWBaseLearner):
    name = "Nearest Neighbors"
    description = "k-nearest neighbors classification algorithm."
    icon = "icons/KNN.svg"
    priority = 20

    LEARNER = KNNLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWKNNLearner()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
