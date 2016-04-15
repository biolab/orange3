from Orange.regression.mean import MeanLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWMean(OWBaseLearner):
    name = "Mean Learner"
    description = "Regression to the average class value from the training set."
    icon = "icons/Mean.svg"
    priority = 10

    LEARNER = MeanLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWMean()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
