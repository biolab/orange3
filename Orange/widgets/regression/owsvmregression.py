from Orange.data import Table
from Orange.regression import SVRLearner, NuSVRLearner
from Orange.widgets.classify.owsvmclassification import OWBaseSVM


class OWSVMRegression(OWBaseSVM):
    name = "SVM Regression"
    description = "Support vector machine regression algorithm."
    icon = "icons/SVMRegression.svg"
    priority = 50

    LEARNER = SVRLearner
    Learners = [SVRLearner(), NuSVRLearner()]

    outputs = [("Support vectors", Table)]


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWSVMRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
