"""Naive Bayes Learner
"""
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWNaiveBayes(OWBaseLearner):
    name = "Naive Bayes"
    description = "Naive Bayesian classifier."
    icon = "icons/NaiveBayes.svg"
    priority = 70

    LEARNER = NaiveBayesLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWNaiveBayes()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
