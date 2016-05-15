from Orange.classification.majority import MajorityLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWMajority(OWBaseLearner):
    name = "Majority"
    description = "Classification to the most frequent class " \
                  "from the training set."
    icon = "icons/Majority.svg"
    priority = 10

    LEARNER = MajorityLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWMajority()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
