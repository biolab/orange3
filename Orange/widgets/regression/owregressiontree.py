from Orange.regression.tree import OrangeTreeLearner
from Orange.widgets.classify.owclassificationtree import OWTreeLearner


class OWRegressionTree(OWTreeLearner):
    name = "Regression Tree"
    description = "A regression tree algorithm with forward pruning."
    icon = "icons/RegressionTree.svg"
    priority = 30

    LEARNER = OrangeTreeLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWRegressionTree()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
