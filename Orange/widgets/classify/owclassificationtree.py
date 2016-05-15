from Orange.classification.tree import TreeLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWClassificationTree(OWBaseLearner):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    description = "Classification tree algorithm with forward pruning."
    priority = 30

    LEARNER = TreeLearner


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWClassificationTree()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
