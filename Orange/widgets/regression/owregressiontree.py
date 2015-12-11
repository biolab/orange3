from Orange.regression.tree import TreeRegressionLearner, TreeRegressor
from Orange.widgets.settings import Setting
from Orange.widgets.classify.owclassificationtree import OWClassificationTree


class OWRegressionTree(OWClassificationTree):
    name = "Regression Tree"
    icon = "icons/RegressionTree.svg"
    description = "Regression tree algorithm with forward pruning."

    LEARNER = TreeRegressionLearner

    outputs = [("Learner", LEARNER),
               ("Tree", TreeRegressor)]

    model_name = Setting("Regression Tree")
    attribute_score = Setting(0)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)

    scores = (("Mean Squared Error", "mse"),)


del OWClassificationTree

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
