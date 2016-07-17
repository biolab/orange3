from collections import OrderedDict

from PyQt4.QtCore import Qt

from Orange.base import Tree
from Orange.data import Table
from Orange.classification.tree import TreeLearner, OrangeTreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWClassificationTree(OWBaseLearner):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    description = "Classification tree algorithm with forward pruning."
    priority = 30

    outputs = [("Classifier", Tree)]
    # TODO: Common base class for tree learners
    LEARNER = TreeLearner

    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)
    limit_majority = Setting(True)
    sufficient_majority = Setting(95)
    binary_trees = Setting(True)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, True)
        # the checkbox is put into vBox for alignemnt with other checkboxes
        gui.checkBox(gui.vBox(box), self, "binary_trees", "Induce binary tree",
                     callback=self.settings_changed)
        for label, check, setting, fromv, tov in (
                ("Min. number of instances in leaaves: ",
                 "limit_min_leaf", "min_leaf", 1, 1000),
                ("Do not split subsets smaller than: ",
                 "limit_min_internal", "min_internal", 1, 1000),
                ("Stop when majority reaches [%]: ",
                 "limit_majority", "sufficient_majority", 51, 100),
                ("Limit the maximal tree depth to: ",
                 "limit_depth", "max_depth", 1, 1000)):
            gui.spin(box, self, setting, fromv, tov, label=label, checked=check,
                     alignment=Qt.AlignRight, callback=self.settings_changed,
                     checkCallback=self.settings_changed)

    def create_learner(self):
        return OrangeTreeLearner(
            max_depth=(None, self.max_depth)[self.limit_depth],
            min_samples_split=(2, self.min_internal)[self.limit_min_internal],
            min_samples_leaf=(1, self.min_leaf)[self.limit_min_leaf],
            sufficient_majority=(1, self.sufficient_majority / 100
                                 )[self.limit_majority],
            binarize=self.binary_trees
        )

    def get_learner_parameters(self):
        from Orange.canvas.report import plural_w
        items = OrderedDict()
        items["Pruning"] = ", ".join(s for s, c in (
            (plural_w("at least {number} instance{s} in leaves",
                      self.min_leaf), self.limit_min_leaf),
            (plural_w("at least {number} instance{s} in internal nodes",
                      self.min_internal), self.limit_min_internal),
            ("stop splitting when the majority class reaches {} %".
             format(self.sufficient_majority), self.limit_majority),
            ("maximum depth {}".format(self.max_depth), self.limit_depth)
            ) if c) or "None"
        items["Binary trees"] = ("No", "Yes")[self.binary_trees]
        return items

if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWClassificationTree()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
