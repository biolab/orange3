from collections import OrderedDict

from PyQt4.QtCore import Qt

from Orange.tree import Tree
from Orange.data import Table
from Orange.classification.tree import OrangeTreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWTreeLearner(OWBaseLearner):
    binary_trees = Setting(True)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)

    spin_boxes = (
        ("Min. number of instances in leaaves: ",
         "limit_min_leaf", "min_leaf", 1, 1000),
        ("Do not split subsets smaller than: ",
         "limit_min_internal", "min_internal", 1, 1000),
        ("Limit the maximal tree depth to: ",
         "limit_depth", "max_depth", 1, 1000))

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, True)
        # the checkbox is put into vBox for alignemnt with other checkboxes
        gui.checkBox(gui.vBox(box), self, "binary_trees", "Induce binary tree",
                     callback=self.settings_changed)
        for label, check, setting, fromv, tov in self.spin_boxes:
            gui.spin(box, self, setting, fromv, tov, label=label, checked=check,
                     alignment=Qt.AlignRight, callback=self.settings_changed)

    def learner_kwargs(self):
        return dict(
            max_depth=(None, self.max_depth)[self.limit_depth],
            min_samples_split=(2, self.min_internal)[self.limit_min_internal],
            min_samples_leaf=(1, self.min_leaf)[self.limit_min_leaf],
            binarize=self.binary_trees)

    def create_learner(self):
        return self.LEARNER(**self.learner_kwargs())

    def get_learner_parameters(self):
        from Orange.canvas.report import plural_w
        items = OrderedDict()
        items["Pruning"] = ", ".join(s for s, c in (
            (plural_w("at least {number} instance{s} in leaves",
                      self.min_leaf), self.limit_min_leaf),
            (plural_w("at least {number} instance{s} in internal nodes",
                      self.min_internal), self.limit_min_internal),
            ("maximum depth {}".format(self.max_depth), self.limit_depth)
        ) if c) or "None"
        items["Binary trees"] = ("No", "Yes")[self.binary_trees]
        return items


class OWClassificationTree(OWTreeLearner):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    description = "Classification tree algorithm with forward pruning."
    priority = 30

    LEARNER = OrangeTreeLearner

    limit_majority = Setting(True)
    sufficient_majority = Setting(95)

    spin_boxes = \
        OWTreeLearner.spin_boxes[:-1] + \
        (("Stop when majority reaches [%]: ",
          "limit_majority", "sufficient_majority", 51, 100),) + \
        OWTreeLearner.spin_boxes[-1:]

    def learner_opts(self):
        opts = super().learner_kwargs()
        opts['sufficient_majority'] = \
            (1, self.sufficient_majority / 100)[self.limit_majority]

    def get_learner_parameters(self):
        items = super().get_learner_parameters()
        if self.limit_majority:
            items["Pruning"] = ", " * bool(items["Pruning"]) + \
                "stop splitting when the majority class reaches {} %".format(
                self.sufficient_majority)

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
