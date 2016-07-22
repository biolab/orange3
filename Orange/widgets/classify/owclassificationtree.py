from collections import OrderedDict

from Orange.data import Table
from Orange.classification.tree import TreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWClassificationTree(OWBaseLearner):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    description = "Classification tree algorithm with forward pruning."
    priority = 30

    LEARNER = TreeLearner

    attribute_score = Setting(0)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)

    scores = (("Entropy", "entropy"), ("Gini Index", "gini"))

    def add_main_layout(self):
        self.score_combo = gui.comboBox(
            self.controlArea, self, "attribute_score", box='Feature Selection',
            items=[name for name, _ in self.scores],
            callback=self.settings_changed)

        box = gui.vBox(self.controlArea, 'Pruning')
        self.min_leaf_spin = gui.spin(
            box, self, "min_leaf", 1, 1000, label="Min. instances in leaves: ",
            checked="limit_min_leaf", callback=self.settings_changed,
            checkCallback=self.settings_changed)
        self.min_internal_spin = gui.spin(
            box, self, "min_internal", 1, 1000,
            label="Stop splitting nodes with less instances than: ",
            checked="limit_min_internal", callback=self.settings_changed,
            checkCallback=self.settings_changed)
        self.max_depth_spin = gui.spin(
            box, self, "max_depth", 1, 1000, label="Limit the depth to: ",
            checked="limit_depth", callback=self.settings_changed,
            checkCallback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            criterion=self.scores[self.attribute_score][1],
            max_depth=self.max_depth if self.limit_depth else None,
            min_samples_split=(self.min_internal if self.limit_min_internal
                               else 2),
            min_samples_leaf=(self.min_leaf if self.limit_min_leaf else 1),
            preprocessors=self.preprocessors
        )

    def get_learner_parameters(self):
        from Orange.canvas.report import plural_w
        items = OrderedDict()
        items["Split selection"] = self.scores[self.attribute_score][0]
        items["Pruning"] = ", ".join(s for s, c in (
            (plural_w("at least {number} instance{s} in leaves", self.min_leaf),
             self.limit_min_leaf),
            (plural_w("at least {number} instance{s} in internal nodes", self.min_internal),
             self.limit_min_internal),
            ("maximum depth {}".format(self.max_depth), self.limit_depth)) if c) or "None"
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
