# -*- coding: utf-8 -*-
from PyQt4 import QtGui
from PyQt4.QtGui import QLabel, QGridLayout
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification.random_forest import RandomForestLearner
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWRandomForest(OWBaseLearner):
    name = "Random Forest Classification"
    description = "Predict using an ensemble of decision trees."
    icon = "icons/RandomForest.svg"
    priority = 40

    LEARNER = RandomForestLearner

    n_estimators = settings.Setting(10)
    max_features = settings.Setting(5)
    use_max_features = settings.Setting(False)
    random_state = settings.Setting(0)
    use_random_state = settings.Setting(False)
    max_depth = settings.Setting(3)
    use_max_depth = settings.Setting(False)
    max_leaf_nodes = settings.Setting(5)
    use_max_leaf_nodes = settings.Setting(True)
    index_output = settings.Setting(0)

    def add_main_layout(self):
        form = QGridLayout()
        basic_box = gui.widgetBox(
            self.controlArea, "Basic Properties", orientation=form)

        form.addWidget(QLabel(self.tr("Number of trees: ")),
                       0, 0, Qt.AlignLeft)
        spin = gui.spin(basic_box, self, "n_estimators", minv=1, maxv=1e4,
                        callback=self.settings_changed, addToLayout=False,
                        controlWidth=50)
        form.addWidget(spin, 0, 1, Qt.AlignRight)

        max_features_cb = gui.checkBox(
            basic_box, self, "use_max_features",
            callback=self.settings_changed, addToLayout=False,
            label="Number of attributes considered at each split: ")

        max_features_spin = gui.spin(
            basic_box, self, "max_features", 2, 50, addToLayout=False,
            callback=self.settings_changed, controlWidth=50)

        form.addWidget(max_features_cb, 1, 0, Qt.AlignLeft)
        form.addWidget(max_features_spin, 1, 1, Qt.AlignRight)

        random_state_cb = gui.checkBox(
            basic_box, self, "use_random_state", callback=self.settings_changed,
            addToLayout=False, label="Fixed seed for random generator: ")
        random_state_spin = gui.spin(
            basic_box, self, "random_state", 0, 2 ** 31 - 1, addToLayout=False,
            callback=self.settings_changed, controlWidth=50)

        form.addWidget(random_state_cb, 2, 0, Qt.AlignLeft)
        form.addWidget(random_state_spin, 2, 1, Qt.AlignRight)
        self._max_features_spin = max_features_spin
        self._random_state_spin = random_state_spin

        # Growth control
        form = QGridLayout()
        growth_box = gui.widgetBox(
            self.controlArea, "Growth Control", orientation=form)

        max_depth_cb = gui.checkBox(
            growth_box, self, "use_max_depth",
            label="Limit depth of individual trees: ",
            callback=self.settings_changed,
            addToLayout=False)

        max_depth_spin = gui.spin(
            growth_box, self, "max_depth", 2, 50, addToLayout=False,
            callback=self.settings_changed)

        form.addWidget(max_depth_cb, 3, 0, Qt.AlignLeft)
        form.addWidget(max_depth_spin, 3, 1, Qt.AlignRight)

        max_leaf_nodes_cb = gui.checkBox(
            growth_box, self, "use_max_leaf_nodes",
            label="Do not split subsets smaller than: ",
            callback=self.settings_changed, addToLayout=False)

        max_leaf_nodes_spin = gui.spin(
            growth_box, self, "max_leaf_nodes", 0, 100, addToLayout=False,
            callback=self.settings_changed)

        form.addWidget(max_leaf_nodes_cb, 4, 0, Qt.AlignLeft)
        form.addWidget(max_leaf_nodes_spin, 4, 1, Qt.AlignRight)
        self._max_depth_spin = max_depth_spin
        self._max_leaf_nodes_spin = max_leaf_nodes_spin

        # Index on the output
        # gui.doubleSpin(self.controlArea, self, "index_output", 0, 10000, 1,
        #                label="Index of tree on the output")

    def create_learner(self):
        common_args = {"n_estimators": self.n_estimators}
        if self.use_max_features:
            common_args["max_features"] = self.max_features
        if self.use_random_state:
            common_args["random_state"] = self.random_state
        if self.use_max_depth:
            common_args["max_depth"] = self.max_depth
        if self.use_max_leaf_nodes:
            common_args["max_leaf_nodes"] = self.max_leaf_nodes

        return self.LEARNER(preprocessors=self.preprocessors, **common_args)

    def settings_changed(self):
        super().settings_changed()
        self._max_features_spin.setEnabled(self.use_max_features)
        self._random_state_spin.setEnabled(self.use_random_state)
        self._max_depth_spin.setEnabled(self.use_max_depth)
        self._max_leaf_nodes_spin.setEnabled(self.use_max_leaf_nodes)

    def get_learner_parameters(self):
        return (("Number of trees", self.n_estimators),
                ("Maximal number of considered features",
                 self.max_features if self.use_max_features else "unlimited"),
                ("Fixed random seed", self.use_random_state and self.random_state),
                ("Maximal tree depth",
                 self.max_depth if self.use_max_depth else "unlimited"),
                ("Stop splitting nodes with maximum instances",
                 self.max_leaf_nodes if self.use_max_leaf_nodes else "unlimited"))


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWRandomForest()
    w.set_data(Table("iris"))
    w.show()
    app.exec_()
