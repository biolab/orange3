# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4.QtGui import QLabel, QGridLayout, QLayout
from PyQt4.QtCore import Qt

import Orange
from Orange.preprocess.preprocess import Preprocess
from Orange.classification import random_forest
from Orange.widgets import widget, settings, gui


class OWRandomForest(widget.OWWidget):
    name = "Random Forest"
    description = "Random forest classication algorithm."
    icon = "icons/RandomForest.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", random_forest.RandomForestLearner),
               ("Classifier", random_forest.RandomForestClassifier)]

    want_main_area = False

    learner_name = settings.Setting("Random Forest Learner")
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

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        # Learner name
        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        # Basic properties
        form = QGridLayout()
        basic_box = gui.widgetBox(
            self.controlArea, "Basic properties", orientation=form)

        form.addWidget(QLabel(self.tr("Number of trees in the forest: ")),
                       0, 0, Qt.AlignLeft)
        spin = gui.spin(basic_box, self, "n_estimators", minv=1, maxv=1e4,
                        callback=self.settingsChanged, addToLayout=False,
                        controlWidth=50)
        form.addWidget(spin, 0, 1, Qt.AlignRight)

        max_features_cb = gui.checkBox(
            basic_box, self, "use_max_features",
            callback=self.settingsChanged, addToLayout=False,
            label="Consider a number of best attributes at each split")

        max_features_spin = gui.spin(
            basic_box, self, "max_features", 2, 50, addToLayout=False,
            callback=self.settingsChanged, controlWidth=50)

        form.addWidget(max_features_cb, 1, 0, Qt.AlignLeft)
        form.addWidget(max_features_spin, 1, 1, Qt.AlignRight)

        random_state_cb = gui.checkBox(
            basic_box, self, "use_random_state", callback=self.settingsChanged,
            addToLayout=False, label="Use seed for random generator:")
        random_state_spin = gui.spin(
            basic_box, self, "random_state", 0, 2 ** 31 - 1, addToLayout=False,
            callback=self.settingsChanged, controlWidth=50)

        form.addWidget(random_state_cb, 2, 0, Qt.AlignLeft)
        form.addWidget(random_state_spin, 2, 1, Qt.AlignRight)
        self._max_features_spin = max_features_spin
        self._random_state_spin = random_state_spin

        # Growth control
        form = QGridLayout()
        growth_box = gui.widgetBox(
            self.controlArea, "Growth control", orientation=form)

        max_depth_cb = gui.checkBox(
            growth_box, self, "use_max_depth",
            label="Set maximal depth of individual trees",
            callback=self.settingsChanged,
            addToLayout=False)

        max_depth_spin = gui.spin(
            growth_box, self, "max_depth", 2, 50, addToLayout=False,
            callback=self.settingsChanged)

        form.addWidget(max_depth_cb, 3, 0, Qt.AlignLeft)
        form.addWidget(max_depth_spin, 3, 1, Qt.AlignRight)

        max_leaf_nodes_cb = gui.checkBox(
            growth_box, self, "use_max_leaf_nodes",
            label="Stop splitting nodes with maximum instances: ",
            callback=self.settingsChanged, addToLayout=False)

        max_leaf_nodes_spin = gui.spin(
            growth_box, self, "max_leaf_nodes", 0, 100, addToLayout=False,
            callback=self.settingsChanged)

        form.addWidget(max_leaf_nodes_cb, 4, 0, Qt.AlignLeft)
        form.addWidget(max_leaf_nodes_spin, 4, 1, Qt.AlignRight)
        self._max_depth_spin = max_depth_spin
        self._max_leaf_nodes_spin = max_leaf_nodes_spin

        # Index on the output
#         gui.doubleSpin(self.controlArea, self, "index_output", 0, 10000, 1,
#                        label="Index of tree on the output")

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self.setSizePolicy(
            QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                              QtGui.QSizePolicy.Fixed)
        )
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.settingsChanged()
        self.apply()

    def set_data(self, data):
        """Set the input train data set."""
        self.warning(0)
        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        common_args = dict()
        common_args["n_estimators"] = self.n_estimators
        if self.use_max_features:
            common_args["max_features"] = self.max_features
        if self.use_random_state:
            common_args["random_state"] = self.random_state
        if self.use_max_depth:
            common_args["max_depth"] = self.max_depth
        if self.use_max_leaf_nodes:
            common_args["max_leaf_nodes"] = self.max_leaf_nodes

        learner = Orange.classification.RandomForestLearner(
            preprocessors=self.preprocessors, **common_args)

        learner.name = self.learner_name
        classifier = None
        if self.data is not None:
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def settingsChanged(self):
        self._max_features_spin.setEnabled(self.use_max_features)
        self._random_state_spin.setEnabled(self.use_random_state)
        self._max_depth_spin.setEnabled(self.use_max_depth)
        self._max_leaf_nodes_spin.setEnabled(self.use_max_leaf_nodes)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWRandomForest()
    w.set_data(Orange.data.Table("iris"))
    w.show()
    app.exec_()
