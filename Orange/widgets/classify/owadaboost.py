from PyQt4.QtCore import Qt

from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table
from Orange.classification import TreeLearner
from Orange.ensembles import SklAdaBoostLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWAdaBoostClassification(OWBaseLearner):
    name = "AdaBoost"
    description = "An ensemble meta-algorithm that combines weak learners " \
                  "and adapts to the 'hardness' of each training sample. "
    icon = "icons/AdaBoost.svg"
    priority = 80

    LEARNER = SklAdaBoostLearner

    inputs = [("Learner", LearnerClassification, "set_base_learner")]

    losses = ["SAMME", "SAMME.R"]

    n_estimators = Setting(50)
    learning_rate = Setting(1.)
    algorithm = Setting(0)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = TreeLearner()
        self.base_label = gui.label(box, self, "Base estimator: " + self.base_estimator.name)

        gui.spin(box, self, "n_estimators", 1, 100, label="Number of estimators:",
                 alignment=Qt.AlignRight, callback=self.settings_changed)
        gui.doubleSpin(box, self, "learning_rate", 1e-5, 1.0, 1e-5,
                       label="Learning rate:", decimals=5, alignment=Qt.AlignRight,
                       controlWidth=90, callback=self.settings_changed)
        self.add_specific_parameters(box)

    def add_specific_parameters(self, box):
        gui.comboBox(box, self, "algorithm", label="Algorithm:",
                     orientation=Qt.Horizontal, items=self.losses,
                     callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            preprocessors=self.preprocessors
        )

    def set_base_learner(self, model):
        self.base_estimator = model
        if self.base_estimator:
            self.base_label.setText("Base estimator: " + self.base_estimator.name)
            self.apply_button.setDisabled(False)
        else:
            self.base_label.setText("No base estimator")
            self.apply_button.setDisabled(True)

    def get_learner_parameters(self):
        return (("Base estimator", self.base_estimator),
                ("Number of estimators", self.n_estimators),
                ("Algorithm", self.losses[self.algorithm].capitalize()))


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    a = QApplication(sys.argv)
    ow = OWAdaBoostClassification()
    ow.set_data(Table("iris"))
    ow.show()
    a.exec_()
    ow.saveSettings()
