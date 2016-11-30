from AnyQt.QtCore import Qt

from Orange.classification import SklTreeLearner
from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table
from Orange.ensembles import SklAdaBoostLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Msg


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

    DEFAULT_BASE_ESTIMATOR = SklTreeLearner()

    class Error(OWBaseLearner.Error):
        no_weight_support = Msg('The base learner does not support weights.')

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = self.DEFAULT_BASE_ESTIMATOR
        self.base_label = gui.label(
            box, self, "Base estimator: " + self.base_estimator.name)

        self.n_estimators_spin = gui.spin(
            box, self, "n_estimators", 1, 100, label="Number of estimators:",
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.learning_rate_spin = gui.doubleSpin(
            box, self, "learning_rate", 1e-5, 1.0, 1e-5, label="Learning rate:",
            decimals=5, alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.add_specific_parameters(box)

    def add_specific_parameters(self, box):
        self.algorithm_combo = gui.comboBox(
            box, self, "algorithm", label="Algorithm:", items=self.losses,
            orientation=Qt.Horizontal, callback=self.settings_changed)

    def create_learner(self):
        if self.base_estimator is None:
            return None
        return self.LEARNER(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            preprocessors=self.preprocessors,
            algorithm=self.losses[self.algorithm])

    def set_base_learner(self, learner):
        self.Error.no_weight_support.clear()
        if learner and not learner.supports_weights:
            # Clear the error and reset to default base learner
            self.Error.no_weight_support()
            self.base_estimator = None
            self.base_label.setText("Base estimator: INVALID")
        else:
            self.base_estimator = learner or self.DEFAULT_BASE_ESTIMATOR
            self.base_label.setText("Base estimator: " + self.base_estimator.name)
        if self.auto_apply:
            self.apply()

    def get_learner_parameters(self):
        return (("Base estimator", self.base_estimator),
                ("Number of estimators", self.n_estimators),
                ("Algorithm", self.losses[self.algorithm].capitalize()))


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWAdaBoostClassification()
    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    a.exec_()
    ow.saveSettings()
