from AnyQt.QtCore import Qt

from Orange.base import Learner
from Orange.data import Table
from Orange.modelling import SklAdaBoostLearner, SklTreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Msg


class OWAdaBoost(OWBaseLearner):
    name = "AdaBoost"
    description = "An ensemble meta-algorithm that combines weak learners " \
                  "and adapts to the 'hardness' of each training sample. "
    icon = "icons/AdaBoost.svg"
    replaces = [
        "Orange.widgets.classify.owadaboost.OWAdaBoostClassification",
        "Orange.widgets.regression.owadaboostregression.OWAdaBoostRegression",
    ]
    priority = 80

    LEARNER = SklAdaBoostLearner

    inputs = [("Learner", Learner, "set_base_learner")]

    #: Algorithms for classification problems
    algorithms = ["SAMME", "SAMME.R"]
    #: Losses for regression problems
    losses = ["Linear", "Square", "Exponential"]

    n_estimators = Setting(50)
    learning_rate = Setting(1.)
    algorithm_index = Setting(1)
    loss_index = Setting(0)
    use_random_seed = Setting(False)
    random_seed = Setting(0)

    DEFAULT_BASE_ESTIMATOR = SklTreeLearner()

    class Error(OWBaseLearner.Error):
        no_weight_support = Msg('The base learner does not support weights.')

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = self.DEFAULT_BASE_ESTIMATOR
        self.base_label = gui.label(
            box, self, "Base estimator: " + self.base_estimator.name.title())

        self.n_estimators_spin = gui.spin(
            box, self, "n_estimators", 1, 100, label="Number of estimators:",
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.learning_rate_spin = gui.doubleSpin(
            box, self, "learning_rate", 1e-5, 1.0, 1e-5,
            label="Learning rate:", decimals=5, alignment=Qt.AlignRight,
            controlWidth=80, callback=self.settings_changed)
        self.random_seed_spin = gui.spin(
            box, self, "random_seed", 0, 2 ** 31 - 1, controlWidth=80,
            label="Fixed seed for random generator:", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_random_seed",
            checkCallback=self.settings_changed)

        # Algorithms
        box = gui.widgetBox(self.controlArea, "Boosting method")
        self.cls_algorithm_combo = gui.comboBox(
            box, self, "algorithm_index", label="Classification algorithm:",
            items=self.algorithms,
            orientation=Qt.Horizontal, callback=self.settings_changed)
        self.reg_algorithm_combo = gui.comboBox(
            box, self, "loss_index", label="Regression loss function:",
            items=self.losses,
            orientation=Qt.Horizontal, callback=self.settings_changed)

    def create_learner(self):
        if self.base_estimator is None:
            return None
        return self.LEARNER(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
            preprocessors=self.preprocessors,
            algorithm=self.algorithms[self.algorithm_index],
            loss=self.losses[self.loss_index].lower())

    def set_base_learner(self, learner):
        self.Error.no_weight_support.clear()
        if learner and not learner.supports_weights:
            # Clear the error and reset to default base learner
            self.Error.no_weight_support()
            self.base_estimator = None
            self.base_label.setText("Base estimator: INVALID")
        else:
            self.base_estimator = learner or self.DEFAULT_BASE_ESTIMATOR
            self.base_label.setText(
                "Base estimator: %s" % self.base_estimator.name.title())
        if self.auto_apply:
            self.apply()

    def get_learner_parameters(self):
        return (("Base estimator", self.base_estimator),
                ("Number of estimators", self.n_estimators),
                ("Algorithm (classification)", self.algorithms[
                    self.algorithm_index].capitalize()),
                ("Loss (regression)", self.losses[
                    self.loss_index].capitalize()))


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWAdaBoost()
    ow.resetSettings()
    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    a.exec_()
    ow.saveSettings()
