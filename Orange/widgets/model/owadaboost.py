from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout, QLabel

from Orange.base import Learner
from Orange.data import Table
from Orange.modelling import SklAdaBoostLearner, SklTreeLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input


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
    keywords = "adaboost, boost"

    LEARNER = SklAdaBoostLearner

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    #: Losses for regression problems
    losses = ["Linear", "Square", "Exponential"]

    n_estimators = Setting(50)
    learning_rate = Setting(1.)
    loss_index = Setting(0)
    use_random_seed = Setting(False)
    random_seed = Setting(0)

    DEFAULT_BASE_ESTIMATOR = SklTreeLearner()

    class Error(OWBaseLearner.Error):
        no_weight_support = Msg('The base learner does not support weights.')

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        grid = QFormLayout()
        gui.widgetBox(self.controlArea, box=True, orientation=grid)
        self.base_estimator = self.DEFAULT_BASE_ESTIMATOR
        self.base_label = QLabel(self.base_estimator.name.title())
        grid.addRow("Base estimator:", self.base_label)

        self.n_estimators_spin = gui.spin(
            None, self, "n_estimators", 1, 10000,
            controlWidth=80, alignment=Qt.AlignRight,
            callback=self.settings_changed)
        grid.addRow("Number of estimators:", self.n_estimators_spin)

        self.learning_rate_spin = gui.doubleSpin(
            None, self, "learning_rate", 1e-5, 1.0, 1e-5, decimals=5,
            alignment=Qt.AlignRight,
            callback=self.settings_changed)
        grid.addRow("Learning rate:", self.learning_rate_spin)

        self.reg_algorithm_combo = gui.comboBox(
            None, self, "loss_index", items=self.losses,
            callback=self.settings_changed)
        grid.addRow("Loss (regression):", self.reg_algorithm_combo)

        box = gui.widgetBox(self.controlArea, box="Reproducibility")
        self.random_seed_spin = gui.spin(
            box, self, "random_seed", 0, 2 ** 31 - 1, controlWidth=80,
            label="Fixed seed for random generator:", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_random_seed",
            checkCallback=self.settings_changed)

    def create_learner(self):
        if self.base_estimator is None:
            return None
        return self.LEARNER(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
            preprocessors=self.preprocessors,
            loss=self.losses[self.loss_index].lower())

    @Inputs.learner
    def set_base_learner(self, learner):
        # base_estimator is defined in add_main_layout
        # pylint: disable=attribute-defined-outside-init
        self.Error.no_weight_support.clear()
        if learner and not learner.supports_weights:
            # Clear the error and reset to default base learner
            self.Error.no_weight_support()
            self.base_estimator = None
            self.base_label.setText("INVALID")
        else:
            self.base_estimator = learner or self.DEFAULT_BASE_ESTIMATOR
            self.base_label.setText(self.base_estimator.name.title())
        self.learner = self.model = None

    def get_learner_parameters(self):
        return (("Base estimator", self.base_estimator),
                ("Number of estimators", self.n_estimators),
                ("Loss (regression)", self.losses[
                    self.loss_index].capitalize()))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWAdaBoost).run(Table("iris"))
