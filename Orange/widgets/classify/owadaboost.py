from AnyQt.QtCore import Qt

from Orange.base import Learner
from Orange.data import Table
from Orange.ensembles import SklAdaBoostLearner
from Orange.widgets import gui
from Orange.widgets.model.owadaboost import OWAdaBoost


class OWAdaBoost(OWAdaBoost):
    name = "AdaBoost Classification"

    LEARNER = SklAdaBoostLearner

    inputs = [("Learner", Learner, "set_base_learner")]

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = self.DEFAULT_BASE_ESTIMATOR
        self.base_label = gui.label(
            box, self, "Base estimator: " + self.base_estimator.name.title())

        self.n_estimators_spin = gui.spin(
            box, self, "n_estimators", 1, 100, label="Number of estimators:",
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.random_seed_spin = gui.spin(
            box, self, "random_seed", 0, 2 ** 31 - 1, controlWidth=80,
            label="Fixed seed for random generator:", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_random_seed",
            checkCallback=self.settings_changed)
        self.learning_rate_spin = gui.doubleSpin(
            box, self, "learning_rate", 1e-5, 1.0, 1e-5,
            label="Learning rate:", decimals=5, alignment=Qt.AlignRight,
            controlWidth=80, callback=self.settings_changed)
        self.cls_algorithm_combo = gui.comboBox(
            box, self, "algorithm_index", label="Algorithm:",
            items=self.algorithms,
            orientation=Qt.Horizontal, callback=self.settings_changed)


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWAdaBoost()
    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    a.exec_()
    ow.saveSettings()
