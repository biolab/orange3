from PyQt4.QtCore import Qt

from Orange.regression.base_regression import LearnerRegression
from Orange.data import Table
from Orange.regression import TreeRegressionLearner
from Orange.ensembles import SklAdaBoostRegressionLearner
from Orange.widgets import gui
from Orange.widgets.classify import owadaboost
from Orange.widgets.settings import Setting


class OWAdaBoostRegression(owadaboost.OWAdaBoostClassification):
    name = "AdaBoost"
    description = "An AdaBoost regression algorithm."
    icon = "icons/AdaBoost.svg"
    priority = 80

    LEARNER = SklAdaBoostRegressionLearner
    OUTPUT_MODEL_NAME = "Model"

    inputs = [("Model", LearnerRegression, "set_base_model")]
    want_main_area = False
    resizing_enabled = False

    losses = ["linear", "square", "exponential"]

    learner_name = Setting("AdaBoost Regression")
    n_estimators = Setting(50)
    learning_rate = Setting(1.)
    loss = Setting(0)

    def add_main_layout(self):
        self.base_estimator = TreeRegressionLearner()
        box = gui.widgetBox(self.controlArea, "Parameters")
        gui.spin(box, self, "n_estimators", 1, 100, label="Number of estimators",
                 alignment=Qt.AlignRight, callback=self.settings_changed)
        gui.doubleSpin(box, self, "learning_rate", 1e-5, 1.0, 1e-5,
                       label="Learning rate", decimals=5, alignment=Qt.AlignRight,
                       controlWidth=90, callback=self.settings_changed)
        gui.comboBox(box, self, "loss", label="Loss",
                     orientation="horizontal",
                     items=self.losses,
                     callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            preprocessors=self.preprocessors
        )

    def get_model_parameters(self):
        return (("Base estimator", self.base_estimator),
                ("Number of estimators", self.n_estimators),
                ("Loss", self.losses[self.loss].capitalize()))


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    a = QApplication(sys.argv)
    ow = OWAdaBoostRegression()
    ow.set_data(Table("housing"))
    ow.show()
    a.exec_()
    ow.saveSettings()
