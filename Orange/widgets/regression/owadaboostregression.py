from PyQt4.QtCore import Qt

from Orange.regression.base_regression import LearnerRegression
from Orange.data import Table
from Orange.ensembles import SklAdaBoostRegressionLearner
from Orange.widgets import gui
from Orange.widgets.classify import owadaboost
from Orange.widgets.settings import Setting


class OWAdaBoostRegression(owadaboost.OWAdaBoostClassification):
    name = "AdaBoost"
    description = "An ensemble meta-algorithm that combines weak learners " \
                  "and adapts to the 'hardness' of each training sample. "
    icon = "icons/AdaBoost.svg"
    priority = 80

    LEARNER = SklAdaBoostRegressionLearner

    inputs = [("Learner", LearnerRegression, "set_base_learner")]

    losses = ["Linear", "Square", "Exponential"]
    loss = Setting(0)

    def add_specific_parameters(self, box):
        gui.comboBox(box, self, "loss", label="Loss:",
                     orientation=Qt.Horizontal, items=self.losses,
                     callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            preprocessors=self.preprocessors
        )

    def get_learner_parameters(self):
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
