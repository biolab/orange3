from Orange.classification.base_classification import LearnerClassification
from Orange.ensembles import SklAdaBoostLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWAdaBoostClassification(OWBaseLearner):
    name = "AdaBoost"
    description = "An AdaBoost classifier."
    icon = "icons/AdaBoost.svg"
    priority = 80

    LEARNER = SklAdaBoostLearner

    inputs = [("Learner", LearnerClassification, "set_base_learner")]

    def __init__(self):
        super().__init__()
        self.set_base_learner(None)

    def set_base_learner(self, model):
        self.base_estimator = model
        self.learner.base_estimator = model
        if self.base_estimator:
            self.apply_button.setDisabled(False)
        else:
            self.apply_button.setDisabled(True)


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWAdaBoostClassification()
    ow.set_data(Table("iris"))
    ow.show()
    a.exec_()
    ow.saveSettings()
