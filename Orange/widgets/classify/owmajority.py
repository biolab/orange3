from Orange.data import Table
from Orange.classification.majority import MajorityLearner
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWMajority(OWBaseLearner):
    name = "Majority"
    description = "Classification to the most frequent class " \
                  "from the training set."
    icon = "icons/Majority.svg"
    priority = 10

    LEARNER = MajorityLearner
    OUTPUT_MODEL_NAME = "Classifier"

    learner_name = Setting("Majority")

    want_main_area = False
    resizing_enabled = False

    def create_learner(self):
        return self.LEARNER(
            preprocessors=self.preprocessors
        )


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWMajority()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
