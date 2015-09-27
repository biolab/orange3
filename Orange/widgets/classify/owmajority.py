from Orange.data import Table
from Orange.classification.majority import MajorityLearner, ConstantModel
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWMajority(widget.OWWidget):
    name = "Majority"
    description = "Classification to the most frequent class " \
                  "from the training set."
    priority = 20
    icon = "icons/Majority.svg"

    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", MajorityLearner),
               ("Classifier", ConstantModel)]

    learner_name = Setting("Majority")

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.data = None
        self.preprocessors = None
        gui.lineEdit(
            gui.widgetBox(self.controlArea, "Learner/Classifier Name"),
            self, "learner_name"
        )
        gui.button(self.controlArea, self, "Apply", callback=self.apply,
                   default=True)

        self.apply()

    def set_data(self, data):
        self.error(0)
        if data is not None and not data.domain.has_discrete_class:
            self.error(0, "Data does not have a discrete target variable")
            data = None
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
        learner = MajorityLearner(
            preprocessors=self.preprocessors
        )
        learner.name = self.learner_name
        classifier = None

        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def send_report(self):
        self.report_items("", (("Name", self.learner_name),))


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
