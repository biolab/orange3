import Orange.data
import Orange.classification.majority
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWMajority(widget.OWWidget):
    name = "Majority"
    description = "Classification to the most frequent class " \
                  "from the training set."
    priority = 20
    icon = "icons/Majority.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", Orange.classification.majority.MajorityLearner),
               ("Classifier", Orange.classification.majority.ConstantModel)]

    learner_name = Setting("Majority")

    def __init__(self, parent=None):
        super().__init__(parent)
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
        learner = Orange.classification.majority.MajorityLearner(
            preprocessors=self.preprocessors)

        learner.name = self.learner_name
        classifier = None
        if self.data is not None:
            try:
                self.warning(0)
                classifier = learner(self.data)
                classifier.name = self.learner_name
            except ValueError as err:
                self.warning(0, str(err))

        self.send("Learner", learner)
        self.send("Classifier", classifier)
