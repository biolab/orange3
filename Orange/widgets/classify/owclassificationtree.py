from PyQt4.QtGui import QApplication
import Orange.data
from Orange.classification import tree
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWClassificationTree(widget.OWWidget):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    priority = 30
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [
        ("Learner", tree.ClassificationTreeLearner),
        ("Classification Tree", tree.ClassificationTreeClassifier)
    ]
    want_main_area = False

    model_name = Setting("Classification Tree")
    attribute_score = Setting(0)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)
    
    scores = (("Entropy", "entropy"), ("Gini Index", "gini"))

    def __init__(self):
        super().__init__()

        self.data = None
        self.learner = None
        self.preprocessor = None
        self.classifier = None

        gui.lineEdit(self.controlArea, self, 'model_name', box='Name',
                     tooltip='The name will identify this model in other '
                             'widgets')

        gui.comboBox(self.controlArea, self, "attribute_score",
                     box='Feature selection',
                     items=[name for name, _ in self.scores])

        box = gui.widgetBox(self.controlArea, 'Pruning')
        gui.spin(box, self, "min_leaf", 1, 1000,
                 label="Min. instances in leaves ", checked="limit_min_leaf")
        gui.spin(box, self, "min_internal", 1, 1000,
                 label="Stop splitting nodes with less instances than ",
                 checked="limit_min_internal")
        gui.spin(box, self, "max_depth", 1, 1000,
                 label="Limit the depth to ", checked="limit_depth")

        self.btn_apply = gui.button(self.controlArea, self, "&Apply",
                                    callback=self.set_learner, disabled=0,
                                    default=True)

        gui.rubber(self.controlArea)
        self.resize(100, 100)

        self.set_learner()

    def sendReport(self):
        self.reportSettings(
            "Model parameters",
            [("Attribute selection", self.scores[self.attribute_score][0]),
             ("Pruning", ", ".join(s for s, c in (
                 ("%i instances in leaves" % self.min_leaf,
                  self.limit_min_leaf),
                 ("%i instance in internal node" % self.min_internal,
                  self.limit_min_internal),
                 ("maximum depth %i" % self.max_depth, self.limit_depth)) if c)
              or ": None")])
        self.reportData(self.data)

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        self.set_learner()

    def set_learner(self):
        self.btn_apply.setFocus()
        self.learner = tree.ClassificationTreeLearner(
            criterion=self.scores[self.attribute_score][1],
            max_depth=self.max_depth,
            min_samples_split=self.min_internal, min_samples_leaf=self.min_leaf)
        self.learner.name = self.model_name
        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.send("Learner", self.learner)

        self.error(1)
        if self.data is not None:
            try:
                self.classifier = self.learner(self.data)
                self.classifier.name = self.model_name
                self.classifier.instances = self.data
            except Exception as errValue:
                self.error(1, str(errValue))
                self.classifier = None
        else:
            self.classifier = None
        self.send("Classification Tree", self.classifier)

    def set_data(self, data):
        self.error(0)
        self.data = data
        if data is not None and data.domain.class_var is None:
            self.error(0, "Data has no target variable")
            self.data = None
        self.set_learner()


if __name__ == "__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWClassificationTree()
    d = Orange.data.table.Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
