from Orange.data import Table
from Orange.classification.tree import TreeLearner, TreeClassifier
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner
from Orange.widgets.utils.sql import check_sql_input


class OWClassificationTree(OWProvidesLearner, widget.OWWidget):
    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    description = "Classification tree algorithm with forward pruning."
    priority = 30

    LEARNER = TreeLearner

    inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
    outputs = [
        ("Learner", LEARNER),
        ("Tree", TreeClassifier)
    ]
    want_main_area = False
    resizing_enabled = False

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
        self.preprocessors = None
        self.model = None

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

        box = gui.widgetBox(self.controlArea, True, orientation="horizontal")
        box.layout().addWidget(self.report_button)
        gui.separator(box, 20)
        self.btn_apply = gui.button(box, self, "&Apply",
                                    callback=self.apply, disabled=0,
                                    default=True)

        self.apply()

    def send_report(self):
        from Orange.canvas.report import plural_w
        self.report_items((("Name", self.model_name),))
        self.report_items(
            "Model parameters",
            [("Split selection", self.scores[self.attribute_score][0]),
             ("Pruning", ", ".join(s for s, c in (
                 (plural_w("at least {number} instance{s} in leaves",
                           self.min_leaf), self.limit_min_leaf),
                 (plural_w("at least {number} instance{s} in internal nodes",
                           self.min_internal), self.limit_min_internal),
                 ("maximum depth {}".format(self.max_depth),
                  self.limit_depth))
              if c) or "None")])
        if self.data:
            self.report_data("Data", self.data)

    def apply(self):
        self.learner = self.LEARNER(
            criterion=self.scores[self.attribute_score][1],
            max_depth=self.max_depth if self.limit_depth else None,
            min_samples_split=(self.min_internal if self.limit_min_internal
                               else 2),
            min_samples_leaf=(self.min_leaf if self.limit_min_leaf else 1),
            preprocessors=self.preprocessors
        )
        self.learner.name = self.model_name
        self.send("Learner", self.learner)
        self.update_model()

    def update_model(self):
        self.model = None
        if self.data is not None:
            self.error(1)
            if not self.learner.check_learner_adequacy(self.data.domain):
                self.error(1, self.learner.learner_adequacy_err_msg)
            else:
                self.model = self.learner(self.data)
                self.model.name = self.model_name
                self.model.instances = self.data

        self.send("Tree", self.model)

    @check_sql_input
    def set_data(self, data):
        self.error(0)
        self.data = data
        if data is not None and data.domain.class_var is None:
            self.error(0, "Data has no target variable")
            self.data = None
        self.update_model()


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWClassificationTree()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
