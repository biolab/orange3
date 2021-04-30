from typing import List

from Orange.base import Learner
from Orange.data import Table
from Orange.ensembles.stack import StackedFitter
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Input, MultiInput


class OWStackedLearner(OWBaseLearner):
    name = "Stacking"
    description = "Stack multiple models."
    icon = "icons/Stacking.svg"
    priority = 100

    LEARNER = StackedFitter

    learner_name = Setting("Stack")

    class Inputs(OWBaseLearner.Inputs):
        learners = MultiInput("Learners", Learner, filter_none=True)
        aggregate = Input("Aggregate", Learner)

    def __init__(self):
        self.learners: List[Learner] = []
        self.aggregate = None
        super().__init__()

    def add_main_layout(self):
        pass

    @Inputs.learners
    def set_learner(self, index: int, learner: Learner):
        self.learners[index] = learner

    @Inputs.learners.insert
    def insert_learner(self, index, learner):
        self.learners.insert(index, learner)

    @Inputs.learners.remove
    def remove_learner(self, index):
        self.learners.pop(index)

    @Inputs.aggregate
    def set_aggregate(self, aggregate):
        self.aggregate = aggregate

    def handleNewSignals(self):
        super().handleNewSignals()
        self.apply()

    def create_learner(self):
        if not self.learners:
            return None
        return self.LEARNER(
            tuple(self.learners), aggregate=self.aggregate,
            preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return (("Base learners", [l.name for l in self.learners]),
                ("Aggregator",
                 self.aggregate.name if self.aggregate else 'default'))


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWStackedLearner()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()
