from Orange.data import Table
from Orange.modelling.constant import ConstantLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWConstant(OWBaseLearner):
    name = "Constant"
    description = "Predict the most frequent class or mean value " \
                  "from the training set."
    icon = "icons/Constant.svg"
    replaces = [
        "Orange.widgets.classify.owmajority.OWMajority",
        "Orange.widgets.regression.owmean.OWMean",
    ]
    priority = 10

    LEARNER = ConstantLearner


if __name__ == "__main__":  # pragma: no cover
    OWConstant.test_run(Table("iris"))
