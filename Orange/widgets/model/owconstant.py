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


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWConstant()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else "iris")
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
