from PyQt4 import QtGui

from Orange.data import Table
from Orange.classification.svm import SVMLearner, NuSVMLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseMultipleLearner


class OWBaseSVM(OWBaseMultipleLearner):
    def update_model(self):
        super().update_model()

        sv = None
        if self.valid_data:
            sv = self.data[self.model.skl_model.support_]
        self.send("Support vectors", sv)


class OWSVMClassification(OWBaseSVM):
    name = "SVM"
    description = "Support vector machines classifier with standard " \
                  "selection of kernels."
    icon = "icons/SVM.svg"
    priority = 50

    LEARNER = SVMLearner
    Learners = [SVMLearner(), NuSVMLearner()]

    outputs = [("Support vectors", Table)]


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWSVMClassification()
    w.set_data(Table("iris")[:50])
    w.show()
    app.exec_()
