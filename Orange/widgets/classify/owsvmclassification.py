from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QGridLayout
from Orange.data import Table
from Orange.modelling import SVMFitter
from Orange.widgets import gui
from Orange.widgets.model.owsvm import OWSVM


class OWSVM(OWSVM):
    name = "SVM Classification"
    description = "Support Vector Machines map inputs to higher-dimensional " \
                  "feature spaces that best separate different classes. "

    LEARNER = SVMFitter

    def _add_type_box(self):
        form = QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "svm_type", [], box="SVM Type",
            orientation=form, callback=self.settings_changed)

        self.c_radio = gui.appendRadioButton(box, "C-SVM", addToLayout=False)
        self.nu_radio = gui.appendRadioButton(box, "ν-SVM", addToLayout=False)
        self.c_spin = gui.doubleSpin(
            box, self, "C", 1e-3, 1000.0, 0.1, decimals=3,
            alignment=Qt.AlignRight, controlWidth=80, addToLayout=False,
            callback=self.settings_changed)
        self.nu_spin = gui.doubleSpin(
            box, self, "nu", 0.05, 1.0, 0.05, decimals=2,
            alignment=Qt.AlignRight, controlWidth=80, addToLayout=False,
            callback=self.settings_changed)
        form.addWidget(self.c_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 0, 1, Qt.AlignRight)
        form.addWidget(self.c_spin, 0, 2)
        form.addWidget(self.nu_radio, 1, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Complexity (ν):"), 1, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 1, 2)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication

    app = QApplication([])
    w = OWSVM()
    w.set_data(Table("iris")[:50])
    w.show()
    app.exec_()
