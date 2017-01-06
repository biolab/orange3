from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QGridLayout
from Orange.data import Table
from Orange.modelling import SVMFitter
from Orange.widgets import gui
from Orange.widgets.model.owsvm import OWSVM


class OWSVM(OWSVM):
    name = "SVM Regression"
    description = "Support Vector Machines map inputs to higher-dimensional " \
                  "feature spaces that best map instances to a linear function."
    icon = "icons/SVMRegression.svg"

    LEARNER = SVMFitter

    def _add_type_box(self):
        form = QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "svm_type", [], box="SVR Type",
            orientation=form)

        self.epsilon_radio = gui.appendRadioButton(
            box, "ε-SVR", addToLayout=False)
        self.C_spin = gui.doubleSpin(
            box, self, "C", 0.1, 512.0, 0.1, decimals=2, addToLayout=False)
        self.epsilon_spin = gui.doubleSpin(
            box, self, "epsilon", 0.1, 512.0, 0.1, decimals=2,
            addToLayout=False)
        form.addWidget(self.epsilon_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 0, 1, Qt.AlignRight)
        form.addWidget(self.C_spin, 0, 2)
        form.addWidget(QLabel("Loss epsilon (ε):"), 1, 1, Qt.AlignRight)
        form.addWidget(self.epsilon_spin, 1, 2)

        self.nu_radio = gui.appendRadioButton(box, "ν-SVR", addToLayout=False)
        self.nu_C_spin = gui.doubleSpin(
            box, self, "nu_C", 0.1, 512.0, 0.1, decimals=2, addToLayout=False)
        self.nu_spin = gui.doubleSpin(
            box, self, "nu", 0.05, 1.0, 0.05, decimals=2, addToLayout=False)
        form.addWidget(self.nu_radio, 2, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 2, 1, Qt.AlignRight)
        form.addWidget(self.nu_C_spin, 2, 2)
        form.addWidget(QLabel("Complexity bound (ν):"), 3, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 3, 2)


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWSVM()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
