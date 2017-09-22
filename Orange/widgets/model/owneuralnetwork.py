import re
import sys

from AnyQt.QtWidgets import QApplication
from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.modelling import NNLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWNNLearner(OWBaseLearner):
    name = "Neural Network"
    description = "A multi-layer perceptron (MLP) algorithm with " \
                  "backpropagation."
    icon = "icons/NN.svg"
    priority = 90

    LEARNER = NNLearner

    activation = ["identity", "logistic", "tanh", "relu"]
    act_lbl = ["Identity", "Logistic", "tanh", "ReLu"]
    solver = ["lbfgs", "sgd", "adam"]
    solv_lbl = ["L-BFGS-B", "SGD", "Adam"]

    learner_name = Setting("Neural Network")
    hidden_layers_input = Setting("100,")
    activation_index = Setting(3)
    solver_index = Setting(2)
    alpha = Setting(0.0001)
    max_iterations = Setting(200)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Network")
        self.hidden_layers_edit = gui.lineEdit(
            box, self, "hidden_layers_input", label="Neurons per hidden layer:",
            orientation=Qt.Horizontal, callback=self.settings_changed,
            tooltip="A list of integers defining neurons. Length of list "
                    "defines the number of layers. E.g. 4, 2, 2, 3.",
            placeholderText="e.g. 100,")
        self.activation_combo = gui.comboBox(
            box, self, "activation_index", orientation=Qt.Horizontal,
            label="Activation:", items=[i for i in self.act_lbl],
            callback=self.settings_changed)
        self.solver_combo = gui.comboBox(
            box, self, "solver_index", orientation=Qt.Horizontal,
            label="Solver:", items=[i for i in self.solv_lbl],
            callback=self.settings_changed)
        self.alpha_spin = gui.doubleSpin(
            box, self, "alpha", 1e-5, 1.0, 1e-2,
            label="Alpha:", decimals=5, alignment=Qt.AlignRight,
            callback=self.settings_changed, controlWidth=80)
        self.max_iter_spin = gui.spin(
            box, self, "max_iterations", 10, 300, step=10,
            label="Max iterations:", orientation=Qt.Horizontal,
            alignment=Qt.AlignRight, callback=self.settings_changed,
            controlWidth=80)

    def create_learner(self):
        return self.LEARNER(
            hidden_layer_sizes=self.get_hidden_layers(),
            activation=self.activation[self.activation_index],
            solver=self.solver[self.solver_index],
            alpha=self.alpha,
            max_iter=self.max_iterations,
            preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return (("Hidden layers", ', '.join(map(str, self.get_hidden_layers()))),
                ("Activation", self.act_lbl[self.activation_index]),
                ("Solver", self.solv_lbl[self.solver_index]),
                ("Alpha", self.alpha),
                ("Max iterations", self.max_iterations))

    def get_hidden_layers(self):
        layers = tuple(map(int, re.findall(r'\d+', self.hidden_layers_input)))
        if not layers:
            layers = (100,)
            self.hidden_layers_edit.setText("100,")
        return layers


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWNNLearner()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
