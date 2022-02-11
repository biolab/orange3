from functools import partial
import copy
import logging
import re
import concurrent.futures
from itertools import chain

import numpy as np

from AnyQt.QtWidgets import QFormLayout, QLabel
from AnyQt.QtCore import Qt, QThread, QObject
from AnyQt.QtCore import pyqtSlot as Slot, pyqtSignal as Signal

from Orange.data import Table
from Orange.modelling import NNLearner
from Orange.widgets import gui
from Orange.widgets.widget import Msg
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher
from Orange.widgets.utils.widgetpreview import WidgetPreview


class Task(QObject):
    """
    A class that will hold the state for an learner evaluation.
    """
    done = Signal(object)
    progressChanged = Signal(float)

    future = None      # type: concurrent.futures.Future
    watcher = None     # type: FutureWatcher
    cancelled = False  # type: bool

    def setFuture(self, future):
        if self.future is not None:
            raise RuntimeError("future is already set")
        self.future = future
        self.watcher = FutureWatcher(future, parent=self)
        self.watcher.done.connect(self.done)

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        self.future.cancel()
        concurrent.futures.wait([self.future])

    def emitProgressUpdate(self, value):
        self.progressChanged.emit(value)

    def isInterruptionRequested(self):
        return self.cancelled


class CancelTaskException(BaseException):
    pass


class OWNNLearner(OWBaseLearner):
    name = "Neural Network"
    description = "A multi-layer perceptron (MLP) algorithm with " \
                  "backpropagation."
    icon = "icons/NN.svg"
    priority = 90
    keywords = ["mlp"]

    LEARNER = NNLearner

    activation = ["identity", "logistic", "tanh", "relu"]
    act_lbl = ["Identity", "Logistic", "tanh", "ReLu"]
    solver = ["lbfgs", "sgd", "adam"]
    solv_lbl = ["L-BFGS-B", "SGD", "Adam"]

    learner_name = Setting("Neural Network")
    hidden_layers_input = Setting("100,")
    activation_index = Setting(3)
    solver_index = Setting(2)
    max_iterations = Setting(200)
    alpha_index = Setting(1)
    replicable = Setting(True)
    settings_version = 2

    alphas = list(chain([0], [x / 10000 for x in range(1, 10)],
                        [x / 1000 for x in range(1, 10)],
                        [x / 100 for x in range(1, 10)],
                        [x / 10 for x in range(1, 10)],
                        range(1, 10),
                        range(10, 100, 5),
                        range(100, 200, 10),
                        range(100, 1001, 50)))

    class Warning(OWBaseLearner.Warning):
        no_layers = Msg("ANN without hidden layers is equivalent to logistic "
                        "regression with worse fitting.\nWe recommend using "
                        "logistic regression.")

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        gui.widgetBox(self.controlArea, True, orientation=form)
        form.addRow(
            "Neurons in hidden layers:",
            gui.lineEdit(
                None, self, "hidden_layers_input",
                orientation=Qt.Horizontal, callback=self.settings_changed,
                tooltip="A list of integers defining neurons. Length of list "
                        "defines the number of layers. E.g. 4, 2, 2, 3.",
                placeholderText="e.g. 10,"))
        form.addRow(
            "Activation:",
            gui.comboBox(
                None, self, "activation_index", orientation=Qt.Horizontal,
                label="Activation:", items=[i for i in self.act_lbl],
                callback=self.settings_changed))

        form.addRow(
            "Solver:",
            gui.comboBox(
                None, self, "solver_index", orientation=Qt.Horizontal,
                label="Solver:", items=[i for i in self.solv_lbl],
                callback=self.settings_changed))
        self.reg_label = QLabel()
        slider = gui.hSlider(
            None, self, "alpha_index",
            minValue=0, maxValue=len(self.alphas) - 1,
            callback=lambda: (self.set_alpha(), self.settings_changed()),
            createLabel=False)
        form.addRow(self.reg_label, slider)
        self.set_alpha()

        form.addRow(
            "Maximal number of iterations:",
            gui.spin(
                None, self, "max_iterations", 10, 1000000, step=10,
                label="Max iterations:", orientation=Qt.Horizontal,
                alignment=Qt.AlignRight, callback=self.settings_changed))

        form.addRow(
            gui.checkBox(
                None, self, "replicable", label="Replicable training",
                callback=self.settings_changed, attribute=Qt.WA_LayoutUsesWidgetRect)
        )

    def set_alpha(self):
        # called from init, pylint: disable=attribute-defined-outside-init
        self.strength_C = self.alphas[self.alpha_index]
        self.reg_label.setText("Regularization, α={}:".format(self.strength_C))

    @property
    def alpha(self):
        return self.alphas[self.alpha_index]

    def setup_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        super().setup_layout()

        self._task = None  # type: Optional[Task]
        self._executor = ThreadExecutor()

        # just a test cancel button
        b = gui.button(self.apply_button, self, "Cancel",
                       callback=self.cancel, addToLayout=False)
        self.apply_button.layout().insertStretch(0, 100)
        self.apply_button.layout().insertWidget(0, b)

    def create_learner(self):
        return self.LEARNER(
            hidden_layer_sizes=self.get_hidden_layers(),
            activation=self.activation[self.activation_index],
            solver=self.solver[self.solver_index],
            alpha=self.alpha,
            random_state=1 if self.replicable else None,
            max_iter=self.max_iterations,
            preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return (("Hidden layers", ', '.join(map(str, self.get_hidden_layers()))),
                ("Activation", self.act_lbl[self.activation_index]),
                ("Solver", self.solv_lbl[self.solver_index]),
                ("Alpha", self.alpha),
                ("Max iterations", self.max_iterations),
                ("Replicable training", self.replicable))

    def get_hidden_layers(self):
        self.Warning.no_layers.clear()
        layers = tuple(map(int, re.findall(r'\d+', self.hidden_layers_input)))
        if not layers:
            self.Warning.no_layers()
        return layers

    def update_model(self):
        self.show_fitting_failed(None)
        self.model = None
        if self.check_data():
            self.__update()
        else:
            self.Outputs.model.send(self.model)

    @Slot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    def __update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        max_iter = self.learner.kwargs["max_iter"]

        # Setup the task state
        task = Task()
        lastemitted = 0.

        def callback(iteration):
            nonlocal task
            nonlocal lastemitted
            if task.isInterruptionRequested():
                raise CancelTaskException()
            progress = round(iteration / max_iter * 100)
            if progress != lastemitted:
                task.emitProgressUpdate(progress)
                lastemitted = progress

        # copy to set the callback so that the learner output is not modified
        # (currently we can not pass callbacks to learners __call__)
        learner = copy.copy(self.learner)
        learner.callback = callback

        def build_model(data, learner):
            try:
                return learner(data)
            except CancelTaskException:
                return None

        build_model_func = partial(build_model, self.data, learner)

        task.setFuture(self._executor.submit(build_model_func))
        task.done.connect(self._task_finished)
        task.progressChanged.connect(self.setProgressValue)

        # set in setup_layout; pylint: disable=attribute-defined-outside-init
        self._task = task

        self.progressBarInit()
        self.setBlocking(True)

    @Slot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters
        ----------
        f : Future
            The future instance holding the built model
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()
        self._task.deleteLater()
        self._task = None  # pylint: disable=attribute-defined-outside-init
        self.setBlocking(False)
        self.progressBarFinished()

        try:
            self.model = f.result()
        except Exception as ex:  # pylint: disable=broad-except
            # Log the exception with a traceback
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.model = None
            self.show_fitting_failed(ex)
        else:
            self.model.name = self.learner_name
            self.model.instances = self.data
            self.model.skl_model.orange_callback = None  # remove unpicklable callback
            self.Outputs.model.send(self.model)

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect from the task
            self._task.done.disconnect(self._task_finished)
            self._task.progressChanged.disconnect(self.setProgressValue)
            self._task.deleteLater()
            self._task = None  # pylint: disable=attribute-defined-outside-init

        self.progressBarFinished()
        self.setBlocking(False)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version:
            alpha = settings.pop("alpha", None)
            if alpha is not None:
                settings["alpha_index"] = \
                    np.argmin(np.abs(np.array(cls.alphas) - alpha))
        elif version < 2:
            settings["alpha_index"] = settings.get("alpha_index", 0) + 1


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWNNLearner).run(Table("iris"))
