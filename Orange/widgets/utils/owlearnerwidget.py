from copy import deepcopy

import numpy as np

from AnyQt.QtCore import QTimer, Qt

from Orange.data import Table
from Orange.modelling import Fitter, Learner, Model
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import getmembers
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, WidgetMetaClass, Msg


class OWBaseLearnerMeta(WidgetMetaClass):
    """ Meta class for learner widgets

    OWBaseLearner declares two outputs, learner and model with
    generic type (Learner and Model).

    This metaclass ensures that each of the subclasses gets
    its own Outputs class with output that match the corresponding
    learner.
    """
    def __new__(cls, name, bases, attributes):
        def abstract_widget():
            return not attributes.get("name")

        def copy_outputs(template):
            result = type("Outputs", (), {})
            for name, signal in getmembers(template, Output):
                setattr(result, name, deepcopy(signal))
            return result

        obj = super().__new__(cls, name, bases, attributes)
        if abstract_widget():
            return obj

        learner = attributes.get("LEARNER")
        if not learner:
            raise AttributeError(
                "'{}' must declare attribute LEARNER".format(name))

        outputs = obj.Outputs = copy_outputs(obj.Outputs)
        outputs.learner.type = learner
        outputs.model.type = learner.__returns__

        return obj


class OWBaseLearner(OWWidget, metaclass=OWBaseLearnerMeta):
    """Abstract widget for classification/regression learners.

    Notes
    -----
    All learner widgets should define learner class LEARNER.
    LEARNER should have __returns__ attribute.

    Overwrite `create_learner`, `add_main_layout` and `get_learner_parameters`
    in case LEARNER has extra parameters.

    """
    LEARNER = None
    supports_sparse = True

    learner_name = Setting(None, schema_only=True)
    want_main_area = False
    resizing_enabled = False
    auto_apply = Setting(True)

    class Error(OWWidget.Error):
        data_error = Msg("{}")
        fitting_failed = Msg("Fitting failed.\n{}")
        sparse_not_supported = Msg("Sparse data is not supported.")
        out_of_memory = Msg("Out of memory.")

    class Warning(OWWidget.Warning):
        outdated_learner = Msg("Press Apply to submit changes.")

    class Inputs:
        data = Input("Data", Table)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        learner = Output("Learner", Learner, dynamic=False)
        model = Output("Model", Model, dynamic=False,
                       replaces=["Classifier", "Predictor"])

    OUTPUT_MODEL_NAME = Outputs.model.name  # Attr for backcompat w/ self.send() code

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = False
        self.learner = None
        if self.learner_name is None:
            self.learner_name = self.name
        self.model = None
        self.preprocessors = None
        self.outdated_settings = False

        self.setup_layout()
        QTimer.singleShot(0, getattr(self, "unconditional_apply", self.apply))

    def create_learner(self):
        """Creates a learner with current configuration.

        Returns:
            Learner: an instance of Orange.base.learner subclass.
        """
        return self.LEARNER(preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        """Creates an `OrderedDict` or a sequence of pairs with current model
        configuration.

        Returns:
            OrderedDict or List: (option, value) pairs or dict
        """
        return []

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.preprocessors = preprocessor
        self.apply()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set the input train dataset."""
        self.Error.data_error.clear()
        self.data = data
        if data is not None and data.domain.class_var is None:
            self.Error.data_error("Data has no target variable.")
            self.data = None

        self.update_model()

    def apply(self):
        """Applies learner and sends new model."""
        self.update_learner()
        self.update_model()

    def update_learner(self):
        self.learner = self.create_learner()
        if self.learner and issubclass(self.LEARNER, Fitter):
            self.learner.use_default_preprocessors = True
        if self.learner is not None:
            self.learner.name = self.learner_name
        self.Outputs.learner.send(self.learner)
        self.outdated_settings = False
        self.Warning.outdated_learner.clear()

    def show_fitting_failed(self, exc):
        """Show error when fitting fails.
            Derived widgets can override this to show more specific messages."""
        self.Error.fitting_failed(str(exc), shown=exc is not None)

    def update_model(self):
        self.show_fitting_failed(None)
        self.model = None
        if self.check_data():
            try:
                self.model = self.learner(self.data)
            except BaseException as exc:
                self.show_fitting_failed(exc)
            else:
                self.model.name = self.learner_name
                self.model.instances = self.data
        self.Outputs.model.send(self.model)

    def check_data(self):
        self.valid_data = False
        self.Error.sparse_not_supported.clear()
        if self.data is not None and self.learner is not None:
            self.Error.data_error.clear()
            if not self.learner.check_learner_adequacy(self.data.domain):
                self.Error.data_error(self.learner.learner_adequacy_err_msg)
            elif not len(self.data):
                self.Error.data_error("Dataset is empty.")
            elif len(np.unique(self.data.Y)) < 2:
                self.Error.data_error("Data contains a single target value.")
            elif self.data.X.size == 0:
                self.Error.data_error("Data has no features to learn from.")
            elif self.data.is_sparse() and not self.supports_sparse:
                self.Error.sparse_not_supported()
            else:
                self.valid_data = True
        return self.valid_data

    def settings_changed(self, *args, **kwargs):
        self.outdated_settings = True
        self.Warning.outdated_learner(shown=not self.auto_apply)
        self.apply()

    def _change_name(self, instance, output):
        if instance:
            instance.name = self.learner_name
            if self.auto_apply:
                output.send(instance)

    def learner_name_changed(self):
        self._change_name(self.learner, self.Outputs.learner)
        self._change_name(self.model, self.Outputs.model)

    def send_report(self):
        self.report_items((("Name", self.learner_name),))

        model_parameters = self.get_learner_parameters()
        if model_parameters:
            self.report_items("Model parameters", model_parameters)

        if self.data:
            self.report_data("Data", self.data)

    # GUI
    def setup_layout(self):
        self.add_learner_name_widget()
        self.add_main_layout()
        # Options specific to target variable type, if supported
        if issubclass(self.LEARNER, Fitter):
            # Only add a classification section if the method is overridden
            if type(self).add_classification_layout is not \
                    OWBaseLearner.add_classification_layout:
                classification_box = gui.widgetBox(
                    self.controlArea, 'Classification')
                self.add_classification_layout(classification_box)
            # Only add a regression section if the method is overridden
            if type(self).add_regression_layout is not \
                    OWBaseLearner.add_regression_layout:
                regression_box = gui.widgetBox(self.controlArea, 'Regression')
                self.add_regression_layout(regression_box)
        self.add_bottom_buttons()

    def add_main_layout(self):
        """Creates layout with the learner configuration widgets.

        Override this method for laying out any learner-specific parameter controls.
        See setup_layout() method for execution order.
        """
        pass

    def add_classification_layout(self, box):
        """Creates layout for classification specific options.

        If a widget outputs a learner dispatcher, sometimes the classification
        and regression learners require different options.
        See `setup_layout()` method for execution order.
        """
        pass

    def add_regression_layout(self, box):
        """Creates layout for regression specific options.

        If a widget outputs a learner dispatcher, sometimes the classification
        and regression learners require different options.
        See `setup_layout()` method for execution order.
        """
        pass

    def add_learner_name_widget(self):
        self.name_line_edit = gui.lineEdit(
            self.controlArea, self, 'learner_name', box='Name',
            tooltip='The name will identify this model in other widgets',
            orientation=Qt.Horizontal, callback=self.learner_name_changed)

    def add_bottom_buttons(self):
        self.apply_button = gui.auto_commit(
            self.controlArea, self, 'auto_apply', '&Apply',
            box=True, commit=self.apply)

    def send(self, signalName, value, id=None):
        # A subclass might still use the old syntax to send outputs
        # defined on this class
        for _, output in getmembers(self.Outputs, Output):
            if output.name == signalName or signalName in output.replaces:
                output.send(value, id=id)
                return

        super().send(signalName, value, id)

    @classmethod
    def get_widget_description(cls):
        # When a subclass defines defines old-style signals, those override
        # the new-style ones, so we add them manually
        desc = super().get_widget_description()

        if cls.outputs:
            desc["outputs"].extend(cls.get_signals("outputs", True))
        if cls.inputs:
            desc["inputs"].extend(cls.get_signals("inputs", True))
        return desc
