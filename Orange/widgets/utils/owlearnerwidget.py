from copy import deepcopy
import numpy as np

from AnyQt.QtCore import QTimer, Qt

from Orange.data import Table
from Orange.modelling import Fitter, Learner, Model
from Orange.preprocess.preprocess import Preprocess
from Orange.statistics import util as ut
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
    def __new__(cls, name, bases, attributes, **kwargs):
        def abstract_widget():
            return not attributes.get("name")

        def copy_outputs(template):
            result = type("Outputs", (), {})
            for name, signal in getmembers(template, Output):
                setattr(result, name, deepcopy(signal))
            return result

        obj = super().__new__(cls, name, bases, attributes, **kwargs)
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


class OWBaseLearner(OWWidget, metaclass=OWBaseLearnerMeta, openclass=True):
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

    learner_name = Setting("", schema_only=True)
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

    class Information(OWWidget.Information):
        ignored_preprocessors = Msg(
            "Ignoring default preprocessing.\n"
            "Default preprocessing, such as scaling, one-hot encoding and "
            "treatment of missing data, has been replaced with user-specified "
            "preprocessors. Problems may occur if these are inadequate "
            "for the given data.")

    class Inputs:
        data = Input("Data", Table)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        learner = Output("Learner", Learner, dynamic=False)
        model = Output("Model", Model, dynamic=False,
                       replaces=["Classifier", "Predictor"])

    OUTPUT_MODEL_NAME = Outputs.model.name  # Attr for backcompat w/ self.send() code

    _SEND, _SOFT, _UPDATE = range(3)

    def __init__(self, preprocessors=None):
        super().__init__()
        self.__default_learner_name = ""
        self.data = None
        self.valid_data = False
        self.learner = None
        self.model = None
        self.preprocessors = preprocessors
        self.outdated_settings = False
        self.__apply_level = []

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

    def default_learner_name(self) -> str:
        """
        Return the default learner name.

        By default this is the same as the widget's name.
        """
        return self.__default_learner_name or self.captionTitle

    def set_default_learner_name(self, name: str) -> None:
        """
        Set the default learner name if not otherwise specified by the user.
        """
        changed = name != self.__default_learner_name
        if name:
            self.name_line_edit.setPlaceholderText(name)
        else:
            self.name_line_edit.setPlaceholderText(self.captionTitle)
        self.__default_learner_name = name
        if not self.learner_name and changed:
            self.learner_name_changed()

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.preprocessors = preprocessor
        # invalidate learner and model, so handleNewSignals will renew them
        self.learner = self.model = None

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set the input train dataset."""
        self.Error.data_error.clear()
        self.data = data

        if data is not None and data.domain.class_var is None:
            if data.domain.class_vars:
                self.Error.data_error(
                    "Data contains multiple target variables.\n"
                    "Select a single one with the Select Columns widget.")
            else:
                self.Error.data_error(
                    "Data has no target variable.\n"
                    "Select one with the Select Columns widget.")
            self.data = None

        # invalidate the model so that handleNewSignals will update it
        self.model = None


    def apply(self):
        level, self.__apply_level = max(self.__apply_level, default=self._UPDATE), []
        """Applies learner and sends new model."""
        if level == self._SEND:
            self._send_learner()
            self._send_model()
        elif level == self._UPDATE:
            self.update_learner()
            self.update_model()
        else:
            self.learner or self.update_learner()
            self.model or self.update_model()

    def apply_as(self, level, unconditional=False):
        self.__apply_level.append(level)
        if unconditional:
            self.unconditional_apply()
        else:
            self.apply()

    def update_learner(self):
        self.learner = self.create_learner()
        if self.learner and issubclass(self.LEARNER, Fitter):
            self.learner.use_default_preprocessors = True
        if self.learner is not None:
            self.learner.name = self.effective_learner_name()
        self._send_learner()

    def _send_learner(self):
        self.Outputs.learner.send(self.learner)
        self.outdated_settings = False
        self.Warning.outdated_learner.clear()

    def handleNewSignals(self):
        self.apply_as(self._SOFT, True)
        self.Information.ignored_preprocessors(
            shown=not getattr(self.learner, "use_default_preprocessors", False)
                  and getattr(self.LEARNER, "preprocessors", False)
                  and self.preprocessors is not None)

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
                self.model.name = self.learner_name or self.captionTitle
                self.model.instances = self.data
        self._send_model()

    def _send_model(self):
        self.Outputs.model.send(self.model)

    def check_data(self):
        self.valid_data = False
        self.Error.sparse_not_supported.clear()
        if self.data is not None and self.learner is not None:
            self.Error.data_error.clear()

            reason = self.learner.incompatibility_reason(self.data.domain)
            if reason is not None:
                self.Error.data_error(reason)
            elif not len(self.data):
                self.Error.data_error("Dataset is empty.")
            elif len(np.asarray(ut.unique(self.data.Y))) < 2:
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

    def learner_name_changed(self):
        if self.model is not None:
            self.model.name = self.effective_learner_name()
        if self.learner is not None:
            self.learner.name = self.effective_learner_name()
        self.apply_as(self._SEND)

    def effective_learner_name(self):
        """Return the effective learner name."""
        return self.learner_name or self.name_line_edit.placeholderText()

    def send_report(self):
        self.report_items((("Name", self.effective_learner_name()),))

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

    def add_classification_layout(self, box):
        """Creates layout for classification specific options.

        If a widget outputs a learner dispatcher, sometimes the classification
        and regression learners require different options.
        See `setup_layout()` method for execution order.
        """

    def add_regression_layout(self, box):
        """Creates layout for regression specific options.

        If a widget outputs a learner dispatcher, sometimes the classification
        and regression learners require different options.
        See `setup_layout()` method for execution order.
        """

    def add_learner_name_widget(self):
        self.name_line_edit = gui.lineEdit(
            self.controlArea, self, 'learner_name', box='Name',
            placeholderText=self.captionTitle,
            tooltip='The name will identify this model in other widgets',
            orientation=Qt.Horizontal, callback=self.learner_name_changed)

    def setCaption(self, caption):
        super().setCaption(caption)
        if not self.__default_learner_name:
            self.name_line_edit.setPlaceholderText(caption)
            if not self.learner_name:
                self.learner_name_changed()

    def add_bottom_buttons(self):
        self.apply_button = gui.auto_apply(self.buttonsArea, self, commit=self.apply)

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
