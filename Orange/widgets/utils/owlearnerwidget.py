import numpy as np
from AnyQt.QtCore import QTimer, Qt

from Orange.canvas.registry import InputSignal, OutputSignal
from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table
from Orange.modelling import Fitter
from Orange.preprocess.preprocess import Preprocess
from Orange.regression.base_regression import LearnerRegression
from Orange.widgets import gui
from Orange.widgets import widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, WidgetMetaClass, Msg, Input, Output


class DefaultWidgetChannelsMetaClass(WidgetMetaClass):
    """Metaclass that adds default inputs and outputs objects.
    """

    REQUIRED_ATTRIBUTES = []

    def __new__(mcls, name, bases, attrib):
        # check whether it is abstract class
        if attrib.get('name', False):
            # Ensure all needed attributes are present
            if not all(attr in attrib for attr in mcls.REQUIRED_ATTRIBUTES):
                raise AttributeError(
                    "'{}' must have '{}' attributes"
                    .format(name, "', '".join(mcls.REQUIRED_ATTRIBUTES)))

            outputs = ()
            if 'Outputs' in attrib and isinstance(attrib['Outputs'], type):
                outputs = attrib.pop('Outputs')
            elif 'outputs' in attrib and isinstance(attrib['outputs'], (list, tuple)):
                outputs = attrib.pop('outputs')

            inputs = ()
            if isinstance(attrib.get('Inputs'), type):
                inputs = attrib.pop('Inputs')
            elif isinstance(attrib.get('inputs'), (list, tuple)):
                inputs = attrib.pop('inputs')

            attrib['Outputs'] = mcls._merge_signals(
                mcls.default_outputs(attrib), outputs, attrib, Output)
            attrib['Inputs'] = mcls._merge_signals(
                mcls.default_inputs(attrib), inputs, attrib, Input)

            mcls.add_extra_attributes(name, bases, attrib)

        return super().__new__(mcls, name, bases, attrib)

    @classmethod
    def default_inputs(cls, attrib):
        return []

    @classmethod
    def default_outputs(cls, attrib):
        return []

    @classmethod
    def _merge_signals(cls, existing, signals, attrib, sigcls):
        # signals DON'T override existing with same names
        default_names = set(sig.name for _, sig in existing)

        # If signals is Inputs or Outputs class, just add the default signals
        # as attributes. They will be sorted first because of forced _seq_id-s.
        if isinstance(signals, type):
            for name, sig in existing:
                setattr(signals, name, sig)
            existing = signals

        else:
            # Otherwise, old-style signal spec can be a sequence of
            # raw *Signal specifications or of 2- or 3-tuples.
            # Handle all these cases in an arbitrarily mixed manner.
            assert isinstance(signals, (list, tuple))
            SIGNAL_TYPE = (InputSignal, OutputSignal, Input, Output)
            normed = lambda name: name.lower().replace(' ', '_')

            for sig in signals:
                if isinstance(sig, SIGNAL_TYPE):
                    if sig.name not in default_names:
                        existing.append((normed(sig.name), sig))
                else:
                    assert isinstance(sig, (tuple, list))
                    if sig[0] not in default_names:
                        sig_obj = sigcls(sig[0], sig[1])
                        existing.append((normed(sig[0]), sig_obj))

                        # New-style-decorate the handler method
                        try:
                            handler_name = sig[2]
                        except IndexError:
                            pass  # Output signal; no handler
                        else:
                            assert isinstance(handler_name, str)
                            attrib[handler_name] = sig_obj(attrib[handler_name])

            # Finally, make this a new-style type
            existing = type('Inputs' if sigcls == Input else
                            'Outputs' if sigcls == Output else None,
                            (), dict(existing))

        return existing

    @classmethod
    def add_extra_attributes(cls, name, bases, attrib):
        pass


class OWBaseLearnerMeta(DefaultWidgetChannelsMetaClass):
    """Metaclass that adds default inputs (table, preprocess) and
    outputs (learner, model) for learner widgets.
    """

    REQUIRED_ATTRIBUTES = ['LEARNER']

    @classmethod
    def default_inputs(cls, attrib):
        return [('data', Input("Data", Table, default=True, _seq_id=-2)),
                ('preprocessor', Input("Preprocessor", Preprocess, _seq_id=-1))]

    @classmethod
    def default_outputs(cls, attrib):
        learner_class = attrib['LEARNER']
        model_name, replaces = (
            ('Classifier', ()) if issubclass(learner_class, LearnerClassification) else
            ('Predictor', ()) if issubclass(learner_class, LearnerRegression) else
            ('Model', ('Classifier', 'Predictor')))

        # Compat for old-style out-signals, e.g. self.send(self.OUTPUT_MODEL_NAME, model)
        attrib['OUTPUT_MODEL_NAME'] = model_name

        return [('learner', Output('Learner', learner_class, _seq_id=-2)),
                ('model', Output(model_name, learner_class.__returns__,
                                 replaces=replaces, _seq_id=-1))]

    @classmethod
    def add_extra_attributes(cls, name, bases, attrib):
        if 'learner_name' not in attrib:
            attrib['learner_name'] = Setting(attrib['name'])

        # Decorate default input handlers with associated Input signals
        attr_dicts = [attrib] + [base.__dict__ for base in bases]
        handler = lambda name: next(d[name] for d in attr_dicts if name in d)
        attrib['set_data'] = attrib['Inputs'].data(handler('set_data'))
        attrib['set_preprocessor'] = attrib['Inputs'].preprocessor(handler('set_preprocessor'))


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

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = False
        self.learner = None
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

    def set_preprocessor(self, preprocessor):
        self.preprocessors = preprocessor
        self.apply()

    @check_sql_input
    def set_data(self, data):
        """Set the input train data set."""
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
                self.Error.data_error("Data set is empty.")
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

    def _change_name(self, instance, signal):
        if instance:
            instance.name = self.learner_name
            if self.auto_apply:
                signal.send(instance)

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
        box = gui.hBox(self.controlArea, True)
        box.layout().addWidget(self.report_button)
        gui.separator(box, 15)
        self.apply_button = gui.auto_commit(box, self, 'auto_apply', '&Apply',
                                            box=False, commit=self.apply)
