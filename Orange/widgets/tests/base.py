import os
import sip
import unittest
from unittest.mock import Mock

import numpy as np
from AnyQt.QtWidgets import (
    QApplication, QComboBox, QSpinBox, QDoubleSpinBox, QSlider
)

from Orange.base import SklModel, Model
from Orange.canvas.report.owreport import OWReport
from Orange.classification.base_classification import (
    LearnerClassification, ModelClassification
)
from Orange.data import Table
from Orange.modelling import Fitter
from Orange.preprocess import RemoveNaNColumns, Randomize
from Orange.preprocess.preprocess import PreprocessorList
from Orange.regression.base_regression import (
    LearnerRegression, ModelRegression
)
from Orange.widgets.utils.annotated_data import (
    ANNOTATED_DATA_FEATURE_NAME, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

sip.setdestroyonexit(False)

app = None


class DummySignalManager:
    def __init__(self):
        self.outputs = {}

    def send(self, widget, signal_name, value, id):
        self.outputs[(widget, signal_name)] = value


class GuiTest(unittest.TestCase):
    """Base class for tests that require a QApplication instance

    GuiTest ensures that a QApplication exists before tests are run an
    """
    @classmethod
    def setUpClass(cls):
        """Prepare for test execution.

        Ensure that a (single copy of) QApplication has been created
        """
        global app
        if app is None:
            app = QApplication([])


class WidgetTest(GuiTest):
    """Base class for widget tests

    Contains helper methods widget creation and working with signals.

    All widgets should be created by the create_widget method, as this
    will ensure they are created correctly.
    """

    #: list[OwWidget]
    widgets = []

    @classmethod
    def setUpClass(cls):
        """Prepare environment for test execution

        Construct a dummy signal manager and monkey patch
        OWReport.get_instance to return a manually created instance.
        """
        super().setUpClass()

        cls.widgets = []

        cls.signal_manager = DummySignalManager()

        report = OWReport()
        cls.widgets.append(report)
        OWReport.get_instance = lambda: report

    def create_widget(self, cls, stored_settings=None, reset_default_settings=True):
        """Create a widget instance using mock signal_manager.

        When used with default parameters, it also overrides settings stored
        on disk with default defined in class.

        After widget is created, QApplication.process_events is called to
        allow any singleShot timers defined in __init__ to execute.

        Parameters
        ----------
        cls : WidgetMetaClass
            Widget class to instantiate
        stored_settings : dict
            Default values for settings
        reset_default_settings : bool
            If set, widget will start with default values for settings,
            if not, values accumulated through the session will be used

        Returns
        -------
        Widget instance : cls
        """
        if reset_default_settings:
            self.reset_default_settings(cls)
        widget = cls.__new__(cls, signal_manager=self.signal_manager,
                             stored_settings=stored_settings)
        widget.__init__()
        self.process_events()
        self.widgets.append(widget)
        return widget

    @staticmethod
    def reset_default_settings(cls):
        """Reset default setting values for widget

        Discards settings read from disk and changes stored by fast_save

        Parameters
        ----------
        cls : OWWidget
            widget to reset settings for
        """
        settings_handler = getattr(cls, "settingsHandler", None)
        if settings_handler:
            # Rebind settings handler to get fresh copies of settings
            # in known_settings
            settings_handler.bind(cls)
            # Reset defaults read from disk
            settings_handler.defaults = {}
            # Reset context settings
            settings_handler.global_contexts = []

    @staticmethod
    def process_events():
        """Process Qt events.

        Needs to be called manually as QApplication.exec is never called.
        """
        app.processEvents()

    def show(self, widget=None):
        """Show widget in interactive mode.

        Useful for debugging tests, as widget can be inspected manually.
        """
        widget = widget or self.widget
        widget.show()
        app.exec()

    def send_signal(self, input_name, value, *args, widget=None):
        """ Send signal to widget by calling appropriate triggers.

        Parameters
        ----------
        input_name : str
        value : Object
        id : int
            channel id, used for inputs with flag Multiple
        widget : Optional[OWWidget]
            widget to send signal to. If not set, self.widget is used
        """
        if widget is None:
            widget = self.widget
        for input_signal in widget.inputs:
            if input_signal.name == input_name:
                getattr(widget, input_signal.handler)(value, *args)
                break
        else:
            raise ValueError("'{}' is not an input name for widget {}"
                             .format(input_name, type(widget).__name__))
        widget.handleNewSignals()

    def get_output(self, output_name, widget=None):
        """Return the last output that has been sent from the widget.

        Parameters
        ----------
        output_name : str
        widget : Optional[OWWidget]
            widget whose output is returned. If not set, self.widget is used

        Returns
        -------
        The last sent value of given output or None if nothing has been sent.
        """
        if widget is None:
            widget = self.widget
        return self.signal_manager.outputs.get((widget, output_name), None)


class BaseParameterMapping:
    """Base class for mapping between gui components and learner's parameters
    when testing learner widgets.

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    gui_element : QWidget
        Gui component who's corresponding parameter is to be tested.

    values: list
        List of values to be tested.

    getter: function
        It gets component's value.

    setter: function
        It sets component's value.
    """

    def __init__(self, name, gui_element, values, getter, setter,
                 problem_type="both"):
        self.name = name
        self.gui_element = gui_element
        self.values = values
        self.get_value = getter
        self.set_value = setter
        self.problem_type = problem_type

    def __str__(self):
        if self.problem_type == "both":
            return self.name
        else:
            return "%s (%s)" % (self.name, self.problem_type)


class DefaultParameterMapping(BaseParameterMapping):
    """Class for mapping between gui components and learner's parameters
    when testing unchecked properties and therefore default parameters
    should be used.

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    default_value: str, int,
        Value that should be used by default.
    """

    def __init__(self, name, default_value):
        super().__init__(name, None, [default_value],
                         lambda: default_value, lambda x: None)


class ParameterMapping(BaseParameterMapping):
    """Class for mapping between gui components and learner parameters
    when testing learner widgets

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    gui_element : QWidget
        Gui component who's corresponding parameter is to be tested.

    values: list, mandatory for ComboBox, optional otherwise
        List of values to be tested. When None, it is set according to
        component's type.

    getter: function, optional
        It gets component's value. When None, it is set according to
        component's type.

    setter: function, optional
        It sets component's value. When None, it is set according to
        component's type.
    """

    def __init__(self, name, gui_element, values=None,
                 getter=None, setter=None, **kwargs):
        super().__init__(
            name, gui_element,
            values or self._default_values(gui_element),
            getter or self._default_get_value(gui_element, values),
            setter or self._default_set_value(gui_element, values),
            **kwargs)

    @staticmethod
    def get_gui_element(widget, attribute):
        return widget.controlled_attributes[attribute][0].control

    @classmethod
    def from_attribute(cls, widget, attribute, parameter=None):
        return cls(parameter or attribute, cls.get_gui_element(widget, attribute))

    @staticmethod
    def _default_values(gui_element):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return [gui_element.minimum(), gui_element.maximum()]
        else:
            raise TypeError("{} is not supported".format(gui_element))

    @staticmethod
    def _default_get_value(gui_element, values):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return lambda: gui_element.value()
        elif isinstance(gui_element, QComboBox):
            return lambda: values[gui_element.currentIndex()]
        else:
            raise TypeError("{} is not supported".format(gui_element))

    @staticmethod
    def _default_set_value(gui_element, values):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return lambda val: gui_element.setValue(val)
        elif isinstance(gui_element, QComboBox):
            def fun(val):
                value = values.index(val)
                gui_element.activated.emit(value)
                gui_element.setCurrentIndex(value)

            return fun
        else:
            raise TypeError("{} is not supported".format(gui_element))


class WidgetLearnerTestMixin:
    """Base class for widget learner tests.

    Contains init method to set up testing parameters and test methods.

    All widget learner tests should extend it (beside extending WidgetTest
    class as well). Learners with extra parameters, which can be set on the
    widget, should override self.parameters list in the setUp method. The
    list should contain mapping: learner parameter - gui component.
    """

    widget = None  # type: OWBaseLearner

    def init(self):
        self.iris = iris = Table("iris")
        self.housing = housing = Table("housing")

        if issubclass(self.widget.LEARNER, Fitter):
            self.data = iris
            self.valid_datasets = (iris, housing)
            self.inadequate_dataset = ()
            self.learner_class = Fitter
            self.model_class = Model
            self.model_name = 'Model'
        elif issubclass(self.widget.LEARNER, LearnerClassification):
            self.data = iris
            self.valid_datasets = (iris,)
            self.inadequate_dataset = (housing,)
            self.learner_class = LearnerClassification
            self.model_class = ModelClassification
            self.model_name = 'Classifier'
        else:
            self.data = housing
            self.valid_datasets = (housing,)
            self.inadequate_dataset = (iris,)
            self.learner_class = LearnerRegression
            self.model_class = ModelRegression
            self.model_name = 'Predictor'

        self.parameters = []

    def test_has_unconditional_apply(self):
        self.assertTrue(hasattr(self.widget, "unconditional_apply"))

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal("Data", self.data)
        self.assertEqual(self.widget.data, self.data)

    def test_input_data_disconnect(self):
        """Check widget's data and model after disconnecting data from input"""
        self.send_signal("Data", self.data)
        self.assertEqual(self.widget.data, self.data)
        self.widget.apply_button.button.click()
        self.send_signal("Data", None)
        self.assertEqual(self.widget.data, None)
        self.assertIsNone(self.get_output(self.model_name))

    def test_input_data_learner_adequacy(self):
        """Check if error message is shown with inadequate data on input"""
        for inadequate in self.inadequate_dataset:
            self.send_signal("Data", inadequate)
            self.widget.apply_button.button.click()
            self.assertTrue(self.widget.Error.data_error.is_shown())
        for valid in self.valid_datasets:
            self.send_signal("Data", valid)
            self.assertFalse(self.widget.Error.data_error.is_shown())

    def test_input_preprocessor(self):
        """Check learner's preprocessors with an extra pp on input"""
        self.send_signal("Preprocessor", Randomize)
        self.assertEqual(
            Randomize, self.widget.preprocessors,
            'Preprocessor not added to widget preprocessors')
        self.widget.apply_button.button.click()
        self.assertEqual(
            (Randomize,), self.widget.learner.preprocessors,
            'Preprocessors were not passed to the learner')

    def test_input_preprocessors(self):
        """Check multiple preprocessors on input"""
        pp_list = PreprocessorList([Randomize, RemoveNaNColumns])
        self.send_signal("Preprocessor", pp_list)
        self.widget.apply_button.button.click()
        self.assertEqual(
            (pp_list,), self.widget.learner.preprocessors,
            '`PreprocessorList` was not added to preprocessors')

    def test_input_preprocessor_disconnect(self):
        """Check learner's preprocessors after disconnecting pp from input"""
        self.send_signal("Preprocessor", Randomize)
        self.widget.apply_button.button.click()
        self.assertEqual(Randomize, self.widget.preprocessors)

        self.send_signal("Preprocessor", None)
        self.widget.apply_button.button.click()
        self.assertIsNone(self.widget.preprocessors,
                          'Preprocessors not removed on disconnect.')

    def test_output_learner(self):
        """Check if learner is on output after apply"""
        initial = self.get_output("Learner")
        self.assertIsNotNone(initial, "Does not initialize the learner output")
        self.widget.apply_button.button.click()
        newlearner = self.get_output("Learner")
        self.assertIsNot(initial, newlearner,
                         "Does not send a new learner instance on `Apply`.")
        self.assertIsNotNone(newlearner)
        self.assertIsInstance(newlearner, self.widget.LEARNER)

    def test_output_model(self):
        """Check if model is on output after sending data and apply"""
        self.assertIsNone(self.get_output(self.model_name))
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.model_name))
        self.send_signal('Data', self.data)
        self.widget.apply_button.button.click()
        model = self.get_output(self.model_name)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, self.widget.LEARNER.__returns__)
        self.assertIsInstance(model, self.model_class)

    def test_output_learner_name(self):
        """Check if learner's name properly changes"""
        new_name = "Learner Name"
        self.widget.apply_button.button.click()
        self.assertEqual(self.widget.learner.name,
                         self.widget.name_line_edit.text())
        self.widget.name_line_edit.setText(new_name)
        self.widget.apply_button.button.click()
        self.assertEqual(self.get_output("Learner").name, new_name)

    def test_output_model_name(self):
        """Check if model's name properly changes"""
        new_name = "Model Name"
        self.widget.name_line_edit.setText(new_name)
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.assertEqual(self.get_output(self.model_name).name, new_name)

    def _get_param_value(self, learner, param):
        if isinstance(learner, Fitter):
            # Both is just a was to indicate to the tests, fitters don't
            # actually support this
            if param.problem_type == "both":
                problem_type = learner.CLASSIFICATION
            else:
                problem_type = param.problem_type
            return learner.get_params(problem_type).get(param.name)
        else:
            return learner.params.get(param.name)

    def test_parameters_default(self):
        """Check if learner's parameters are set to default (widget's) values
        """
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)
            self.widget.apply_button.button.click()
            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if self._should_check_parameter(parameter, dataset):
                    self.assertEqual(
                        self._get_param_value(self.widget.learner, parameter),
                        parameter.get_value())

    def test_parameters(self):
        """Check learner and model for various values of all parameters"""
        # Test params on every valid dataset, since some attributes may apply
        # to only certain problem types
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)

            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if not self._should_check_parameter(parameter, dataset):
                    continue

                assert isinstance(parameter, BaseParameterMapping)

                for value in parameter.values:
                    parameter.set_value(value)
                    self.widget.apply_button.button.click()
                    param = self._get_param_value(self.widget.learner, parameter)
                    self.assertEqual(
                        param, parameter.get_value(),
                        "Mismatching setting for parameter '%s'" % parameter)
                    self.assertEqual(
                        param, value,
                        "Mismatching setting for parameter '%s'" % parameter)
                    param = self._get_param_value(self.get_output("Learner"), parameter)
                    self.assertEqual(
                        param, value,
                        "Mismatching setting for parameter '%s'" % parameter)

                    if issubclass(self.widget.LEARNER, SklModel):
                        model = self.get_output(self.model_name)
                        if model is not None:
                            self.assertEqual(self._get_param_value(model, parameter), value)
                            self.assertFalse(self.widget.Error.active)
                        else:
                            self.assertTrue(self.widget.Error.active)

    def test_params_trigger_settings_changed(self):
        """Check that the learner gets updated whenever a param is changed."""
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)

            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if not self._should_check_parameter(parameter, dataset):
                    continue

                assert isinstance(parameter, BaseParameterMapping)
                # Set the mock here so we can include the param name in the
                # error message, so if any test fails, we see where
                # We mock `apply` and not `settings_changed` since that's
                # sometimes connected with Qt signals, which are not directly
                # called
                self.widget.apply = Mock(name="apply(%s)" % parameter)
                # Since the settings only get updated when the value actually
                # changes, find a value that isn't the same as the current
                # value and try with that
                new_value = [x for x in parameter.values
                             if x != parameter.get_value()][0]
                parameter.set_value(new_value)
                self.widget.apply.assert_called_once_with()

    @staticmethod
    def _should_check_parameter(parameter, data):
        """Should the param be passed into the learner given the data"""
        return ((parameter.problem_type == "classification" and
                 data.domain.has_discrete_class) or
                (parameter.problem_type == "regression" and
                 data.domain.has_continuous_class) or
                (parameter.problem_type == "both"))


class WidgetOutputsTestMixin:
    """Class for widget's outputs testing.

    Contains init method to set up testing parameters and a test method, which
    checks Selected Data and (Annotated) Data outputs.

    Since widgets have different ways of selecting data instances, _select_data
    method should be implemented when subclassed. The method should assign
    value to selected_indices parameter.

    If output's expected domain differs from input's domain, parameter
    same_input_output_domain should be set to False.

    If Selected Data and Data domains differ, override method
    _compare_selected_annotated_domains.
    """

    def init(self):
        self.data = Table("iris")
        self.same_input_output_domain = True

    def test_outputs(self):
        self.send_signal(self.signal_name, self.signal_data)

        # only needed in TestOWMDS
        if type(self).__name__ == "TestOWMDS":
            from AnyQt.QtCore import QEvent
            self.widget.customEvent(QEvent(QEvent.User))
            self.widget.commit()

        # check selected data output
        self.assertIsNone(self.get_output("Selected Data"))

        # check annotated data output
        feature_name = ANNOTATED_DATA_FEATURE_NAME
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(0, np.sum([i[feature_name] for i in annotated]))

        # select data instances
        selected_indices = self._select_data()

        # check selected data output
        selected = self.get_output("Selected Data")
        n_sel, n_attr = len(selected), len(self.data.domain.attributes)
        self.assertGreater(n_sel, 0)
        self.assertEqual(selected.domain == self.data.domain,
                         self.same_input_output_domain)
        np.testing.assert_array_equal(selected.X[:, :n_attr],
                                      self.data.X[selected_indices])
        self.assertEqual(selected.attributes, self.data.attributes)

        # check annotated data output
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(n_sel, np.sum([i[feature_name] for i in annotated]))
        self.assertEqual(annotated.attributes, self.data.attributes)

        # compare selected and annotated data domains
        self._compare_selected_annotated_domains(selected, annotated)

        # check output when data is removed
        self.send_signal(self.signal_name, None)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(ANNOTATED_DATA_SIGNAL_NAME))

    def _select_data(self):
        raise NotImplementedError("Subclasses should implement select_data")

    def _compare_selected_annotated_domains(self, selected, annotated):
        selected_vars = selected.domain.variables + selected.domain.metas
        annotated_vars = annotated.domain.variables + annotated.domain.metas
        self.assertLess(set(selected_vars), set(annotated_vars))


class datasets:
    @staticmethod
    def path(filename):
        dirname = os.path.join(os.path.dirname(__file__), "datasets")
        return os.path.join(dirname, filename)

    @classmethod
    def missing_data_1(cls):
        """
        Data set with 3 continuous features (X{1,2,3}) where all the columns
        and rows contain at least one NaN value.

        One discrete class D with NaN values
        Mixed continuous/discrete/string metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_1.tab"))

    @classmethod
    def missing_data_2(cls):
        """
        Data set with 3 continuous features (X{1,2,3}) where all the columns
        and rows contain at least one NaN value and X1, X2 are constant.

        One discrete constant class D with NaN values.
        Mixed continuous/discrete/string class metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_2.tab"))

    @classmethod
    def missing_data_3(cls):
        """
        Data set with 3 discrete features D{1,2,3} where all the columns and
        rows contain at least one NaN value

        One discrete class D with NaN values
        Mixes continuous/discrete/string metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_3.tab"))
