import unittest

import numpy as np

from Orange.base import SklLearner, SklModel
from AnyQt.QtWidgets import (
    QApplication, QComboBox, QSpinBox, QDoubleSpinBox, QSlider
)
import sip

from Orange.data import Table
from Orange.preprocess import RemoveNaNColumns, Randomize
from Orange.preprocess.preprocess import PreprocessorList
from Orange.classification.base_classification import (LearnerClassification,
                                                       ModelClassification)
from Orange.regression.base_regression import LearnerRegression, ModelRegression
from Orange.canvas.report.owreport import OWReport
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.annotated_data import (ANNOTATED_DATA_FEATURE_NAME,
                                                 ANNOTATED_DATA_SIGNAL_NAME)

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

    All widgets should be created by the create_widget method, as it
    remembers created widgets and properly destroys them in tearDownClass
    to avoid segmentation faults when QApplication gets destroyed.
    """

    #: list[OwWidget]
    widgets = []

    @classmethod
    def setUpClass(cls):
        """Prepare environment for test execution

        Prepare a list for tracking created widgets and construct a
        dummy signal manager. Monkey patch OWReport.get_instance to
        return a manually_created instance.
        """
        super().setUpClass()

        cls.widgets = []

        cls.signal_manager = DummySignalManager()

        report = OWReport()
        cls.widgets.append(report)
        OWReport.get_instance = lambda: report

    @classmethod
    def tearDownClass(cls):
        """Cleanup after tests

        Process any pending events and properly destroy created widgets by
        calling their onDeleteWidget method which does the widget-specific
        cleanup.

        NOTE: sip.delete is mandatory. In some cases, widgets are deleted by
        python while some references in QApplication remain
        (QApplication::topLevelWidgets()), causing a segmentation fault when
        QApplication is destroyed.
        """
        app.processEvents()
        for w in cls.widgets:
            w.onDeleteWidget()
            sip.delete(w)

    def create_widget(self, cls, stored_settings=None, reset_default_settings=True):
        """Create a widget instance.

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

    def __init__(self, name, gui_element, values, getter, setter):
        self.name = name
        self.gui_element = gui_element
        self.values = values
        self.get_value = getter
        self.set_value = setter


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
                 getter=None, setter=None):
        super().__init__(name, gui_element,
                         values or self._default_values(gui_element),
                         getter or self._default_get_value(gui_element, values),
                         setter or self._default_set_value(gui_element, values))

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
        self.iris = Table("iris")
        self.housing = Table("housing")
        if issubclass(self.widget.LEARNER, LearnerClassification):
            self.data = self.iris
            self.inadequate_data = self.housing
            self.learner_class = LearnerClassification
            self.model_name = "Classifier"
            self.model_class = ModelClassification
        else:
            self.data = self.housing
            self.inadequate_data = self.iris
            self.learner_class = LearnerRegression
            self.model_name = "Predictor"
            self.model_class = ModelRegression
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
        self.send_signal("Data", self.inadequate_data)
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Error.data_error.is_shown())
        self.send_signal("Data", self.data)
        self.assertFalse(self.widget.Error.data_error.is_shown())

    def test_input_preprocessor(self):
        """Check learner's preprocessors with an extra pp on input"""
        self.assertIsNone(self.widget.preprocessors)
        self.send_signal("Preprocessor", Randomize)
        pp = (Randomize,) + tuple(self.widget.LEARNER.preprocessors)
        self.assertEqual(pp, self.widget.preprocessors)
        self.widget.apply_button.button.click()
        self.assertEqual(pp, tuple(self.widget.learner.preprocessors))

    def test_input_preprocessors(self):
        """Check multiple preprocessors on input"""
        pp_list = PreprocessorList([Randomize, RemoveNaNColumns])
        self.send_signal("Preprocessor", pp_list)
        self.widget.apply_button.button.click()
        self.assertEqual([pp_list] + list(self.widget.LEARNER.preprocessors),
                         self.widget.learner.preprocessors)

    def test_input_preprocessor_disconnect(self):
        """Check learner's preprocessors after disconnecting pp from input"""
        self.send_signal("Preprocessor", Randomize)
        self.assertIsNotNone(self.widget.preprocessors)
        self.send_signal("Preprocessor", None)
        self.assertEqual(self.widget.preprocessors,
                         tuple(self.widget.LEARNER.preprocessors))

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
        self.send_signal("Data", self.data)
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

    def test_parameters_default(self):
        """Check if learner's parameters are set to default (widget's) values"""
        self.widget.apply_button.button.click()
        if hasattr(self.widget.learner, "params"):
            learner_params = self.widget.learner.params
            for parameter in self.parameters:
                self.assertEqual(learner_params.get(parameter.name),
                                 parameter.get_value())

    def test_parameters(self):
        def get_value(learner, name):
            # Handle SKL and skl-like learners, and non-SKL learners
            if hasattr(learner, "params"):
                return learner.params.get(name)
            else:
                return getattr(learner, name)

        """Check learner and model for various values of all parameters"""
        for parameter in self.parameters:
            assert isinstance(parameter, BaseParameterMapping)
            for value in parameter.values:
                self.send_signal("Data", self.data)
                parameter.set_value(value)
                self.widget.apply_button.button.click()
                param = get_value(self.widget.learner, parameter.name)
                self.assertEqual(param, parameter.get_value(),
                                 "Mismatching setting for parameter '{}'".
                                 format(parameter.name))
                self.assertEqual(param, value,
                                 "Mismatching setting for parameter '{}'".
                                 format(parameter.name))
                param = get_value(self.get_output("Learner"), parameter.name)
                self.assertEqual(param, value,
                                 "Mismatching setting for parameter '{}'".
                                 format(parameter.name))
                if issubclass(self.widget.LEARNER, SklModel):
                    model = self.get_output(self.model_name)
                    if model is not None:
                        self.assertEqual(get_value(model, parameter.name), value)
                        self.assertFalse(self.widget.Error.active)
                    else:
                        self.assertTrue(self.widget.Error.active)


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
