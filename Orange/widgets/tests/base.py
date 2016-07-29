import unittest
from collections import namedtuple

from PyQt4.QtGui import QApplication
import sip

from Orange.data import Table
from Orange.preprocess import RemoveNaNColumns, Randomize
from Orange.preprocess.preprocess import PreprocessorList
from Orange.classification.base_classification import (LearnerClassification,
                                                       ModelClassification)
from Orange.regression.base_regression import LearnerRegression, ModelRegression
from Orange.canvas.report.owreport import OWReport
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner

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

    def create_widget(self, cls, stored_settings=None):
        """Create a widget instance.

        Parameters
        ----------
        cls : WidgetMetaClass
            Widget class to instantiate
        stored_settings : dict
            Default values for settings

        Returns
        -------
        Widget instance : cls
        """
        widget = cls.__new__(cls, signal_manager=self.signal_manager,
                             stored_settings=stored_settings)
        widget.__init__()
        self.process_events()
        self.widgets.append(widget)
        return widget

    @staticmethod
    def process_events():
        """Process Qt events.

        Needs to be called manually as QApplication.exec is never called.
        """
        app.processEvents()

    def send_signal(self, input_name, value, id=None, widget=None):
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
                getattr(widget, input_signal.handler)(value)
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


GuiToParam = namedtuple('GuiParam', 'name gui_el get set values set_values')


class WidgetLearnerTestMixin:
    """Base class for widget learner tests.

    Contains init method to set up testing parameters and test methods.

    All widget learner tests should extend it (beside extending WidgetTest
    class as well). Learners with extra parameters, which can be set on the
    widget, should override self.gui_to_params list in the setUp method. The
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
        self.gui_to_params = []

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
        error = self.widget.Error.data_error
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
        self.assertIsNone(self.get_output("Learner"))
        self.widget.apply_button.button.click()
        self.assertIsNotNone(self.get_output("Learner"))
        self.assertIsInstance(self.get_output("Learner"), self.widget.LEARNER)

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
            for element in self.gui_to_params:
                self.assertEqual(learner_params.get(element.name),
                                 element.get(element.gui_el))

    def test_parameters(self):
        """Check learner and model for various values of all parameters"""
        for element in self.gui_to_params:
            for val, set_val in zip(element.values, element.set_values):
                self.send_signal("Data", self.data)
                element.set(set_val, element.gui_el)
                self.widget.apply_button.button.click()
                param = self.widget.learner.params.get(element.name)
                self.assertEqual(param, element.get(element.gui_el))
                self.assertEqual(param, val)
                param = self.get_output("Learner").params.get(element.name)
                self.assertEqual(param, val)
                model = self.get_output(self.model_name)
                if model is not None:
                    self.assertEqual(model.params.get(element.name), val)
                else:
                    self.assertTrue(self.widget.Error.active)
