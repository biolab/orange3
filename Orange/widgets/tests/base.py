import unittest

from PyQt4.QtGui import QApplication
import sip

from Orange.canvas.report.owreport import OWReport

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
