from PyQt4 import QtGui

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans

from Orange.data import Table
class TestOWKMeans(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWKMeans,
                                         stored_settings={"auto_apply": False})
        self.iris = Table("iris")

    def test_optimization_report_display(self):
        """ Check visibility of optimization report after we select the number of clusters  """
        self.send_signal("Data", self.iris)
        radio_buttons = self.widget.n_clusters.findChildren(QtGui.QRadioButton)
        radio_buttons[0].click()
        self.assertEqual(self.widget.mainArea.isHidden(), True)
        radio_buttons[1].click()
        self.assertEqual(self.widget.mainArea.isHidden(), False)
        self.widget.apply_button.button.click()
        self.assertEqual(self.widget.table_model.rowCount() > 0, True)

    def test_data_on_output(self):
        """ Check if data is on output after create widget and run """
        # Connect iris to widget
        self.send_signal("Data", self.iris)
        self.widget.apply_button.button.click()
        self.assertNotEqual(self.widget.data, None)
        # Disconnect the data
        self.send_signal("Data", None)
        # removing data should have cleared the output
        self.assertEqual(self.widget.data, None)
