from unittest.mock import patch

from AnyQt.QtWidgets import QRadioButton

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans
import Orange.clustering

from Orange.data import Table
class TestOWKMeans(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(
            OWKMeans, stored_settings={"auto_apply": False})  # type: OWKMeans
        self.iris = Table("iris")

    def test_optimization_report_display(self):
        """ Check visibility of optimization report after we select the number of clusters  """
        self.send_signal("Data", self.iris)
        radio_buttons = self.widget.n_clusters.findChildren(QRadioButton)
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

    class KMeansFail(Orange.clustering.KMeans):
        fail_on = set()

        def fit(self, *args):
            # when not optimizing, params is empty?!
            k = self.params.get("n_clusters", 3)
            if k in self.fail_on:
                raise ValueError("k={} fails".format(k))
            return super().fit(*args)

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_optimization_fails(self):
        widget = self.widget
        widget.k_from = 3
        widget.k_to = 8
        widget.scoring = 0
        widget.optimize_k = True

        self.KMeansFail.fail_on = {3, 5, 7}
        self.send_signal("Data", self.iris)
        self.assertIsInstance(widget.optimization_runs[0][1], str)
        self.assertIsInstance(widget.optimization_runs[2][1], str)
        self.assertIsInstance(widget.optimization_runs[4][1], str)
        self.assertNotIsInstance(widget.optimization_runs[1][1], str)
        self.assertNotIsInstance(widget.optimization_runs[3][1], str)
        self.assertNotIsInstance(widget.optimization_runs[5][1], str)
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 1)
        self.assertIsNotNone(self.get_output("Annotated Data"))

        self.KMeansFail.fail_on = set(range(3, 9))
        widget.run()
        self.assertTrue(widget.Error.failed.is_shown())
        self.assertEqual(widget.optimization_runs, [])
        self.assertIsNone(self.get_output("Annotated Data"))

        self.KMeansFail.fail_on = set()
        widget.run()
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 0)
        self.assertIsNotNone(self.get_output("Annotated Data"))

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_run_fails(self):
        self.widget.k = 3
        self.widget.optimize_k = False
        self.KMeansFail.fail_on = {3}
        self.send_signal("Data", self.iris)
        self.assertTrue(self.widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output("Annotated Data"))

        self.KMeansFail.fail_on = set()
        self.widget.run()
        self.assertFalse(self.widget.Error.failed.is_shown())
        self.assertIsNotNone(self.get_output("Annotated Data"))
