# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import time
from unittest.mock import patch

from Orange.data import Table
from Orange.widgets.data.owcorrelations import (
    OWCorrelations, KMeansCorrelationHeuristic
)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from Orange.widgets.widget import AttributeList


class TestOWCorrelations(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_cont = Table("iris")
        cls.data_disc = Table("zoo")
        cls.data_mixed = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWCorrelations)

    def test_input_data_cont(self):
        """Check correlation table for dataset with continuous attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        n_attrs = len(self.data_cont.domain.attributes)
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 2)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 0)

    def test_input_data_disc(self):
        """Check correlation table for dataset with discrete attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_disc)
        self.assertTrue(self.widget.Information.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.not_enough_vars.is_shown())

    def test_input_data_mixed(self):
        """Check correlation table for dataset with continuous and discrete
        attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_mixed)
        domain = self.data_mixed.domain
        n_attrs = len([a for a in domain.attributes if a.is_continuous])
        time.sleep(0.1)
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 2)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)

    def test_input_data_one_feature(self):
        """Check correlation table for dataset with one attribute"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:, [0, 4]])
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Information.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.not_enough_vars.is_shown())

    def test_input_data_one_instance(self):
        """Check correlation table for dataset with one instance"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:1])
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Information.not_enough_inst.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.not_enough_inst.is_shown())

    def test_output_data(self):
        """Check dataset on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        self.process_events()
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(self.data_cont, output)

    def test_output_features(self):
        """Check features on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        self.process_events()
        attrs = self.widget.cont_data.domain.attributes
        self.widget._vizrank_selection_changed(attrs[0], attrs[1])
        features = self.get_output(self.widget.Outputs.features)
        self.assertIsInstance(features, AttributeList)
        self.assertEqual(len(features), 2)

    def test_output_correlations(self):
        """Check correlation table on on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        self.process_events()
        self.widget.commit()
        correlations = self.get_output(self.widget.Outputs.correlations)
        self.assertIsInstance(correlations, Table)
        self.assertEqual(len(correlations), 6)
        self.assertEqual(len(correlations.domain.attributes), 1)
        self.assertEqual(len(correlations.domain.metas), 2)

    def test_scatterplot_input_features(self):
        """Check if attributes have been set after sent to scatterplot"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        spw = self.create_widget(OWScatterPlot)
        attrs = self.widget.cont_data.domain.attributes
        self.widget._vizrank_selection_changed(attrs[2], attrs[3])
        features = self.get_output(self.widget.Outputs.features)
        self.send_signal(self.widget.Inputs.data, self.data_cont, widget=spw)
        self.send_signal(spw.Inputs.features, features, widget=spw)
        self.assertIs(spw.attr_x, self.data_cont.domain[2])
        self.assertIs(spw.attr_y, self.data_cont.domain[3])

    def test_heuristic(self):
        """Check attribute pairs got by heuristic"""
        heuristic = KMeansCorrelationHeuristic(self.data_cont)
        heuristic.n_clusters = 2
        self.assertListEqual(list(heuristic.get_states(None)),
                             [(0, 2), (0, 3), (2, 3)])

    def test_heuristic_get_states(self):
        """Check attribute pairs after the widget has been paused"""
        heuristic = KMeansCorrelationHeuristic(self.data_cont)
        heuristic.n_clusters = 2
        states = heuristic.get_states(None)
        _ = next(states)
        self.assertListEqual(list(heuristic.get_states(next(states))),
                             [(0, 3), (2, 3)])

    def test_correlation_type(self):
        c_type = self.widget.controls.correlation_type
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        self.process_events()
        self.widget.commit()
        pearson_corr = self.get_output(self.widget.Outputs.correlations)

        simulate.combobox_activate_item(c_type, "Spearman correlation")
        time.sleep(0.1)
        self.process_events()
        self.widget.commit()
        sperman_corr = self.get_output(self.widget.Outputs.correlations)
        self.assertFalse((pearson_corr.X == sperman_corr.X).all())

    @patch("Orange.widgets.data.owcorrelations.SIZE_LIMIT", 2000)
    @patch("Orange.widgets.data.owcorrelations."
           "KMeansCorrelationHeuristic.n_clusters", 2)
    def test_vizrank_use_heuristic(self):
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        time.sleep(0.1)
        self.process_events()
        self.widget.commit()

    def test_send_report(self):
        """Test report """
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()
