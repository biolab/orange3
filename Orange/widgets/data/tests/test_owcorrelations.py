# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access, unsubscriptable-object
import unittest
from unittest.mock import patch, Mock

import numpy as np
import numpy.testing as npt

from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.tests import test_filename
from Orange.widgets.data.owcorrelations import (
    OWCorrelations, KMeansCorrelationHeuristic, CorrelationRank,
    CorrelationType
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
        cls.housing = Table("housing")

    def setUp(self):
        self.widget = self.create_widget(OWCorrelations)

    def test_input_data_cont(self):
        """Check correlation table for dataset with continuous attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        n_attrs = len(self.data_cont.domain.attributes)
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 3)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 0)

    def test_input_data_disc(self):
        """Check correlation table for dataset with discrete attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_disc)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.not_enough_vars.is_shown())

    def test_input_data_mixed(self):
        """Check correlation table for dataset with continuous and discrete
        attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_mixed)
        domain = self.data_mixed.domain
        n_attrs = len([a for a in domain.attributes if a.is_continuous])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 3)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)

    def test_input_data_one_feature(self):
        """Check correlation table for dataset with one attribute"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:, [0, 4]])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.not_enough_vars.is_shown())

    def test_input_data_one_instance(self):
        """Check correlation table for dataset with one instance"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:1])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())
        self.assertTrue(self.widget.Warning.not_enough_inst.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.not_enough_inst.is_shown())

    def test_input_data_with_constant_features(self):
        """Check correlation table for dataset with constant columns"""
        np.random.seed(0)
        # pylint: disable=no-member
        X = np.random.randint(3, size=(4, 3)).astype(float)
        X[:, 2] = 1

        domain = Domain([ContinuousVariable("c1"), ContinuousVariable("c2"),
                         DiscreteVariable("d1")])
        self.send_signal(self.widget.Inputs.data, Table(domain, X))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 1)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

        domain = Domain([ContinuousVariable(str(i)) for i in range(3)])
        self.send_signal(self.widget.Inputs.data, Table(domain, X))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 1)
        self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

        X = np.ones((4, 3), dtype=float)
        self.send_signal(self.widget.Inputs.data, Table(domain, X))
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())
        self.assertTrue(self.widget.Information.removed_cons_feat.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())

    def test_input_data_cont_target(self):
        """Check correlation table for dataset with continuous class variable"""
        data = self.housing[:5, 11:]
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 2)
        self.assertEqual(self.widget.controls.feature.count(), 4)
        self.assertEqual(self.widget.controls.feature.currentText(), "MEDV")

        data = self.housing[:5, 13:]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.not_enough_vars.is_shown())

    def test_output_data(self):
        """Check dataset on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(self.data_cont, output)

    def test_output_features(self):
        """Check features on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        features = self.get_output(self.widget.Outputs.features)
        self.assertIsInstance(features, AttributeList)
        self.assertEqual(len(features), 2)

    def test_output_correlations(self):
        """Check correlation table on on output"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        correlations = self.get_output(self.widget.Outputs.correlations)
        self.assertIsInstance(correlations, Table)
        self.assertEqual(len(correlations), 6)
        self.assertEqual(len(correlations.domain.metas), 2)
        self.assertListEqual(["Correlation", "FDR"],
                             [m.name for m in correlations.domain.attributes])
        array = np.array([[0.963, 0], [0.872, 0], [0.818, 0], [-0.421, 0],
                          [-0.357, 0.000009], [-0.109, 0.1827652]])
        npt.assert_almost_equal(correlations.X, array)

    def test_input_changed(self):
        """Check whether changing input emits commit"""
        self.widget.commit = Mock()
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        self.widget.commit.assert_called_once()

        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data_mixed)
        self.wait_until_finished()
        self.process_events()
        self.widget.commit.assert_called_once()

    def test_saved_selection(self):
        """Select row from settings"""
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        attrs = self.widget.cont_data.domain.attributes
        self.widget._vizrank_selection_changed(attrs[3], attrs[1])
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWCorrelations, stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.data_cont, widget=w)
        self.wait_until_finished(w)
        self.process_events()
        sel_row = w.vizrank.rank_table.selectionModel().selectedRows()[0].row()
        self.assertEqual(sel_row, 4)

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
                             [(0, 2), (0, 3), (2, 3), (0, 1), (1, 2), (1, 3)])

    def test_heuristic_get_states(self):
        """Check attribute pairs after the widget has been paused"""
        heuristic = KMeansCorrelationHeuristic(self.data_cont)
        heuristic.n_clusters = 2
        states = heuristic.get_states(None)
        _ = next(states)
        self.assertListEqual(list(heuristic.get_states(next(states))),
                             [(0, 3), (2, 3), (0, 1), (1, 2), (1, 3)])

    def test_correlation_type(self):
        c_type = self.widget.controls.correlation_type
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        pearson_corr = self.get_output(self.widget.Outputs.correlations)

        simulate.combobox_activate_item(c_type, "Spearman correlation")
        self.wait_until_finished()
        self.process_events()
        sperman_corr = self.get_output(self.widget.Outputs.correlations)
        self.assertFalse((pearson_corr.X == sperman_corr.X).all())

    def test_feature_combo(self):
        """Check content of feature selection combobox"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.data_mixed)
        cont_attributes = [attr for attr in self.data_mixed.domain.attributes
                           if attr.is_continuous]
        self.assertEqual(len(feature_combo.model()), len(cont_attributes) + 1)

        self.wait_until_stop_blocking()
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(len(feature_combo.model()), 15)

    def test_select_feature(self):
        """Test feature selection"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)
        self.assertListEqual(["petal length", "petal width"],
                             [a.name for a in self.get_output(
                                 self.widget.Outputs.features)])

        simulate.combobox_activate_index(feature_combo, 1)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 3)
        self.assertListEqual(["petal length", "sepal length"],
                             [a.name for a in self.get_output(
                                 self.widget.Outputs.features)])

        simulate.combobox_activate_index(feature_combo, 0)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)
        self.assertListEqual(["petal length", "sepal length"],
                             [a.name for a in self.get_output(
                                 self.widget.Outputs.features)])

    @patch("Orange.widgets.data.owcorrelations.SIZE_LIMIT", 2000)
    def test_vizrank_use_heuristic(self):
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.wait_until_finished()
        self.process_events()
        self.assertTrue(self.widget.vizrank.use_heuristic)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 6)

    @patch("Orange.widgets.data.owcorrelations.SIZE_LIMIT", 2000)
    def test_select_feature_against_heuristic(self):
        """Never use heuristic if feature is selected"""
        feature_combo = self.widget.controls.feature
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        simulate.combobox_activate_index(feature_combo, 2)
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 3)

    def test_send_report(self):
        """Test report """
        self.send_signal(self.widget.Inputs.data, self.data_cont)
        self.widget.report_button.click()
        self.wait_until_stop_blocking()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()


class TestCorrelationRank(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.attrs = cls.iris.domain.attributes

    def setUp(self):
        self.vizrank = CorrelationRank(None)
        self.vizrank.attrs = self.attrs

    def test_compute_score(self):
        self.vizrank.master = Mock()
        self.vizrank.master.cont_data = self.iris
        self.vizrank.master.correlation_type = CorrelationType.PEARSON
        npt.assert_almost_equal(self.vizrank.compute_score((1, 0)),
                                [-0.1094, -0.1094, 0.1828], 4)

    def test_row_for_state(self):
        row = self.vizrank.row_for_state((-0.2, 0.2, 0.1), (1, 0))
        self.assertEqual(row[0].data(Qt.DisplayRole), "+0.200")
        self.assertEqual(row[0].data(CorrelationRank.PValRole), 0.1)
        self.assertEqual(row[1].data(Qt.DisplayRole), self.attrs[0].name)
        self.assertEqual(row[2].data(Qt.DisplayRole), self.attrs[1].name)

    def test_iterate_states(self):
        self.assertListEqual(list(self.vizrank.iterate_states(None)),
                             [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
        self.assertListEqual(list(self.vizrank.iterate_states((1, 0))),
                             [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
        self.assertListEqual(list(self.vizrank.iterate_states((2, 1))),
                             [(2, 1), (3, 0), (3, 1), (3, 2)])

    def test_iterate_states_by_feature(self):
        self.vizrank.sel_feature_index = 2
        states = self.vizrank.iterate_states_by_feature()
        self.assertListEqual([(2, 0), (2, 1), (2, 3)], list(states))

    def test_state_count(self):
        self.assertEqual(self.vizrank.state_count(), 6)
        self.vizrank.sel_feature_index = 2
        self.assertEqual(self.vizrank.state_count(), 3)


class TestKMeansCorrelationHeuristic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table(test_filename("datasets/breast-cancer-wisconsin"))
        cls.heuristic = KMeansCorrelationHeuristic(cls.data)

    def test_n_clusters(self):
        self.assertEqual(self.heuristic.n_clusters, 3)

    def test_get_clusters_of_attributes(self):
        clusters = self.heuristic.get_clusters_of_attributes()
        # results depend on scikit-learn k-means implementation
        result = sorted([c.instances for c in clusters])
        self.assertListEqual([[0], [1, 2, 3, 4, 5, 6, 7], [8]],
                             result)

    def test_get_states(self):
        n_attrs = len(self.data.domain.attributes)
        states = set(self.heuristic.get_states(None))
        self.assertEqual(len(states), n_attrs * (n_attrs - 1) / 2)
        self.assertSetEqual(set((min(i, j), max(i, j)) for i in
                                range(n_attrs) for j in range(i)), states)

    def test_get_states_one_cluster(self):
        heuristic = KMeansCorrelationHeuristic(Table("iris")[:, :2])
        states = set(heuristic.get_states(None))
        self.assertEqual(len(states), 1)
        self.assertSetEqual(states, {(0, 1)})


if __name__ == "__main__":
    unittest.main()
