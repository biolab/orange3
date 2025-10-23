# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access, unsubscriptable-object
import unittest
from unittest.mock import patch, Mock

import numpy as np
import numpy.testing as npt

from AnyQt.QtCore import Qt
from scipy.stats import pearsonr

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.tests import test_filename
from Orange.widgets.data.owcorrelations import (
    OWCorrelations, KMeansCorrelationHeuristic, CorrelationRank,
    CorrelationType, mock_data
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
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(), 0)

    def test_input_data_disc(self):
        """Check correlation table for dataset with discrete attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_disc)
        self.assertTrue(self.widget.Error.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_vars.is_shown())

    def test_input_data_mixed(self):
        """Check correlation table for dataset with continuous and discrete
        attributes"""
        self.send_signal(self.widget.Inputs.data, self.data_mixed)
        domain = self.data_mixed.domain
        n_attrs = len([a for a in domain.attributes if a.is_continuous])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 4)
        self.assertEqual(self.widget.vizrank.rank_model.rowCount(),
                         n_attrs * (n_attrs - 1) / 2)

    def test_input_data_one_feature(self):
        """Check correlation table for dataset with one attribute"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:, [0, 4]])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertTrue(self.widget.Error.not_enough_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_vars.is_shown())

    def test_input_data_one_instance(self):
        """Check correlation table for dataset with one instance"""
        self.send_signal(self.widget.Inputs.data, self.data_cont[:1])
        self.wait_until_finished()
        self.process_events()
        self.assertEqual(self.widget.vizrank.rank_model.columnCount(), 0)
        self.assertFalse(self.widget.Information.removed_cons_feat.is_shown())
        self.assertTrue(self.widget.Error.not_enough_inst.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_inst.is_shown())

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
        self.assertTrue(self.widget.Error.not_enough_vars.is_shown())
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
        self.assertTrue(self.widget.Error.not_enough_vars.is_shown())

    def test_feature_model_imputation(self):
        data = mock_data()
        attributes = data.domain.attributes[:-1]
        class_var = data.domain.attributes[-1]
        data = data.transform(Domain(attributes, class_var))
        self.send_signal(data)
        assert self.widget.feature is not class_var, "No imputation?"
        self.assertEqual(self.widget.feature.name, class_var.name)

        self.widget.feature = self.widget.actual_data.domain.attributes[-1]
        assert self.widget.feature is not attributes[-1], "No imputation?"
        self.assertEqual(self.widget.feature.name, attributes[-1].name)

    def test_imputation(self):
        self.send_signal(mock_data())
        self.wait_until_finished()
        self.process_events()
        s = 1 / 2
        t = 1 / 3
        exp = np.array(
            [[1, 0, 0, 1, 0, 0],  # d 0
             [0, 1, 1, 0, 1, 1],  # e 1
             [1, 0, 0, 1, 0, 0],  # f 2
             [1, 0, s, 1, 0, s],  # g 3
             [1, 0, s, 1, 0, s],  # h 4
             [t, 0, t, 1, 0, t],  # i 5
             [0, s, s, s, s, 1]]  # j 6
        )
        names = "defghij"
        npt.assert_almost_equal(self.widget.actual_data.X, exp.T)

        model = self.widget.vizrank.rank_model
        ind = model.index
        for i in range(model.rowCount()):
            pair = [var.name
                    for var in ind(i, 0).data(CorrelationRank._AttrRole)]
            r, _ = pearsonr(*(exp[names.index(name)] for name in pair))
            self.assertAlmostEqual(
                ind(i, 0).data(CorrelationRank.CorrRole), r,
                places=7,
                msg=f"Mismatch for {pair}")

    def test_no_imputation(self):
        data = mock_data()
        self.send_signal(data)
        self.assertFalse(np.any(np.isnan(self.widget.actual_data.X)))

        self.widget.controls.impute_missing.setChecked(False)
        npt.assert_almost_equal(self.widget.actual_data.X, data.X[:, 1:])

        self.widget.controls.impute_missing.setChecked(True)
        self.assertFalse(np.any(np.isnan(self.widget.actual_data.X)))

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
        self.assertListEqual(["Correlation", "uncorrected p", "FDR"],
                             [m.name for m in correlations.domain.attributes])
        array = np.array(
            [[ 9.62757097e-01, 5.77666099e-86, 3.46599659e-85],
             [ 8.71754157e-01, 1.03845406e-47, 3.11536219e-47],
             [ 8.17953633e-01, 2.31484915e-37, 4.62969830e-37],
             [-4.20516096e-01, 8.42936639e-08, 1.26440496e-07],
             [-3.56544090e-01, 7.52389096e-06, 9.02866915e-06],
             [-1.09369250e-01, 1.82765215e-01, 1.82765215e-01]])
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

    def test_compute_score_iris(self):
        self.vizrank.master = Mock()
        self.vizrank.master.actual_data = self.iris
        self.vizrank.master.correlation_type = CorrelationType.PEARSON
        npt.assert_almost_equal(self.vizrank.compute_score((1, 0)),
                                [-0.1094, -0.1094, 0.1828], 4)

    def test_compute_score_nans(self):
        self.vizrank.master = Mock()
        data = mock_data()[:, 1:]
        self.vizrank.master.actual_data = data
        self.vizrank.attrs = data.domain.attributes
        self.vizrank.master.correlation_type = CorrelationType.PEARSON
        npt.assert_almost_equal(
            self.vizrank.compute_score((1, 0)), [-1, -1, 0])
        npt.assert_almost_equal(
            self.vizrank.compute_score((2, 0)), [-1, 1, 0])

        col0, col3 = data.X[:, 0], data.X[:, 3]
        r, p = pearsonr(col0, col3)
        npt.assert_almost_equal(
            self.vizrank.compute_score((3, 0)), [-abs(r), r, p])

        npt.assert_almost_equal(
            self.vizrank.compute_score((4, 0)), [-1, 1, 0])

        npt.assert_almost_equal(
            self.vizrank.compute_score((5, 4)), [-1, 1, 0])

        # Test that we return inf and nan if there are less than 2 values
        npt.assert_almost_equal(
            self.vizrank.compute_score((6, 4)), [np.inf, np.nan, np.nan])

        npt.assert_almost_equal(
            self.vizrank.compute_score((6, 5)), [np.inf, np.nan, np.nan])

        # I suppose that p-value is 1 because we have only two samples,
        # and B(0, 0) = 1. This is good because it distinguishes this case
        # from the one above
        npt.assert_almost_equal(
            self.vizrank.compute_score((6, 0)), [-1, -1, 1])


    def test_row_for_state(self):
        row = self.vizrank.row_for_state((-0.2, 0.2, 0.1), (1, 0))
        self.assertEqual(row[0].data(Qt.DisplayRole), "+0.200")
        self.assertEqual(row[0].data(CorrelationRank.PValRole), 0.1)
        self.assertEqual(row[1].data(Qt.DisplayRole), self.attrs[0].name)
        self.assertEqual(row[3].data(Qt.DisplayRole), self.attrs[1].name)

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
        self.assertListEqual([[0, 3, 5], [1, 2, 6, 7], [4, 8]],
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

    def test_impute_means(self):
        arr = np.array([[1, 2, np.nan, 3, 4, np.nan],
                        [5, 7, np.nan, np.nan, np.nan, np.nan],
                        [1, 2, 3, 4, 5, 6],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        exp = np.array([[1, 2, 2.5, 3, 4, 2.5],
                        [5, 7, 6, 6, 6, 6],
                        [1, 2, 3, 4, 5, 6],
                        [0, 0, 0, 0, 0, 0]])
        KMeansCorrelationHeuristic._impute_means(arr)
        np.testing.assert_almost_equal(arr, exp)

    def test_nans(self):
        data = Table("iris")
        with data.unlocked(data.X):
            data.X[0, 0] = np.nan
            data.X[:, 1] = np.nan
        heuristic = KMeansCorrelationHeuristic(data)
        clusters = heuristic.get_clusters_of_attributes()
        result = sorted([c.instances for c in clusters])
        self.assertListEqual(result, [[0, 2, 3], [1]]),


if __name__ == "__main__":
    unittest.main()
