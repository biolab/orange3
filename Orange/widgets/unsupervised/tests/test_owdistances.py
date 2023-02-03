# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import Mock, patch

import numpy as np

from Orange import distance
from Orange.data import Table, Domain, ContinuousVariable
from Orange.misc import DistMatrix
from Orange.widgets.unsupervised.owdistances import OWDistances, \
    DistanceRunner, MetricDefs, Cosine, Mahalanobis, Jaccard, MetricDef, \
    ManhattanNormalized, EuclideanNormalized, Manhattan, Spearman, Pearson, \
    Hamming, SpearmanAbsolute, PearsonAbsolute, Euclidean
from Orange.widgets.tests.base import WidgetTest


class TestDistanceRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")[::5].copy()
        with cls.iris.unlocked():
            cls.iris.X[0, 2] = np.nan
            cls.iris.X[1, 3] = np.nan
            cls.iris.X[2, 1] = np.nan

        cls.zoo = Table("zoo")[::5].copy()
        with cls.zoo.unlocked():
            cls.zoo.X[0, 2] = np.nan
            cls.zoo.X[1, 3] = np.nan
            cls.zoo.X[2, 1] = np.nan

    def test_run(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)
        for metricdef in MetricDefs.values():
            metric = metricdef.metric
            data = self.iris
            if not metric.supports_missing:
                data = distance.impute(data)
            elif metric == distance.Jaccard:
                data = self.zoo

            # between rows, normalized
            dist1 = DistanceRunner.run(data, metric, True, 0, state)
            dist2 = metric(data, axis=1, impute=True, normalize=True)
            self.assertDistMatrixEqual(dist1, dist2)

            # between rows, not normalized
            dist1 = DistanceRunner.run(data, metric, False, 0, state)
            dist2 = metric(data, axis=1, impute=True, normalize=False)
            self.assertDistMatrixEqual(dist1, dist2)

            # between columns, normalized
            dist1 = DistanceRunner.run(data, metric, True, 1, state)
            dist2 = metric(data, axis=0, impute=True, normalize=True)
            self.assertDistMatrixEqual(dist1, dist2)

            # between columns, not normalized
            dist1 = DistanceRunner.run(data, metric, False, 1, state)
            dist2 = metric(data, axis=0, impute=True, normalize=False)
            self.assertDistMatrixEqual(dist1, dist2)

    def assertDistMatrixEqual(self, dist1, dist2):
        self.assertIsInstance(dist1, DistMatrix)
        self.assertIsInstance(dist2, DistMatrix)
        self.assertEqual(dist1.axis, dist2.axis)
        self.assertEqual(dist1.row_items, dist2.row_items)
        self.assertEqual(dist1.col_items, dist2.col_items)
        np.testing.assert_array_almost_equal(dist1, dist2)


class TestOWDistances(WidgetTest):
    def setUp(self):
        super().setUp()
        self.iris = Table("iris")[::5].copy()
        self.titanic = Table("titanic")[::10].copy()
        self.widget = self.create_widget(OWDistances)

    def _select(self, id_):
        buttons = self.widget.metric_buttons
        buttons.button(id_).setChecked(True)
        buttons.idClicked.emit(id_)

    def test_distance_combo(self):
        """Check distances when the metric changes"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        for metricdef in MetricDefs.values():
            if metricdef.metric is distance.Jaccard:
                continue
            self._select(metricdef.id)
            self.wait_until_stop_blocking()

            kwargs = dict(normalize=True) if metricdef.normalize else {}
            expected = metricdef.metric(self.iris, **kwargs)

            np.testing.assert_array_almost_equal(
                expected, self.get_output(self.widget.Outputs.distances),
                err_msg=f"at {metricdef.name}")

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        self._select(Cosine)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.wait_until_stop_blocking()
        self.assertTrue(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())

    def test_jaccard_messages(self):
        self._select(Jaccard)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_stop_blocking()
        self.assertTrue(self.widget.Error.no_binary_features.is_shown())
        self.assertFalse(self.widget.Warning.ignoring_nonbinary.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertFalse(self.widget.Warning.ignoring_nonbinary.is_shown())

        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertTrue(self.widget.Warning.ignoring_nonbinary.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertFalse(self.widget.Warning.ignoring_nonbinary.is_shown())

        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertTrue(self.widget.Warning.ignoring_nonbinary.is_shown())

        dom = self.titanic.domain
        dom = Domain(dom.attributes[1:], dom.class_var)
        self.send_signal(self.widget.Inputs.data, self.titanic.transform(dom))
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertFalse(self.widget.Warning.ignoring_nonbinary.is_shown())

        self.send_signal(self.widget.Inputs.data, Table("heart_disease"))
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.no_binary_features.is_shown())
        self.assertFalse(self.widget.Warning.ignoring_discrete.is_shown())

    def test_too_big_array(self):
        """
        Users sees an error message when calculating too large arrays and Orange
        does not crash.
        """
        self.assertEqual(len(self.widget.Error.active), 0)
        self.send_signal(self.widget.Inputs.data, self.iris)

        id_ = self.widget.metric_id
        for exc, err in ((ValueError, self.widget.Error.distances_value_error),
                         (MemoryError, self.widget.Error.distances_memory_error)
                         ):
            with patch.dict(
                    MetricDefs,
                    {id_: MetricDef(id_, "", "", Mock(side_effect=exc))}):
                self.widget.compute_distances(self.iris)
                self.wait_until_finished()
                self.assertTrue(err.is_shown(), msg=f"at {exc}")

    def test_migrate_3_to_4(self):
        settings = {'__version__': 3}
        w = self.create_widget(
            OWDistances,
            stored_settings=dict(metric_idx=0, normalized_dist=True, **settings))
        self.assertEqual(w.metric_id, EuclideanNormalized)
        w = self.create_widget(
            OWDistances,
            stored_settings=dict(metric_idx=1, normalized_dist=False, **settings))
        self.assertEqual(w.metric_id, Manhattan)
        w = self.create_widget(
            OWDistances,
            stored_settings=dict(metric_idx=1, normalized_dist=True, **settings))
        self.assertEqual(w.metric_id, ManhattanNormalized)

        for old, new in ((2, Cosine), (3, Jaccard),
                         (4, Spearman), (5, SpearmanAbsolute),
                         (6, Pearson), (7, PearsonAbsolute),
                         (8, Hamming), (9, Mahalanobis),
                         (10, Euclidean)):
            settings = dict(metric_idx=old, __version__=3)
            w = self.create_widget(OWDistances, stored_settings=settings)
            self.assertEqual(w.metric_id, new,
                             msg=f"at {old} to {MetricDefs[new].name}")

    def test_limit_mahalanobis(self):
        def assert_error_shown():
            self.assertTrue(
                self.widget.Error.data_too_large_for_mahalanobis.is_shown())

        def assert_no_error():
            self.assertFalse(
                self.widget.Error.data_too_large_for_mahalanobis.is_shown())

        widget = self.widget
        axis_buttons = widget.controls.axis.buttons

        self._select(Mahalanobis)
        X = np.random.random((1010, 4))
        bigrows = Table.from_numpy(Domain(self.iris.domain.attributes), X)
        bigcols = Table.from_numpy(
            Domain([ContinuousVariable(f"{i}") for i in range(1010)]), X.T)

        self.send_signal(widget.Inputs.data, self.iris)
        assert_no_error()

        axis_buttons[0].click()
        assert_no_error()
        axis_buttons[1].click()
        assert_no_error()

        # by columns -- cannot handle too many rows
        self.send_signal(self.widget.Inputs.data, bigrows)
        assert_no_error()
        axis_buttons[0].click()
        assert_error_shown()
        axis_buttons[1].click()
        assert_no_error()

        self.send_signal(self.widget.Inputs.data, bigcols)
        assert_error_shown()
        axis_buttons[0].click()
        assert_no_error()
        axis_buttons[1].click()
        assert_error_shown()

        self.send_signal(widget.Inputs.data, self.iris)
        assert_no_error()

    def test_discrete_in_metas(self):
        domain = self.iris.domain
        data = self.iris.transform(
            Domain(domain.attributes[:-1] + (domain.class_var, ),
                   [],
                   domain.attributes[-1:])
        )
        self._select(Cosine)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        out = self.get_output(self.widget.Outputs.distances)
        out_domain = out.row_items.domain
        self.assertEqual(out_domain.attributes, domain.attributes[:-1])
        self.assertEqual(out_domain.metas,
                         (domain.attributes[-1], domain.class_var))

    def test_non_binary_in_metas(self):
        self._select(Jaccard)
        zoo = Table("zoo")[:20]
        self.send_signal(self.widget.Inputs.data, zoo)
        self.wait_until_finished()
        out = self.get_output(self.widget.Outputs.distances)
        domain = zoo.domain
        out_domain = out.row_items.domain
        self.assertEqual(out_domain.metas, (domain["name"], domain["legs"]))


if __name__ == "__main__":
    unittest.main()
