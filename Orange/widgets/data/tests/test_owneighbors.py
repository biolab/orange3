#pylint: disable=missing-docstring,protected-access,unsubscriptable-object
import unittest
from unittest.mock import Mock, patch

import numpy as np

from orangewidget.tests.utils import simulate

from Orange import distance
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.misc.distmatrix import DistMatrix
from Orange.widgets.unsupervised.utils import (
    MetricDefs, MetricDef,
    Euclidean, EuclideanNormalized, Manhattan, Cosine, Mahalanobis,
    Pearson, PearsonAbsolute, Spearman, SpearmanAbsolute, Jaccard
)
from Orange.widgets.data.owneighbors import OWNeighbors
from Orange.widgets.tests.base import WidgetTest, ParameterMapping


class TestOWNeighbors(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNeighbors)
        self.iris = Table("iris")
        self.zoo = Table("zoo")
        self.distances = distance.Euclidean(self.iris)

    def test_input_data(self):
        """Check widget's data with data on the input"""
        widget = self.widget
        self.assertEqual(widget.data, None)
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertEqual(widget.data, self.iris)

    def test_input_data_disconnect(self):
        """Check widget's data after disconnecting data on the input"""
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertEqual(widget.data, self.iris)
        self.send_signal(widget.Inputs.data, None)
        self.assertEqual(widget.data, None)

    def test_input_reference(self):
        """Check widget's reference with reference on the input"""
        widget = self.widget
        self.assertEqual(widget.reference, None)
        self.send_signal(widget.Inputs.reference, self.iris)
        self.assertEqual(widget.reference, self.iris)

    def test_output_neighbors(self):
        """Check if neighbors are on the output after apply"""
        widget = self.widget
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(self.get_output(widget.Outputs.data_w_distances))
        self.send_signals(((widget.Inputs.data, self.iris),
                           (widget.Inputs.reference, self.iris[:10])))
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIsNotNone(self.get_output(widget.Outputs.data_w_distances))
        self.assertIsInstance(self.get_output(widget.Outputs.data), Table)
        self.assertTrue(all([i in self.iris.ids for i in
                             self.get_output(widget.Outputs.data).ids])
                        )

    def test_settings(self):
        """Check neighbors for various distance metrics"""
        widget = self.widget
        settings = [
            ParameterMapping("", widget.controls._distance_index,
                             [metric.name for metric in MetricDefs.values()]),
            ParameterMapping("", widget.controls.n_neighbors)]
        for setting in settings:
            for val in setting.values:
                self.send_signal(widget.Inputs.data, self.iris)
                self.send_signal(widget.Inputs.reference, self.iris[:10])
                setting.set_value(val)
                widget.apply_button.button.click()
                self.assertIsNotNone(self.get_output(widget.Outputs.data))

    def test_include_reference(self):
        widget = self.widget
        widget.n_neighbors = 10
        self.widget.include_reference = False
        reference = self.iris[:5]
        self.send_signal(widget.Inputs.data, self.iris)
        self.send_signal(widget.Inputs.reference, reference)
        neighbors = self.get_output(widget.Outputs.data)
        self.assertEqual(len(neighbors), 10)
        for inst in reference:
            self.assertNotIn(inst, neighbors)
        andata = self.get_output(widget.Outputs.data_w_distances)
        self.assertEqual(len(andata), 150)

        self.widget.include_reference = True
        widget.commit.now()
        neighbors = self.get_output(widget.Outputs.data)
        self.assertEqual(len(neighbors), 15)
        for inst in reference:
            self.assertNotIn(inst, neighbors[:10])
        for inst in reference:
            self.assertIn(inst, neighbors[10:])
        andata = self.get_output(widget.Outputs.data_w_distances)
        self.assertEqual(len(andata), 150)

    def test_similarity(self):
        widget = self.widget
        reference = self.iris[:10]
        self.send_signal(widget.Inputs.data, self.iris)
        self.send_signal(widget.Inputs.reference, reference)
        widget.apply_button.button.click()
        neighbors = self.get_output(widget.Outputs.data)
        self.assertEqual(self.iris.domain.attributes,
                         neighbors.domain.attributes)
        self.assertEqual(self.iris.domain.class_vars,
                         neighbors.domain.class_vars)
        self.assertIn("distance", neighbors.domain)
        self.assertTrue(all(100 >= ins["distance"] >= 0 for ins in neighbors))

    def test_annotated_data(self):
        widget = self.widget
        widget.limit_neighbors = True
        widget.n_neighbors = 3
        reference = self.iris[10:11]
        widget.metric_id = Euclidean
        mock_distances = np.array([[5, 6, 8, 1, 3, 2, 4, 7, 6, 6]]).T
        with patch(
                "Orange.distance.distance.EuclideanRowsModel.compute_distances",
                new=Mock(return_value=mock_distances)):
            self.send_signal(widget.Inputs.data, self.iris[20:30])
            self.send_signal(widget.Inputs.reference, reference)
            data = self.get_output(widget.Outputs.data)
            np.testing.assert_equal(data.X, self.iris[[23, 25, 24]].X)

            andata = self.get_output(widget.Outputs.data_w_distances)
            np.testing.assert_equal(andata.X, self.iris[20:30].X)
            np.testing.assert_equal(andata.get_column("distance"),
                                    mock_distances.flatten())

            self.send_signal(widget.Inputs.data, None)
            self.assertIsNone(self.get_output(widget.Outputs.data_w_distances))

    def test_missing_values_in_ref(self):
        widget = self.widget
        widget.metric_id = Mahalanobis
        data = Table("iris")
        reference = data[:3]
        with data.unlocked():
            data.X[0:10, 0] = np.nan
        self.send_signal(widget.Inputs.data, self.iris)
        self.send_signal(widget.Inputs.reference, reference)
        widget.apply_button.button.click()
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIsNotNone(self.get_output(widget.Outputs.data_w_distances))

    def test_missing_values_in_data(self):
        widget = self.widget
        widget.metric_id = Mahalanobis
        data = Table("iris")
        with data.unlocked():
            data.X[0:10, 0] = np.nan
        self.send_signal(widget.Inputs.data, data)
        self.assertTrue(widget.Warning.imputing_data.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.imputing_data.is_shown())

    def test_compute_distances_apply_called(self):
        """Check compute distances and apply are called when receiving signal"""
        widget = self.widget
        cdist = widget.compute_distances = Mock()
        def_commit = widget.commit.now = Mock()
        self.widget.auto_apply = False
        data = Table("iris")
        self.send_signal(widget.Inputs.data, data)
        cdist.assert_called()
        def_commit.assert_called()
        cdist.reset_mock()
        def_commit.reset_mock()

        self.send_signal(widget.Inputs.reference, data[:10])
        cdist.assert_called()
        def_commit.assert_called()
        cdist.reset_mock()
        def_commit.reset_mock()

        self.send_signals([(widget.Inputs.data, data),
                           (widget.Inputs.reference, data[:10])])
        self.assertEqual(cdist.call_count, 1)
        self.assertEqual(def_commit.call_count, 1)

    def test_compute_distances_calls_distance(self):
        widget = self.widget
        dists = np.random.random((10, 5))
        distance = Mock(return_value=dists)
        metric = MetricDef(42, "foo", "", distance)
        metric_id = widget.metric_id
        orig_metrics = MetricDefs[metric_id]
        try:
            MetricDefs[metric_id] = metric
            data = Table("iris")
            data, refs = data[:10], data[-5:]
            self.send_signals([(widget.Inputs.data, data),
                               (widget.Inputs.reference, refs)])
            # false positive, pylint: disable=unsubscriptable-object
            d1, d2 = distance.call_args[0]
            np.testing.assert_almost_equal(d1.X, data.X)
            np.testing.assert_almost_equal(d2.X, refs.X)
            # false positive, pylint: disable=no-member
            np.testing.assert_almost_equal(widget.distances, dists.min(axis=1))
        finally:
            MetricDefs[metric_id] = orig_metrics

    def test_data_with_similarity(self):
        widget = self.widget
        indices = np.array([5, 10, 15, 100])

        data = Table("iris")
        widget.data = data

        widget.distances = np.arange(1000, 1150).astype(float)
        neighbours = widget._data_with_similarity(indices)
        self.assertEqual(neighbours.metas.shape, (4, 1))
        np.testing.assert_almost_equal(
            neighbours.metas.flatten(), indices + 1000)
        np.testing.assert_almost_equal(neighbours.X, data.X[indices])

        domain = data.domain
        domain2 = Domain([domain[2]], domain.class_var, metas=domain[:2])
        data2 = data.transform(domain2)
        widget.data = data2

        widget.distances = np.arange(1000, 1150).astype(float)
        neighbours = widget._data_with_similarity(indices)
        self.assertEqual(len(neighbours.domain.metas), 3)
        self.assertEqual(neighbours.metas.shape, (4, 3))
        np.testing.assert_almost_equal(
            neighbours.get_column("distance"), indices + 1000)
        np.testing.assert_almost_equal(neighbours.X, data2.X[indices])

    def test_apply(self):
        widget = self.widget
        widget.auto_apply = True
        data = Table("iris")
        indices = np.array([5, 10, 15, 100])

        widget._compute_indices = lambda: \
            indices if widget.distances is not None else None
        self.send_signal(widget.Inputs.data, data)
        self.send_signal(widget.Inputs.reference, data[42:43])
        neigh = self.get_output(widget.Outputs.data)
        np.testing.assert_almost_equal(neigh.X, data.X[indices])
        np.testing.assert_almost_equal(
            neigh.metas.flatten(), widget.distances[indices])

    def test_all_equal_ref(self):
        widget = self.widget
        widget.auto_apply = True

        data = Table("iris")
        self.send_signal(widget.Inputs.data, data[:10])
        self.send_signal(widget.Inputs.reference, data[:10])
        self.assertTrue(widget.Warning.all_data_as_reference.is_shown())
        self.assertFalse(widget.Info.removed_references.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNotNone(self.get_output(widget.Outputs.data_w_distances))

        self.send_signal(widget.Inputs.data, data[:15])
        self.assertFalse(widget.Warning.all_data_as_reference.is_shown())
        self.assertTrue(widget.Info.removed_references.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

    def test_different_domains(self):
        """
        Test weather widget show error when data and a reference have different
        domains.
        """
        w = self.widget
        self.widget.metric_id = Mahalanobis


        domain = Domain([ContinuousVariable("a")])
        domain_ref = Domain([ContinuousVariable("b")])
        data = Table(domain, np.random.rand(2, len(domain.variables)))
        reference = Table(
            domain_ref, np.random.rand(1, len(domain_ref.variables)))

        # no error if one or both of the signals is missing
        self.send_signal(w.Inputs.data, data)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

        self.send_signal(w.Inputs.reference, data[:1])
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

        # same domain - no error
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, data[:1])
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

        # one attribute different attribute name
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, reference)
        self.assertTrue(w.Error.diff_domains_or_missing.is_shown())

        # error disappears when data is set to None
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

        # error disappears when reference is set to None
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, reference)
        self.assertTrue(w.Error.diff_domains_or_missing.is_shown())

        self.send_signal(w.Inputs.reference, None)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())

    def test_different_metas(self):
        """
        Test whether widget does not show error when data and a reference have
        domain that differ only in metas
        """
        w = self.widget

        domain = Domain([ContinuousVariable("a"), ContinuousVariable("b")],
                        metas=[ContinuousVariable("c")])
        data = Table(
            domain, np.random.rand(15, len(domain.attributes)),
            metas=np.random.rand(15, len(domain.metas)))

        # same domain with same metas no error
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, data[:1])
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())
        output = self.get_output(w.Outputs.data)
        self.assertEqual(10, len(output))

        # same domain with different metas no error
        domain_ref = Domain(domain.attributes,
                            metas=[ContinuousVariable("d")])
        reference = Table(
            domain_ref, np.random.rand(1, len(domain_ref.attributes)),
            metas=np.random.rand(1, len(domain.metas)))
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, reference)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())
        output = self.get_output(w.Outputs.data)
        self.assertEqual(10, len(output))

        # same domain with different order - no error
        domain_ref = Domain(domain.attributes[::-1])
        reference = Table(
            domain_ref, np.random.rand(1, len(domain_ref.attributes)))
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, reference)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())
        output = self.get_output(w.Outputs.data)
        self.assertEqual(10, len(output))

        # same domain with different number of metas no error
        domain_ref = Domain(
            domain.attributes,
            metas=[ContinuousVariable("d"), ContinuousVariable("e")])
        reference = Table(
            domain_ref, np.random.rand(1, len(domain_ref.attributes)),
            metas=np.random.rand(1, len(domain_ref.metas)))
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, reference)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())
        output = self.get_output(w.Outputs.data)
        self.assertEqual(10, len(output))

    def test_different_domains_same_names(self):
        """
        Here we create two domains with same names of attributes - the case
        with image-analytics (two embeddings). Widget must not show the
        error in this case.
        """
        w = self.widget
        domain1 = Domain([ContinuousVariable("a"), ContinuousVariable("b")],
                         metas=[ContinuousVariable("c")])
        domain2 = Domain([ContinuousVariable("a"), ContinuousVariable("b")],
                         metas=[ContinuousVariable("d")])
        data = Table(
            domain1, np.random.rand(15, len(domain1.attributes)),
            metas=np.random.rand(15, len(domain1.metas)))
        ref = Table(
            domain2, np.random.rand(2, len(domain2.attributes)),
            metas=np.random.rand(2, len(domain2.metas)))
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.reference, ref)
        self.assertFalse(w.Error.diff_domains_or_missing.is_shown())
        output = self.get_output(w.Outputs.data)
        self.assertEqual(10, len(output))

    def test_n_neighbours_spin_max(self):
        w = self.widget
        sb = w.controls.n_neighbors
        default = sb.maximum()
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual(sb.maximum(), len(self.iris))
        self.send_signal(w.Inputs.data, self.iris[:20])
        self.assertEqual(sb.maximum(), 20)
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(sb.maximum(), default)

    def test_inherited_table(self):
        # pylint: disable=abstract-method
        class Table2(Table):
            pass

        data = Table2(self.iris)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, data[0:1])
        self.assertIsInstance(self.get_output(self.widget.Outputs.data), Table2)
        self.assertIsInstance(self.get_output(self.widget.Outputs.data_w_distances), Table2)

    def test_order_by_distance(self):
        self.widget.metric_id = Euclidean
        domain = Domain([ContinuousVariable(x) for x in "ab"])
        reference = Table.from_numpy(domain, [[1, 0]])
        data = Table.from_numpy(domain, [[1, 0.1], [2, 0], [1, 0], [0, 0.1], [0.1, 0]])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, reference)

        output = self.get_output(self.widget.Outputs.data)
        expected = [[1, 0], [1, 0.1], [0.1, 0], [2, 0], [0, 0.1]]
        np.testing.assert_array_equal(output.X, expected)
        dst = output.get_column("distance").tolist()
        self.assertTrue(dst == sorted(dst))  # check distance in ascending order

        # test on bigger set
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.reference, self.iris[:1])
        dst = self.get_output(self.widget.Outputs.data).get_column("distance").tolist()
        self.assertTrue(dst == sorted(dst))  # check distance in ascending order

    def test_unknowns_data(self):
        self.widget.metric_id = Mahalanobis
        self.assertFalse(MetricDefs[self.widget.metric_id].metric.supports_missing)
        domain = Domain([ContinuousVariable(x) for x in "ab"])
        reference = Table.from_numpy(domain, [[1, 0]])
        data = Table.from_numpy(domain, [[1, "NaN"], [2, 0], [1, 0], [0, 0.1], [0.1, 0]])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, reference)
        dst = self.get_output(self.widget.Outputs.data).get_column("distance")
        self.assertFalse(np.any(np.isnan(dst)))

    def test_unknowns_reference(self):
        self.widget.metric_id = Mahalanobis
        self.assertFalse(MetricDefs[self.widget.metric_id].metric.supports_missing)
        domain = Domain([ContinuousVariable(x) for x in "ab"])
        reference = Table.from_numpy(domain, [[1, "NaN"]])
        data = Table.from_numpy(domain, [[1, 0.1], [2, 0], [1, 0], [0, 0.1], [0.1, 0]])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, reference)
        dst = self.get_output(self.widget.Outputs.data).get_column("distance")
        self.assertFalse(np.any(np.isnan(dst)))

    def test_set_distances_checks_same_table(self):
        w = self.widget
        w.handleNewSignals = Mock()

        distances = Mock()
        distances.col_items = distances_row_items = None
        w.set_distances(distances)
        self.assertFalse(w.Error.not_same_table.is_shown())
        self.assertIs(w.distance_matrix, distances)

        distances.row_items = Mock()
        w.set_distances(distances)
        self.assertFalse(w.Error.not_same_table.is_shown())
        self.assertIs(w.distance_matrix, distances)

        distances.col_items = Mock()
        w.set_distances(distances)
        self.assertTrue(w.Error.not_same_table.is_shown())
        self.assertIsNone(w.distance_matrix)

        distances.col_items = distances.row_items
        w.set_distances(distances)
        self.assertFalse(w.Error.not_same_table.is_shown())
        self.assertIs(w.distance_matrix, distances)

        distances.col_items = Mock()
        w.set_distances(distances)
        self.assertTrue(w.Error.not_same_table.is_shown())
        self.assertIsNone(w.distance_matrix)

        w.set_distances(None)
        self.assertFalse(w.Error.not_same_table.is_shown())
        self.assertIsNone(w.distance_matrix)

    def test_mismatching_input_sizes(self):
        w = self.widget
        distances = DistMatrix(np.array([[1, 2], [3, 4]]))
        self.send_signal(w.Inputs.data, self.iris)
        self.send_signal(w.Inputs.distances, distances)
        self.assertTrue(w.mismatching_input_sizes)
        self.assertTrue(w.Error.diff_number.is_shown())

        self.send_signal(w.Inputs.distances, self.distances)
        self.assertFalse(w.mismatching_input_sizes)
        self.assertFalse(w.Error.diff_number.is_shown())

    def test_act_data_and_dimension(self):
        w = self.widget
        siri = self.iris[::-1]
        iris50 = self.iris[:50]

        self.distances.row_items = siri

        self.send_signal(w.Inputs.distances, self.distances)
        self.assertIs(w.act_data, siri)
        self.assertEqual(w.dimension, 150)

        self.send_signal(w.Inputs.data, iris50)
        self.assertIs(w.act_data, iris50)
        self.assertEqual(w.dimension, 50)

        self.send_signal(w.Inputs.distances, None)
        self.assertIs(w.act_data, iris50)
        self.assertEqual(w.dimension, 50)

        self.send_signal(w.Inputs.distances, self.distances)
        self.send_signal(w.Inputs.data, iris50)
        self.assertIs(w.act_data, iris50)
        self.assertEqual(w.dimension, 50)

        self.send_signal(w.Inputs.data, None)
        self.assertIs(w.act_data, siri)
        self.assertEqual(w.dimension, 150)

        self.send_signal(w.Inputs.distances, None)
        self.assertIsNone(w.act_data)
        self.assertEqual(w.dimension, 0)

    def test_update_variable_model(self):
        w = self.widget

        self.assertEqual(list(self.widget.variable_model), [None])
        self.assertIsNone(w.label)

        domain = Domain(
            [ContinuousVariable(x) for x in "abc"],
            None,
            metas=[StringVariable("m1"), StringVariable("m2"), ContinuousVariable("m3")]
        )
        data = Table(
            domain,
            np.random.rand(2, 3),
            metas=[["x", "y", 0], ["v", "w", 1]]
        )
        self.send_signal(w.Inputs.data, data)

        self.assertEqual(
            list(self.widget.variable_model),
            [None, data.domain.metas[0], data.domain.metas[1]])
        self.assertEqual(
            list(w.label_model),
            ["x", "v"])

        self.assertTrue(w.controls.label.isEnabled())
        self.assertIs(w.label, data.domain.metas[0])

        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual(list(self.widget.variable_model), [None])
        self.assertFalse(w.controls.label.isEnabled())
        self.assertIsNone(w.label)
        self.assertEqual(
            list(w.label_model),
            [str(i) for i in range(1, 1 + len(self.iris))])

        self.send_signal(w.Inputs.data, data)
        self.assertNotEqual(list(self.widget.variable_model), [None])
        self.assertIsNotNone(w.label)
        self.assertTrue(w.controls.label.isEnabled())
        self.assertIsNotNone(w.label)

        self.send_signal(w.Inputs.data, None)
        self.assertEqual(list(self.widget.variable_model), [None])
        self.assertIsNone(w.label)
        self.assertFalse(w.controls.label.isEnabled())
        self.assertIsNone(w.label)

    def test_update_labels(self):
        w = self.widget
        w.selected_item_hint = "v"

        domain = Domain(
            [ContinuousVariable(x) for x in "abc"],
            None,
            metas=[StringVariable("m1"), StringVariable("m2"), ContinuousVariable("m3")]
        )
        data = Table(
            domain,
            np.random.rand(2, 3),
            metas=[["x", "y", 0], ["v", "w", 1]]
        )
        self.send_signal(w.Inputs.data, data)

        self.assertEqual(w.selected_item, "v")

        simulate.combobox_activate_index(w.controls.label, 0)
        self.assertEqual(w.selected_item, "1")
        self.assertEqual(w.selected_item_hint, "1")

        simulate.combobox_activate_index(w.controls.label, 1)
        self.assertEqual(w.selected_item, "x")

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.selected_item)

    def test_update_labels_inapplicable_hint(self):
        w = self.widget
        w.selected_item_hint = "v"

        self.send_signal(w.Inputs.data, self.zoo)
        self.assertEqual(w.selected_item, self.zoo[0, "name"])

    def test_update_info(self):
        w = self.widget

        self.assertFalse(w.metric_box.isHidden())
        self.assertTrue(w.metric_info.isHidden())
        self.assertFalse(w.reference_box.isHidden())
        self.assertTrue(w.reference_info.isHidden())

        self.send_signal(w.Inputs.reference, self.iris[10:12])
        self.assertFalse(w.metric_box.isHidden())
        self.assertTrue(w.metric_info.isHidden())
        self.assertTrue(w.reference_box.isHidden())
        self.assertFalse(w.reference_info.isHidden())

        self.send_signal(w.Inputs.distances, self.distances)
        self.assertTrue(w.metric_box.isHidden())
        self.assertFalse(w.metric_info.isHidden())
        self.assertTrue(w.reference_box.isHidden())
        self.assertFalse(w.reference_info.isHidden())

        self.send_signal(w.Inputs.reference, None)
        self.assertTrue(w.metric_box.isHidden())
        self.assertFalse(w.metric_info.isHidden())
        self.assertFalse(w.reference_box.isHidden())
        self.assertTrue(w.reference_info.isHidden())

        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.metric_box.isHidden())
        self.assertTrue(w.metric_info.isHidden())
        self.assertFalse(w.reference_box.isHidden())
        self.assertTrue(w.reference_info.isHidden())

    def test_item_changed(self):
        w = self.widget
        w.auto_apply = True
        w.n_neighbors = 1
        w.include_reference = True

        self.send_signal(w.Inputs.data, self.zoo)
        out = self.get_output(w.Outputs.data)
        self.assertEqual(out[1, "name"], self.zoo[0, "name"])

        simulate.combobox_activate_index(w.controls.selected_item, 100)
        out = self.get_output(w.Outputs.data)
        self.assertEqual(out[1, "name"], self.zoo[100, "name"])

    def test_mapping_back_and_forth(self):
        # Test that reference example is mapped from data to distance matrix
        # and that neighbors are mapped from distance matrix to data.
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris[:, :2])
        self.send_signal(w.Inputs.distances, self.distances)
        simulate.combobox_activate_index(w.controls.selected_item, 149)
        self.assertIsNone(w.mapping)

        out = self.get_output(w.Outputs.data)

        self.send_signal(w.Inputs.data, self.iris[50:, :2])
        np.testing.assert_equal(w.mapping, np.arange(50, 150))

        simulate.combobox_activate_index(w.controls.selected_item, 99)

        out2 = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.X, out2.X)

    def test_mapping_not_needed(self):
        w = self.widget
        self.assertIsNone(w.mapping)

        # No distances
        self.send_signal(w.Inputs.data, self.iris)
        self.assertIsNone(w.mapping)

        # Same ids
        self.send_signal(w.Inputs.distances, self.distances)
        self.assertIsNone(w.mapping)

        # Different size: mapping is needed!
        self.send_signal(w.Inputs.data, self.iris[1:130])
        np.testing.assert_equal(w.mapping, np.arange(1, 130))

        # No data
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.mapping)

        # Different order: mapping is needed!
        self.send_signal(w.Inputs.data, self.iris[::-1])
        np.testing.assert_equal(w.mapping, np.arange(0, 150)[::-1])

    def test_mapping_fails(self):
        w = self.widget

        iris50 = self.iris[:50]
        distance_50 = distance.Euclidean(self.iris[-50:])

        self.send_signal(w.Inputs.data, iris50)
        self.send_signal(w.Inputs.distances, distance_50)
        self.assertIs(w.mapping, False)
        self.assertTrue(w.Error.dist_data_id_mismatch.is_shown())
        self.assertFalse(w.Error.missing_distances.is_shown())

        self.send_signal(w.Inputs.data, self.iris)
        self.assertIs(w.mapping, False)
        self.assertFalse(w.Error.dist_data_id_mismatch.is_shown())
        self.assertTrue(w.Error.missing_distances.is_shown())

        self.send_signal(w.Inputs.distances, None)
        self.assertIsNone(w.mapping)
        self.assertFalse(w.Error.dist_data_id_mismatch.is_shown())
        self.assertFalse(w.Error.missing_distances.is_shown())

    def test_computation_combinations(self):
        w = self.widget
        w.metric_id = Euclidean
        w.n_neighbors = 2

        self.send_signal(w.Inputs.data, self.iris[50:])
        simulate.combobox_activate_index(w.controls.selected_item, 99)
        dist_data_noref = w.distances
        out_data_noref = self.get_output(w.Outputs.data)

        self.send_signal(w.Inputs.reference, self.iris[149:])
        dist_data_ref = w.distances
        out_data_ref = self.get_output(w.Outputs.data)

        self.send_signal(w.Inputs.distances, self.distances)
        dist_data_dist_ref = w.distances
        out_data_dist_ref = self.get_output(w.Outputs.data)

        self.send_signal(w.Inputs.reference, None)
        dist_data_dist_noref = w.distances
        out_data_dist_noref = self.get_output(w.Outputs.data)

        np.testing.assert_almost_equal(dist_data_noref[:10], dist_data_ref[:10])
        np.testing.assert_almost_equal(dist_data_noref[:10], dist_data_dist_ref[:10])
        np.testing.assert_almost_equal(dist_data_noref[:10], dist_data_dist_noref[:10])
        np.testing.assert_equal(out_data_noref.X, out_data_ref.X)
        np.testing.assert_equal(out_data_noref.X, out_data_dist_ref.X)
        np.testing.assert_equal(out_data_noref.X, out_data_dist_noref.X)

    def test_missing_references_in_distances(self):
        w = self.widget

        distances = distance.Euclidean(self.iris[50:])
        self.send_signal(w.Inputs.distances, distances)

        self.assertFalse(w.Error.no_references.is_shown())
        self.assertFalse(w.Warning.ignored_references.is_shown())

        self.send_signal(w.Inputs.reference, self.iris[45:55])
        self.assertFalse(w.Error.no_references.is_shown())
        self.assertTrue(w.Warning.ignored_references.is_shown())
        self.assertIsNotNone(self.get_output(w.Outputs.data))
        self.assertIsNotNone(w.distances)

        self.send_signal(w.Inputs.reference, self.iris[45:50])
        self.assertTrue(w.Error.no_references.is_shown())
        self.assertFalse(w.Warning.ignored_references.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.data))
        self.assertIsNone(w.distances)

    def test_no_reference_in_distances(self):
        w = self.widget

        distances = distance.Euclidean(self.iris[50:])
        self.send_signal(w.Inputs.distances, distances)

        self.assertFalse(w.Error.no_references.is_shown())
        self.assertFalse(w.Warning.ignored_references.is_shown())

        self.send_signal(w.Inputs.reference, self.iris[45:50])
        self.assertTrue(w.Error.no_references.is_shown())
        self.assertFalse(w.Warning.ignored_references.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.data))
        self.assertIsNone(w.distances)

    def test_reference_from_pairwise_distances(self):
        w = self.widget

        self.send_signal(w.Inputs.distances, self.distances)
        self.send_signal(w.Inputs.data, self.iris)
        self.send_signal(w.Inputs.reference, self.iris[11:12])
        out = self.get_output(w.Outputs.data)

        w2 = self.create_widget(OWNeighbors)
        self.distances.row_items = self.distances.col_items = None
        self.send_signal(w2.Inputs.distances, self.distances)
        self.send_signal(w2.Inputs.data, self.iris)
        self.send_signal(w2.Inputs.reference, self.iris[11:12])
        self.assertTrue(w2.Warning.neighbor_by_indices.is_shown())

        out2 = self.get_output(w2.Outputs.data)
        np.testing.assert_equal(out.X, out2.X)

    def test_metric_fails(self):
        w = self.widget

        distance = Mock(side_effect=RuntimeError)
        metric = MetricDef(42, "foo", "", distance)
        metric_id = w.metric_id
        orig_metrics = MetricDefs[metric_id]
        try:
            MetricDefs[metric_id] = metric
            self.send_signal(w.Inputs.data, self.zoo)
            self.assertIsNone(self.get_output(w.Outputs.data))
            self.assertIsNone(w.distances)
            self.assertTrue(w.Error.metric_failed.is_shown())

            simulate.combobox_activate_index(
                w.controls._distance_index, int(not w._distance_index))
            self.assertIsNotNone(self.get_output(w.Outputs.data))
            self.assertIsNotNone(w.distances)
            self.assertFalse(w.Error.metric_failed.is_shown())
        finally:
            MetricDefs[metric_id] = orig_metrics

    def test_no_distance_matrix_data(self):
        w = self.widget
        w.n_neighbors = 3

        distances = DistMatrix([
            [0, 6, 4, 3, 1, 2, 7],
            [6, 0, 3, 3, 3, 3, 3],
            [4, 3, 0, 3, 3, 3, 3],
            [1, 3, 3, 0, 3, 3, 3],
            [2, 3, 3, 3, 0, 3, 3],
            [3, 3, 3, 3, 3, 0, 3],
            [7, 3, 3, 3, 3, 3, 0]
        ])
        self.send_signal(w.Inputs.distances, distances)
        self.assertTrue(w.Warning.no_distance_matrix_data.is_shown())
        self.assertFalse(w.Warning.neighbor_by_indices.is_shown())
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(
            out.X,
            np.array([[5, 6, 4], [1, 2, 3]]).T)

        self.send_signal(w.Inputs.data, self.zoo[:7])
        self.assertFalse(w.Warning.no_distance_matrix_data.is_shown())
        self.assertTrue(w.Warning.neighbor_by_indices.is_shown())

        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.X, self.zoo.X[[4, 5, 3]])

    def test_nothing_to_match_reference(self):
        w = self.widget

        self.distances.row_items = self.distances.col_items = None
        self.send_signal(w.Inputs.distances, self.distances)
        self.assertFalse(w.Error.no_references.is_shown())
        self.send_signal(w.Inputs.reference, self.iris[50:52])
        self.assertTrue(w.Error.no_references.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.no_references.is_shown())

    def test_output_similarity(self):
        w = self.widget

        self.distances.row_items = self.distances.col_items = None
        self.send_signal(w.Inputs.distances, self.distances)
        out_dist = self.get_output(w.Outputs.data)
        out_dist_w = self.get_output(w.Outputs.data_w_distances)

        w.controls.transform_to_similarity.click()
        out_dist2 = self.get_output(w.Outputs.data)
        out_dist_w2 = self.get_output(w.Outputs.data_w_distances)

        np.testing.assert_equal(out_dist2.X[:, 1], np.max(out_dist.X[:, 1]) - out_dist.X[:, 1])
        np.testing.assert_equal(out_dist_w2.X[:, 1], np.max(out_dist_w.X[:, 1]) - out_dist_w.X[:, 1])

        np.testing.assert_equal(out_dist2.X[:, 0], out_dist.X[:, 0])
        np.testing.assert_equal(out_dist_w2.X[:, 0], out_dist_w.X[:, 0])

    def test_invalid_setting(self):
        w = self.create_widget(
            OWNeighbors, stored_settings={"metric_id": 42, "__version__": 3})
        self.assertEqual(w.metric_id, EuclideanNormalized)
        self.assertEqual(w._distance_index, 0)

    def test_migrate_to_2(self):
        for old, new in ((0, Euclidean), (1, Manhattan),
                         (2, Mahalanobis), (3, Cosine),
                         (4, Jaccard), (5, Spearman),
                         (6, SpearmanAbsolute), (7, Pearson),
                         (8, PearsonAbsolute)):
            settings = dict(distance_index=old, __version__=0)
            w = self.create_widget(OWNeighbors, stored_settings=settings)
            self.assertEqual(w.metric_id, new,
                             msg=f"at {old} to {MetricDefs[new].name}")

    def test_migrate_to_3(self):
        settings = dict(limit_neighbors=False, n_neighbors=5, __version__=2)
        w = self.create_widget(OWNeighbors, stored_settings=settings)
        w.limit_neighbors = False
        w.backward_compatibility_limit = True
        self.send_signal(w.Inputs.data, self.iris[:30])
        self.send_signal(w.Inputs.reference, self.iris[50:52])
        self.assertEqual(len(self.get_output(w.Outputs.data)), 30)
        self.assertTrue(hasattr(w.controls.n_neighbors, "cbox"))

        settings = dict(limit_neighbors=True, n_neighbors=5, __version__=2)
        w = self.create_widget(OWNeighbors, stored_settings=settings)
        w.limit_neighbors = True
        w.backward_compatibility_limit = False
        self.send_signal(w.Inputs.data, self.iris[:30])
        self.send_signal(w.Inputs.reference, self.iris[50:52])
        self.assertEqual(len(self.get_output(w.Outputs.data)), 5)
        self.assertFalse(hasattr(w.controls.n_neighbors, "cbox"))


if __name__ == "__main__":
    unittest.main()
