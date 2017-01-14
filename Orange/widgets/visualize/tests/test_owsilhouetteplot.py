# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

import numpy as np

import Orange.data
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.visualize.owsilhouetteplot import OWSilhouettePlot
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWSilhouettePlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWSilhouettePlot,
                                         stored_settings={"auto_commit": True})
        self.widget = self.widget  # type: OWSilhouettePlot

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal("Data", self.data[:0])

    def test_outputs_add_scores(self):
        # check output when appending scores
        self.send_signal("Data", self.data)
        self.widget.controls.add_scores.setChecked(1)
        selected_indices = self._select_data()
        name = "Silhouette ({})".format(self.data.domain.class_var.name)
        selected = self.get_output("Selected Data")
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(name, selected.domain.metas[0].name)
        self.assertEqual(name, annotated.domain.metas[0].name)
        np.testing.assert_array_equal(selected.X, self.data.X[selected_indices])

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget._silplot.setSelection(points)
        return sorted(points)

    def test_insufficient_clusters(self):
        iris = self.data
        data_one_cluster = iris[:3]  # three instances Iris-setosa only
        self.send_signal("Data", data_one_cluster)
        self.assertTrue(self.widget.Error.need_two_clusters.is_shown())

        data_singletons = iris[[0, 50, 100]]
        assert len(np.unique(data_singletons.Y)) == 3  # 3 instances 3 labels
        self.send_signal("Data", data_singletons)
        self.assertTrue(self.widget.Error.singleton_clusters_all.is_shown())

    def test_unknowns_in_labels(self):
        self.widget.controls.add_scores.setChecked(1)
        scorename = "Silhouette (iris)"
        data = self.data[[0, 1, 2, 50, 51, 52, 100, 101, 102]]
        data.Y[::3] = np.nan
        valid = ~np.isnan(data.Y.flatten())
        self.send_signal("Data", data)
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        scores = output[:, scorename].metas.flatten()
        self.assertTrue(np.all(np.isnan(scores[::3])))
        self.assertTrue(np.all(np.isfinite(scores[valid])))

        # Run again on subset with known labels
        data_1 = data[np.flatnonzero(valid)]
        self.send_signal("Data", data_1)
        output_1 = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        scores_1 = output_1[:, scorename].metas.flatten()
        self.assertTrue(np.all(np.isfinite(scores_1)))
        # the scores must match
        np.testing.assert_almost_equal(scores_1, scores[valid], decimal=12)

    def test_meta_object_dtype(self):
        # gh-1875: Test on mixed string/discrete metas
        data = self.data[::5]
        domain = Orange.data.Domain(
            data.domain.attributes, [],
            [data.domain["iris"],
             Orange.data.StringVariable("S")]
        )
        data = data.from_table(domain, data)
        self.send_signal("Data", data)
