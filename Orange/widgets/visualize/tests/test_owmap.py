import os
import time
import unittest
import numpy as np

from AnyQt.QtCore import QT_VERSION_STR
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.regression.knn import KNNRegressionLearner

QT_TOO_OLD = QT_VERSION_STR < '5.3'

try:
    from Orange.widgets.visualize.owmap import OWMap
except RuntimeError:
    assert QT_TOO_OLD


@unittest.skipIf(QT_TOO_OLD, "not supported in Qt <5.3")
@unittest.skipIf(os.environ.get('APPVEYOR'), 'Segfaults on Appveyor')
class TestOWMap(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        np.random.seed(666)
        cls.data = Table(Domain([ContinuousVariable('latitude'),
                                 ContinuousVariable('longitude'),
                                 DiscreteVariable('foo', list(map(str, range(5))))],
                                ContinuousVariable('cls')),
                         np.c_[np.random.random((20, 2)) * 10,
                               np.random.randint(5, size=20)],
                         np.random.random(20))

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWMap)  # type: OWMap

    def test_inputs(self):
        self.send_signal('Data', self.data)
        self.send_signal('Learner', KNNRegressionLearner())
        self.widget.handleNewSignals()
        self.assertEqual(self.widget.map.lat_attr, self.data.domain[0])

    def test_latlon_detection_heuristic(self):
        xy = np.c_[np.random.uniform(-180, 180, 100),
                   np.random.uniform(-90, 90, 100)]
        data = Table.from_numpy(Domain.from_numpy(xy), xy)
        self.widget.set_data(data)

        self.assertIn(self.widget.lat_attr, data.domain)
        self.assertIn(self.widget.lon_attr, data.domain)

    def test_projection(self):
        lat = np.r_[-89, 0, 89]
        lon = np.r_[-180, 0, 180]
        easting, northing = self.widget.map.Projection.latlon_to_easting_northing(lat, lon)
        x, y = self.widget.map.Projection.easting_northing_to_pixel(
            easting, northing, 0, [0, 0], [0, 0])
        np.testing.assert_equal(x, [0, 128, 256])
        np.testing.assert_equal(y, [256, 128, 0])

    def test_coverage(self):
        # Due to async nature of these calls, these tests just cover
        self.send_signal('Data', self.data)
        self.send_signal('Learner', KNNRegressionLearner())
        self.widget.class_attr = 'cls'
        self.widget.handleNewSignals()

        self.widget.map.fit_to_bounds()
        self.widget.map.selected_area(90, 180, -90, -180)
        self.widget.map.set_map_provider(next(iter(self.widget.TILE_PROVIDERS.values())))
        self.widget.map.set_clustering(True)
        self.widget.map.set_clustering(False)
        self.widget.map.set_jittering(5)
        self.widget.map.set_marker_color('latitude')
        self.widget.map.set_marker_label('latitude')
        self.widget.map.set_marker_shape('foo')
        self.widget.map.set_marker_size('latitude')
        self.process_events(
            until=lambda start_time=time.clock(): time.clock() - start_time > .2)
        self.widget.map.set_marker_color(None)
        self.widget.map.set_marker_label(None)
        self.widget.map.set_marker_shape(None)
        self.widget.map.set_marker_size(None)
        self.widget.map.set_marker_size_coefficient(50)
        self.widget.map.set_marker_opacity(20)
        self.widget.map.recompute_heatmap(np.random.random((20, 2)))

        args = [100, 100, -100, -100, 1000, 1000, 3, [-100, 100], [0, 0]]
        self.widget.map.redraw_markers_overlay_image(*args, new_image=True)
        # Force non-JS overlay redrawing
        self.widget.map.N_POINTS_PER_ITER = 5
        self.widget.map.redraw_markers_overlay_image(*args, new_image=True)
        # pylint: disable=protected-access
        assert (not np.isnan(self.widget.map._image_token) and
                self.widget.map._image_token is not None)
        self.process_events(until=lambda: self.widget.map._image_token is None)

        self.widget.map.bridge.fit_to_bounds()
        self.widget.map.bridge.selected_area(10, 20, 10, 20)
        self.widget.map.bridge.recompute_heatmap(np.random.random((30, 2)))
        self.widget.map.bridge.redraw_markers_overlay_image(1, 2, 3, 4, 5, 6, 7, [1, 2], [3, 4])

        self.widget.clear()
