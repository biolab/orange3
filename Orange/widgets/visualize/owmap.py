import os
from itertools import chain, repeat
from collections import OrderedDict
from tempfile import mkstemp

import numpy as np

from AnyQt.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QTimer, QT_VERSION_STR, \
    QObject
from AnyQt.QtGui import QImage, QPainter, QPen, QBrush, QColor
from AnyQt.QtWidgets import qApp


from Orange.util import color_to_hex
from Orange.base import Learner
from Orange.data.util import scale
from Orange.data import Table, Domain, TimeVariable, DiscreteVariable, ContinuousVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.webview import WebviewWidget
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, ContinuousPaletteGenerator
from Orange.widgets.utils.annotated_data import create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME


if QT_VERSION_STR <= '5.3':
    raise RuntimeError('Map widget only works with Qt 5.3+')


class LeafletMap(WebviewWidget):
    selectionChanged = pyqtSignal(list)

    def __init__(self, parent=None):

        class Bridge(QObject):
            @pyqtSlot()
            def fit_to_bounds(_):
                return self.fit_to_bounds()

            @pyqtSlot(float, float, float, float)
            def selected_area(_, *args):
                return self.selected_area(*args)

            @pyqtSlot('QVariantList')
            def recompute_heatmap(_, *args):
                return self.recompute_heatmap(*args)

            @pyqtSlot(float, float, float, float, int, int, float, 'QVariantList', 'QVariantList')
            def redraw_markers_overlay_image(_, *args):
                return self.redraw_markers_overlay_image(*args)

        super().__init__(parent,
                         bridge=Bridge(),
                         url=QUrl(self.toFileURL(
                             os.path.join(os.path.dirname(__file__), '_owmap', 'owmap.html'))),
                         debug=True,)
        self.jittering = 0
        self._jittering_offsets = None
        self._owwidget = parent
        self._opacity = 255
        self._sizes = None
        self._selected_indices = None

        self.lat_attr = None
        self.lon_attr = None
        self.data = None
        self.model = None
        self._domain = None
        self._latlon_data = None

        self._jittering = None
        self._color_attr = None
        self._label_attr = None
        self._shape_attr = None
        self._size_attr = None
        self._legend_colors = []
        self._legend_shapes = []
        self._legend_sizes = []

        self._drawing_args = None
        self._image_token = None
        self._prev_map_pane_pos = None
        self._prev_origin = None
        self._overlay_image_path = mkstemp(prefix='orange-Map-', suffix='.png')[1]
        self._subset_ids = np.array([])
        self.is_js_path = None

    def __del__(self):
        os.remove(self._overlay_image_path)
        self._image_token = np.nan

    def set_data(self, data, lat_attr, lon_attr):
        self.data = data
        self._image_token = np.nan  # Stop drawing previous image
        self._owwidget.progressBarFinished(None)

        if (data is None or not len(data) or
                lat_attr not in data.domain or
                lon_attr not in data.domain):
            self.data = None
            self.evalJS('clear_markers_js(); clear_markers_overlay_image();')
            self._legend_colors = []
            self._legend_shapes = []
            self._legend_sizes = []
            self._update_legend()
            return

        lat_attr = data.domain[lat_attr]
        lon_attr = data.domain[lon_attr]

        fit_bounds = (self._domain != data.domain or
                      self.lat_attr is not lat_attr or
                      self.lon_attr is not lon_attr)
        self.lat_attr = lat_attr
        self.lon_attr = lon_attr
        self._domain = data.domain

        self._latlon_data = np.array([
            self.data.get_column_view(self.lat_attr)[0],
            self.data.get_column_view(self.lon_attr)[0]],
            dtype=float, order='F').T

        self._recompute_jittering_offsets()

        if fit_bounds:
            QTimer.singleShot(1, self.fit_to_bounds)
        else:
            self.redraw_markers_overlay_image(new_image=True)

    def fit_to_bounds(self, fly=True):
        if self.data is None:
            return
        lat_data, lon_data = self._latlon_data.T
        north, south = np.nanmax(lat_data), np.nanmin(lat_data)
        east, west = np.nanmin(lon_data), np.nanmax(lon_data)
        script = ('map.%sBounds([[%f, %f], [%f, %f]], {padding: [0,0], minZoom: 2, maxZoom: 13})' %
                  ('flyTo' if fly else 'fit', south, west, north, east))
        self.evalJS(script)
        # Sometimes on first data, it doesn't zoom in enough. So let do it
        # once more for good measure!
        self.evalJS(script)

    def selected_area(self, north, east, south, west):
        indices = np.array([])
        prev_selected_indices = self._selected_indices
        if self.data is not None and (north != south and east != west):
            lat, lon = self._latlon_data.T
            indices = ((lat <= north) & (lat >= south) &
                       (lon <= east) & (lon >= west))
            if self._selected_indices is not None:
                indices |= self._selected_indices
            self._selected_indices = indices
        else:
            self._selected_indices = None
        if np.any(self._selected_indices != prev_selected_indices):
            self.selectionChanged.emit(indices.nonzero()[0].tolist())
            self.redraw_markers_overlay_image(new_image=True)

    def set_map_provider(self, provider):
        self.evalJS('set_map_provider("{}");'.format(provider))

    def set_clustering(self, cluster_points):
        self.evalJS('''
            window.cluster_points = {};
            set_cluster_points();
        '''.format(int(cluster_points)))

    def _recompute_jittering_offsets(self):
        if not self._jittering:
            self._jittering_offsets = None
        elif self.data:
            # Calculate offsets randomly distributed within a circle
            screen_size = max(100, min(qApp.desktop().screenGeometry().width(),
                                       qApp.desktop().screenGeometry().height()))
            n = len(self.data)
            r = np.random.random(n)
            theta = np.random.uniform(0, 2*np.pi, n)
            xy_offsets = screen_size * self._jittering * np.c_[r * np.cos(theta),
                                                               r * np.sin(theta)]
            self._jittering_offsets = xy_offsets

    def set_jittering(self, jittering):
        """ In percent, i.e. jittering=3 means 3% of screen height and width """
        self._jittering = jittering / 100
        self._recompute_jittering_offsets()
        self.redraw_markers_overlay_image(new_image=True)

    @staticmethod
    def _legend_values(variable, values):
        strs = [variable.repr_val(val) for val in values]
        if any(len(val) > 10 for val in strs):
            if isinstance(variable, TimeVariable):
                strs = [s.replace(' ', '<br>') for s in strs]
            elif variable.is_continuous:
                strs = ['{:.4e}'.format(val) for val in values]
            elif variable.is_discrete:
                strs = [s if len(s) <= 12 else (s[:8] + '…' + s[-3:])
                        for s in strs]
        return strs

    def set_marker_color(self, attr, update=True):
        try:
            self._color_attr = variable = self.data.domain[attr]
            if len(self.data) == 0:
                raise Exception
        except Exception:
            self._color_attr = None
            self._legend_colors = []
        else:
            if variable.is_continuous:
                self._raw_color_values = values = self.data.get_column_view(variable)[0].astype(float)
                self._scaled_color_values = scale(values)
                self._colorgen = ContinuousPaletteGenerator(*variable.colors)
                min = np.nanmin(values)
                self._legend_colors = (['c',
                                        self._legend_values(variable, [min, np.nanmax(values)]),
                                        [color_to_hex(i) for i in variable.colors if i]]
                                       if not np.isnan(min) else [])
            elif variable.is_discrete:
                _values = np.asarray(self.data.domain[attr].values)
                __values = self.data.get_column_view(variable)[0].astype(np.uint16)
                self._raw_color_values = _values[__values]  # The joke's on you
                self._scaled_color_values = __values
                self._colorgen = ColorPaletteGenerator(len(variable.colors), variable.colors)
                self._legend_colors = ['d',
                                       self._legend_values(variable, range(len(_values))),
                                       list(_values),
                                       [color_to_hex(self._colorgen.getRGB(i))
                                        for i in range(len(_values))]]
        finally:
            if update:
                self.redraw_markers_overlay_image(new_image=True)

    def set_marker_label(self, attr, update=True):
        try:
            self._label_attr = variable = self.data.domain[attr]
            if len(self.data) == 0:
                raise Exception
        except Exception:
            self._label_attr = None
        else:
            if variable.is_continuous or variable.is_string:
                self._label_values = self.data.get_column_view(variable)[0]
            elif variable.is_discrete:
                _values = np.asarray(self.data.domain[attr].values)
                __values = self.data.get_column_view(variable)[0].astype(np.uint16)
                self._label_values = _values[__values]  # The design had lead to poor code for ages
        finally:
            if update:
                self.redraw_markers_overlay_image(new_image=True)

    def set_marker_shape(self, attr, update=True):
        try:
            self._shape_attr = variable = self.data.domain[attr]
            if len(self.data) == 0:
                raise Exception
        except Exception:
            self._shape_attr = None
            self._legend_shapes = []
        else:
            assert variable.is_discrete
            _values = np.asarray(self.data.domain[attr].values)
            self._shape_values = __values = self.data.get_column_view(variable)[0].astype(np.uint16)
            self._raw_shape_values = _values[__values]
            self._legend_shapes = [self._legend_values(variable, range(len(_values))),
                                   list(_values)]
        finally:
            if update:
                self.redraw_markers_overlay_image(new_image=True)

    def set_marker_size(self, attr, update=True):
        try:
            self._size_attr = variable = self.data.domain[attr]
            if len(self.data) == 0:
                raise Exception
        except Exception:
            self._size_attr = None
            self._legend_sizes = []
        else:
            assert variable.is_continuous
            self._raw_sizes = values = self.data.get_column_view(variable)[0].astype(float)
            # Note, [5, 60] is also hardcoded in legend-size-indicator.svg
            self._sizes = scale(values, 5, 60).astype(np.uint8)
            min = np.nanmin(values)
            self._legend_sizes = self._legend_values(variable,
                                                     [min, np.nanmax(values)]) if not np.isnan(min) else []
        finally:
            if update:
                self.redraw_markers_overlay_image(new_image=True)

    def set_marker_size_coefficient(self, size):
        self._size_coef = size / 100
        self.evalJS('''set_marker_size_coefficient({});'''.format(size / 100))
        if not self.is_js_path:
            self.redraw_markers_overlay_image(new_image=True)

    def set_marker_opacity(self, opacity):
        self._opacity = 255 * opacity // 100
        self.evalJS('''set_marker_opacity({});'''.format(opacity / 100))
        if not self.is_js_path:
            self.redraw_markers_overlay_image(new_image=True)

    def set_model(self, model):
        self.model = model
        self.evalJS('clear_heatmap()' if model is None else 'reset_heatmap()')

    def recompute_heatmap(self, points):
        if self.model is None or self.data is None:
            self.exposeObject('model_predictions', {})
            self.evalJS('draw_heatmap()')
            return

        latlons = np.array(points)
        table = Table(Domain([self.lat_attr, self.lon_attr]), latlons)
        try:
            predictions = self.model(table)
        except Exception as e:
            self._owwidget.Error.model_error(e)
            return
        else:
            self._owwidget.Error.model_error.clear()

        class_var = self.model.domain.class_var
        is_regression = class_var.is_continuous
        if is_regression:
            predictions = scale(np.round(predictions, 7))  # Avoid small errors
            kwargs = dict(
                extrema=self._legend_values(class_var, [np.nanmin(predictions),
                                                        np.nanmax(predictions)]))
        else:
            colorgen = ColorPaletteGenerator(len(class_var.values), class_var.colors)
            predictions = colorgen.getRGB(predictions)
            kwargs = dict(
                legend_labels=self._legend_values(class_var, range(len(class_var.values))),
                full_labels=list(class_var.values),
                colors=[color_to_hex(colorgen.getRGB(i))
                        for i in range(len(class_var.values))])
        self.exposeObject('model_predictions', dict(data=predictions, **kwargs))
        self.evalJS('draw_heatmap()')

    def _update_legend(self, is_js_path=False):
        self.evalJS('''
            window.legend_colors = %s;
            window.legend_shapes = %s;
            window.legend_sizes  = %s;
            legendControl.remove();
            legendControl.addTo(map);
        ''' % (self._legend_colors,
               self._legend_shapes if is_js_path else [],
               self._legend_sizes))

    def _update_js_markers(self, visible, in_subset):
        self._visible = visible
        latlon = self._latlon_data
        self.exposeObject('latlon_data', dict(data=latlon[visible]))
        self.exposeObject('jittering_offsets',
                          self._jittering_offsets[visible] if self._jittering_offsets is not None else [])
        self.exposeObject('selected_markers', dict(data=(self._selected_indices[visible]
                                                         if self._selected_indices is not None else 0)))
        self.exposeObject('in_subset', in_subset.astype(np.int8))
        if not self._color_attr:
            self.exposeObject('color_attr', dict())
        else:
            colors = [color_to_hex(rgb)
                      for rgb in self._colorgen.getRGB(self._scaled_color_values[visible])]
            self.exposeObject('color_attr',
                              dict(name=str(self._color_attr), values=colors,
                                   raw_values=self._raw_color_values[visible]))
        if not self._label_attr:
            self.exposeObject('label_attr', dict())
        else:
            self.exposeObject('label_attr',
                              dict(name=str(self._label_attr),
                                   values=self._label_values[visible]))
        if not self._shape_attr:
            self.exposeObject('shape_attr', dict())
        else:
            self.exposeObject('shape_attr',
                              dict(name=str(self._shape_attr),
                                   values=self._shape_values[visible],
                                   raw_values=self._raw_shape_values[visible]))
        if not self._size_attr:
            self.exposeObject('size_attr', dict())
        else:
            self.exposeObject('size_attr',
                              dict(name=str(self._size_attr),
                                   values=self._sizes[visible],
                                   raw_values=self._raw_sizes[visible]))
        self.evalJS('''
            window.latlon_data = latlon_data.data;
            window.selected_markers = selected_markers.data;
            add_markers(latlon_data);
        ''')

    class Projection:
        """This should somewhat model Leaflet's Web Mercator (EPSG:3857).

        Reverse-engineered from L.Map.latlngToContainerPoint().
        """
        @staticmethod
        def latlon_to_easting_northing(lat, lon):
            R = 6378137
            MAX_LATITUDE = 85.0511287798
            DEG = np.pi / 180

            lat = np.clip(lat, -MAX_LATITUDE, MAX_LATITUDE)
            sin = np.sin(DEG * lat)
            x = R * DEG * lon
            y = R / 2 * np.log((1 + sin) / (1 - sin))
            return x, y

        @staticmethod
        def easting_northing_to_pixel(x, y, zoom_level, pixel_origin, map_pane_pos):
            R = 6378137
            PROJ_SCALE = .5 / (np.pi * R)

            zoom_scale = 256 * (2 ** zoom_level)
            x = (zoom_scale * (PROJ_SCALE * x + .5)).round() + (map_pane_pos[0] - pixel_origin[0])
            y = (zoom_scale * (-PROJ_SCALE * y + .5)).round() + (map_pane_pos[1] - pixel_origin[1])
            return x, y

    N_POINTS_PER_ITER = 666

    def redraw_markers_overlay_image(self, *args, new_image=False):
        if not args and not self._drawing_args or self.data is None:
            return

        if args:
            self._drawing_args = args
        north, east, south, west, width, height, zoom, origin, map_pane_pos = self._drawing_args

        lat, lon = self._latlon_data.T
        visible = ((lat <= north) & (lat >= south) &
                   (lon <= east) & (lon >= west)).nonzero()[0]
        in_subset = (np.in1d(self.data.ids, self._subset_ids)
                     if self._subset_ids.size else
                     np.tile(True, len(lon)))

        is_js_path = self.is_js_path = len(visible) < self.N_POINTS_PER_ITER

        self._update_legend(is_js_path)

        np.random.shuffle(visible)
        # Sort points in subset to be painted last
        visible = visible[np.lexsort((in_subset[visible],))]

        if is_js_path:
            self.evalJS('clear_markers_overlay_image()')
            self._update_js_markers(visible, in_subset[visible])
            self._owwidget.disable_some_controls(False)
            return

        self.evalJS('clear_markers_js();')
        self._owwidget.disable_some_controls(True)

        selected = (self._selected_indices
                    if self._selected_indices is not None else
                    np.zeros(len(lat), dtype=bool))
        cur = 0

        im = QImage(self._overlay_image_path)
        if im.isNull() or self._prev_origin != origin or new_image:
            im = QImage(width, height, QImage.Format_ARGB32)
            im.fill(Qt.transparent)
        else:
            dx, dy = self._prev_map_pane_pos - map_pane_pos
            im = im.copy(dx, dy, width, height)
        self._prev_map_pane_pos = np.array(map_pane_pos)
        self._prev_origin = origin

        painter = QPainter(im)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.evalJS('clear_markers_overlay_image(); markersImageLayer.setBounds(map.getBounds());0')

        self._image_token = image_token = np.random.random()

        n_iters = np.ceil(len(visible) / self.N_POINTS_PER_ITER)

        def add_points():
            nonlocal cur, image_token
            if image_token != self._image_token:
                return
            batch = visible[cur:cur + self.N_POINTS_PER_ITER]

            batch_lat = lat[batch]
            batch_lon = lon[batch]

            x, y = self.Projection.latlon_to_easting_northing(batch_lat, batch_lon)
            x, y = self.Projection.easting_northing_to_pixel(x, y, zoom, origin, map_pane_pos)

            if self._jittering:
                dx, dy = self._jittering_offsets[batch].T
                x, y = x + dx, y + dy

            colors = (self._colorgen.getRGB(self._scaled_color_values[batch]).tolist()
                      if self._color_attr else
                      repeat((0xff, 0, 0)))
            sizes = self._size_coef * \
                (self._sizes[batch] if self._size_attr else np.tile(10, len(batch)))

            opacity_subset, opacity_rest = self._opacity, int(.8 * self._opacity)
            for x, y, is_selected, size, color, _in_subset in \
                    zip(x, y, selected[batch], sizes, colors, in_subset[batch]):

                pensize2, selpensize2 = (.35, 1.5) if size >= 5 else (.15, .7)
                pensize2 *= self._size_coef
                selpensize2 *= self._size_coef

                size2 = size / 2
                if is_selected:
                    painter.setPen(QPen(QBrush(Qt.green), 2 * selpensize2))
                    painter.drawEllipse(x - size2 - selpensize2,
                                        y - size2 - selpensize2,
                                        size + selpensize2,
                                        size + selpensize2)
                color = QColor(*color)
                if _in_subset:
                    color.setAlpha(opacity_subset)
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(QBrush(color.darker(180)), 2 * pensize2))
                else:
                    color.setAlpha(opacity_rest)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QBrush(color.lighter(120)), 2 * pensize2))

                painter.drawEllipse(x - size2 - pensize2,
                                    y - size2 - pensize2,
                                    size + pensize2,
                                    size + pensize2)

            im.save(self._overlay_image_path, 'PNG')
            self.evalJS('markersImageLayer.setUrl("{}#{}"); 0;'
                        .format(self.toFileURL(self._overlay_image_path),
                                np.random.random()))

            cur += self.N_POINTS_PER_ITER
            if cur < len(visible):
                QTimer.singleShot(10, add_points)
                self._owwidget.progressBarAdvance(100 / n_iters, None)
            else:
                self._owwidget.progressBarFinished(None)

        self._owwidget.progressBarFinished(None)
        self._owwidget.progressBarInit(None)
        QTimer.singleShot(10, add_points)

    def set_subset_ids(self, ids):
        self._subset_ids = ids
        self.redraw_markers_overlay_image(new_image=True)

    def toggle_legend(self, visible):
        self.evalJS('''
            $(".legend").{0}();
            window.legend_hidden = "{0}";
        '''.format('show' if visible else 'hide'))


class OWMap(widget.OWWidget):
    name = 'Geo Map'
    description = 'Show data points on a world map.'
    icon = "icons/Map.svg"

    inputs = [("Data", Table, "set_data", widget.Default),
              ("Data Subset", Table, "set_subset"),
              ("Learner", Learner, "set_learner")]

    outputs = [("Selected Data", Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Table)]

    settingsHandler = settings.DomainContextHandler()

    want_main_area = True

    autocommit = settings.Setting(True)
    tile_provider = settings.Setting('Black and white')
    lat_attr = settings.ContextSetting('')
    lon_attr = settings.ContextSetting('')
    class_attr = settings.ContextSetting('(None)')
    color_attr = settings.ContextSetting('')
    label_attr = settings.ContextSetting('')
    shape_attr = settings.ContextSetting('')
    size_attr = settings.ContextSetting('')
    opacity = settings.Setting(100)
    zoom = settings.Setting(100)
    jittering = settings.Setting(0)
    cluster_points = settings.Setting(False)
    show_legend = settings.Setting(True)

    TILE_PROVIDERS = OrderedDict((
        ('Black and white', 'OpenStreetMap.BlackAndWhite'),
        ('OpenStreetMap', 'OpenStreetMap.Mapnik'),
        ('Topographic', 'Thunderforest.OpenCycleMap'),
        ('Topographic 2', 'Thunderforest.Outdoors'),
        ('Satellite', 'Esri.WorldImagery'),
        ('Print', 'Stamen.TonerLite'),
        ('Dark', 'CartoDB.DarkMatter'),
        ('Watercolor', 'Stamen.Watercolor'),
    ))

    class Error(widget.OWWidget.Error):
        model_error = widget.Msg("Error predicting: {}")
        learner_error = widget.Msg("Error modelling: {}")

    UserAdviceMessages = [
        widget.Message(
            'Select markers by holding <b><kbd>Shift</kbd></b> key and dragging '
            'a rectangle around them. Clear the selection by clicking anywhere.',
            'shift-selection')
    ]

    graph_name = "map"

    def __init__(self):
        super().__init__()
        self.map = map = LeafletMap(self)
        self.mainArea.layout().addWidget(map)
        self.selection = None
        self.data = None
        self.learner = None

        def selectionChanged(indices):
            self.selection = self.data[indices] if self.data is not None and indices else None
            self._indices = indices
            self.commit()

        map.selectionChanged.connect(selectionChanged)

        def _set_map_provider():
            map.set_map_provider(self.TILE_PROVIDERS[self.tile_provider])

        box = gui.vBox(self.controlArea, 'Map')
        gui.comboBox(box, self, 'tile_provider',
                     orientation=Qt.Horizontal,
                     label='Map:',
                     items=tuple(self.TILE_PROVIDERS.keys()),
                     sendSelectedValue=True,
                     callback=_set_map_provider)

        self._latlon_model = DomainModel(
            parent=self, valid_types=ContinuousVariable)
        self._class_model = DomainModel(
            parent=self, placeholder='(None)', valid_types=DomainModel.PRIMITIVE)
        self._color_model = DomainModel(
            parent=self, placeholder='(Same color)', valid_types=DomainModel.PRIMITIVE)
        self._shape_model = DomainModel(
            parent=self, placeholder='(Same shape)', valid_types=DiscreteVariable)
        self._size_model = DomainModel(
            parent=self, placeholder='(Same size)', valid_types=ContinuousVariable)
        self._label_model = DomainModel(
            parent=self, placeholder='(No labels)')

        def _set_lat_long():
            self.map.set_data(self.data, self.lat_attr, self.lon_attr)
            self.train_model()

        self._combo_lat = combo = gui.comboBox(
            box, self, 'lat_attr', orientation=Qt.Horizontal,
            label='Latitude:', sendSelectedValue=True, callback=_set_lat_long)
        combo.setModel(self._latlon_model)
        self._combo_lon = combo = gui.comboBox(
            box, self, 'lon_attr', orientation=Qt.Horizontal,
            label='Longitude:', sendSelectedValue=True, callback=_set_lat_long)
        combo.setModel(self._latlon_model)

        def _toggle_legend():
            self.map.toggle_legend(self.show_legend)

        gui.checkBox(box, self, 'show_legend', label='Show legend',
                     callback=_toggle_legend)

        box = gui.vBox(self.controlArea, 'Overlay')
        self._combo_class = combo = gui.comboBox(
            box, self, 'class_attr', orientation=Qt.Horizontal,
            label='Target:', sendSelectedValue=True, callback=self.train_model
        )
        self.controls.class_attr.setModel(self._class_model)
        self.set_learner(self.learner)

        box = gui.vBox(self.controlArea, 'Points')
        self._combo_color = combo = gui.comboBox(
            box, self, 'color_attr',
            orientation=Qt.Horizontal,
            label='Color:',
            sendSelectedValue=True,
            callback=lambda: self.map.set_marker_color(self.color_attr))
        combo.setModel(self._color_model)
        self._combo_label = combo = gui.comboBox(
            box, self, 'label_attr',
            orientation=Qt.Horizontal,
            label='Label:',
            sendSelectedValue=True,
            callback=lambda: self.map.set_marker_label(self.label_attr))
        combo.setModel(self._label_model)
        self._combo_shape = combo = gui.comboBox(
            box, self, 'shape_attr',
            orientation=Qt.Horizontal,
            label='Shape:',
            sendSelectedValue=True,
            callback=lambda: self.map.set_marker_shape(self.shape_attr))
        combo.setModel(self._shape_model)
        self._combo_size = combo = gui.comboBox(
            box, self, 'size_attr',
            orientation=Qt.Horizontal,
            label='Size:',
            sendSelectedValue=True,
            callback=lambda: self.map.set_marker_size(self.size_attr))
        combo.setModel(self._size_model)

        def _set_opacity():
            map.set_marker_opacity(self.opacity)

        def _set_zoom():
            map.set_marker_size_coefficient(self.zoom)

        def _set_jittering():
            map.set_jittering(self.jittering)

        def _set_clustering():
            map.set_clustering(self.cluster_points)

        self._opacity_slider = gui.hSlider(
            box, self, 'opacity', None, 1, 100, 5,
            label='Opacity:', labelFormat=' %d%%',
            callback=_set_opacity)
        self._zoom_slider = gui.valueSlider(
            box, self, 'zoom', None, values=(20, 50, 100, 200, 300, 400, 500, 700, 1000),
            label='Symbol size:', labelFormat=' %d%%',
            callback=_set_zoom)
        self._jittering = gui.valueSlider(
            box, self, 'jittering', label='Jittering:', values=(0, .5, 1, 2, 5),
            labelFormat=' %.1f%%', ticks=True,
            callback=_set_jittering)
        self._clustering_check = gui.checkBox(
            box, self, 'cluster_points', label='Cluster points',
            callback=_set_clustering)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Send Selection')

        QTimer.singleShot(0, _set_map_provider)
        QTimer.singleShot(0, _toggle_legend)
        QTimer.singleShot(0, _set_opacity)
        QTimer.singleShot(0, _set_zoom)
        QTimer.singleShot(0, _set_jittering)
        QTimer.singleShot(0, _set_clustering)

    autocommit = settings.Setting(True)

    def __del__(self):
        self.progressBarFinished(None)
        self.map = None

    def commit(self):
        self.send('Selected Data', self.selection)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(self.data, self._indices))

    def set_data(self, data):
        self.data = data

        self.closeContext()

        if data is None or not len(data):
            return self.clear()

        all_vars = list(chain(self.data.domain.variables, self.data.domain.metas))

        domain = data is not None and data.domain
        for model in (self._latlon_model,
                      self._class_model,
                      self._color_model,
                      self._shape_model,
                      self._size_model,
                      self._label_model):
            model.set_domain(domain)

        def _find_lat_lon():
            lat_attr = next(
                (attr for attr in all_vars
                 if attr.is_continuous and
                    attr.name.lower().startswith(('latitude', 'lat'))), None)
            lon_attr = next(
                (attr for attr in all_vars
                 if attr.is_continuous and
                    attr.name.lower().startswith(('longitude', 'lng', 'long', 'lon'))), None)

            def _all_between(vals, min, max):
                return np.all((min <= vals) & (vals <= max))

            if not lat_attr:
                for attr in all_vars:
                    if attr.is_continuous:
                        values = np.nan_to_num(data.get_column_view(attr)[0].astype(float))
                        if _all_between(values, -90, 90):
                            lat_attr = attr
                            break
            if not lon_attr:
                for attr in all_vars:
                    if attr.is_continuous:
                        values = np.nan_to_num(data.get_column_view(attr)[0].astype(float))
                        if _all_between(values, -180, 180):
                            lon_attr = attr
                            break

            return lat_attr, lon_attr

        lat, lon = _find_lat_lon()
        if lat or lon:
            self._combo_lat.setCurrentIndex(-1 if lat is None else self._latlon_model.indexOf(lat))
            self._combo_lon.setCurrentIndex(-1 if lat is None else self._latlon_model.indexOf(lon))
            self.lat_attr = lat.name
            self.lon_attr = lon.name

        if data.domain.class_var:
            self.color_attr = data.domain.class_var.name
        elif len(self._color_model):
            self._combo_color.setCurrentIndex(0)
        if len(self._shape_model):
            self._combo_shape.setCurrentIndex(0)
        if len(self._size_model):
            self._combo_size.setCurrentIndex(0)
        if len(self._label_model):
            self._combo_label.setCurrentIndex(0)
        if len(self._class_model):
            self._combo_class.setCurrentIndex(0)

        self.openContext(data)

        self.map.set_data(self.data, self.lat_attr, self.lon_attr)
        self.map.set_marker_color(self.color_attr, update=False)
        self.map.set_marker_label(self.label_attr, update=False)
        self.map.set_marker_shape(self.shape_attr, update=False)
        self.map.set_marker_size(self.size_attr, update=True)

    def set_subset(self, subset):
        self.map.set_subset_ids(subset.ids if subset is not None else np.array([]))

    def handleNewSignals(self):
        super().handleNewSignals()
        self.train_model()

    def set_learner(self, learner):
        self.learner = learner
        self.controls.class_attr.setEnabled(learner is not None)
        self.controls.class_attr.setToolTip(
            'Needs a Learner input for modelling.' if learner is None else '')

    def train_model(self):
        model = None
        self.Error.clear()
        if self.data and self.learner and self.class_attr != '(None)':
            domain = self.data.domain
            if self.lat_attr and self.lon_attr and self.class_attr in domain:
                domain = Domain([domain[self.lat_attr], domain[self.lon_attr]],
                                [domain[self.class_attr]])  # I am retarded
                train = self.data.transform(domain)
                try:
                    model = self.learner(train)
                except Exception as e:
                    self.Error.learner_error(e)
        self.map.set_model(model)

    def disable_some_controls(self, disabled):
        tooltip = (
            "Available when the zoom is close enough to have "
            "<{} points in the viewport.".format(self.map.N_POINTS_PER_ITER)
            if disabled else '')
        for widget in (self._combo_label,
                       self._combo_shape,
                       self._clustering_check):
            widget.setDisabled(disabled)
            widget.setToolTip(tooltip)

    def clear(self):
        self.map.set_data(None, '', '')
        for model in (self._latlon_model,
                      self._class_model,
                      self._color_model,
                      self._shape_model,
                      self._size_model,
                      self._label_model):
            model.set_domain(None)
        self.lat_attr = self.lon_attr = self.class_attr = self.color_attr = \
        self.label_attr = self.shape_attr = self.size_attr = ''


def test_main():
    from AnyQt.QtWidgets import QApplication
    from Orange.regression import KNNRegressionLearner as Learner
    from Orange.classification import KNNLearner as Learner
    a = QApplication([])

    ow = OWMap()
    ow.show()
    ow.raise_()
    data = Table('philadelphia-crime')
    ow.set_data(data)

    QTimer.singleShot(10, lambda: ow.set_learner(Learner(20)))

    ow.handleNewSignals()
    a.exec()
    ow.saveSettings()

if __name__ == "__main__":
    test_main()
