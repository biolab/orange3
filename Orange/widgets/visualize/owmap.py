from itertools import chain
from collections import OrderedDict

import numpy as np

from AnyQt.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QTimer

from Orange.base import Model
from Orange.data.util import scale
from Orange.data import Table, Domain
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.webview import WebviewWidget
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, ContinuousPaletteGenerator

from os.path import join, dirname
OWMAP_URL = join(dirname(__file__), '_owmap', 'owmap.html')


class LeafletMap(WebviewWidget):
    selectionChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent,
                         url=QUrl(self.toFileURL(OWMAP_URL)),
                         debug=True,)
        self.jittering = 3
        self._owwidget = parent
        self.model = None

    def set_data(self, data, lat_attr, lon_attr):
        self.data = data

        if data is None or not (len(data) and lat_attr and lon_attr):
            self.evalJS('clear_markers();')
            return

        lat_attr = self.lat_attr = data.domain[lat_attr]
        lon_attr = self.lon_attr = data.domain[lon_attr]

        latlon_data = np.column_stack((data[:, lat_attr],
                                       data[:, lon_attr]))
        self.exposeObject('latlon_data', dict(data=latlon_data))

        self.evalJS('''
            window.latlon_data = latlon_data.data;
            add_markers(latlon_data);
        ''')

    @pyqtSlot('QVariantList')
    def _selected_indices(self, indices):
        self.selectionChanged.emit(sorted(map(int, indices)))

    def set_map_provider(self, provider):
        self.evalJS('set_map_provider("{}");'.format(provider))

    def set_jittering(self, jittering):
        """ In percent, i.e. jittering=3 means 3% of screen height and width """
        self.evalJS('''
            window.jittering_percent = {};
            set_jittering();
            if (window.jittering_percent == 0)
                clear_jittering();
        '''.format(jittering))

    def set_marker_color(self, attr):
        try:
            variable = self.data.domain[attr]
        except Exception:
            return self.evalJS('''
                window.color_attr = {};
                set_marker_colors([]);
            ''')

        if variable.is_continuous:
            values = np.ravel(self.data[:, variable])
            colorgen = ContinuousPaletteGenerator(*variable.colors)
            colors = colorgen[scale(values)]
        elif variable.is_discrete:
            _values = np.asarray(self.data.domain[attr].values)
            __values = np.ravel(self.data[:, variable]).astype(int)
            values = _values[__values]  # The joke's on you
            colorgen = ColorPaletteGenerator(len(variable.colors), variable.colors)
            colors = colorgen[__values]

        self.exposeObject('color_attr',
                          dict(name=str(attr), values=values, colors=colors))
        self.evalJS('set_marker_colors(color_attr.colors);')

    def set_marker_label(self, attr):
        try:
            variable = self.data.domain[attr]
        except Exception:
            return self.evalJS('''
                window.label_attr = {};
                set_marker_labels([]);
            ''')

        if variable.is_continuous or variable.is_string:
            values = np.ravel(self.data[:, variable])
        elif variable.is_discrete:
            _values = np.asarray(self.data.domain[attr].values)
            __values = np.ravel(self.data[:, variable]).astype(int)
            values = _values[__values]  # The design had lead to poor code for ages
        self.exposeObject('label_attr',
                          dict(name=str(attr), values=values))
        self.evalJS('set_marker_labels(label_attr.values);')

    def set_marker_shape(self, attr):
        try:
            variable = self.data.domain[attr]
        except Exception:
            return self.evalJS('''
                window.shape_attr = {};
                set_marker_shapes([]);
            ''')

        assert variable.is_discrete
        _values = np.asarray(self.data.domain[attr].values)
        __values = np.ravel(self.data[:, variable]).astype(int)
        values = _values[__values]
        self.exposeObject('shape_attr',
                          dict(name=str(attr), indices=__values, values=values))
        self.evalJS('''set_marker_shapes(shape_attr.indices);''')

    def set_marker_size(self, attr):
        try:
            variable = self.data.domain[attr]
        except Exception:
            return self.evalJS('''
                window.size_attr = {};
                set_marker_sizes([]);
            ''')

        assert variable.is_continuous
        values = np.ravel(self.data[:, variable])
        sizes = scale(values, 10, 60)
        self.exposeObject('size_attr',
                          dict(name=str(attr), sizes=sizes, values=values))
        self.evalJS('''set_marker_sizes(size_attr.sizes);''')

    def set_marker_size_coefficient(self, size):
        ...

    def set_marker_opacity(self, opacity):
        ...

    def set_model(self, model):
        self.model = model
        if model is not None:
            self.reset_heatmap()
        else:
            self.evalJS('clear_heatmap();')

    def reset_heatmap(self):
        if self.data is None:
            # We don't have lat/lon attrs (variables), so we won't be able
            # to construct the domain for the model later on anyway
            return
        self.evalJS('''reset_heatmap();''')

    @pyqtSlot('QVariantList')
    def latlon_viewport_extremes(self, points):
        if self.model is None:
            return

        latlons = np.array(points)
        table = Table(Domain([self.lat_attr, self.lon_attr]), latlons)
        try:
            predictions = self.model(table)
        except Exception as e:
            self._owwidget.Error.model_prediction(e)
            return
        else:
            self._owwidget.Error.model_prediction.clear()
        predictions = scale(np.round(predictions, 7))  # Avoid small errors
        self.exposeObject('model_predictions', dict(data=predictions))
        self.evalJS('draw_heatmap()')





class OWMap(widget.OWWidget):
    name = 'Map'
    description = 'Show data points on a world map.'
    icon = "icons/Map.svg"

    inputs = [("Data", Table, "set_data", widget.Default),
              ("Model", Model, "set_model")]

    outputs = [("Selected Data", Table, widget.Default)]

    settingsHandler = settings.DomainContextHandler()

    want_main_area = True

    autocommit = settings.Setting(True)
    tile_provider = settings.Setting('Black and white')
    lat_attr = settings.Setting('')
    lon_attr = settings.Setting('')
    color_attr = settings.Setting('')
    label_attr = settings.Setting('')
    shape_attr = settings.Setting('')
    size_attr = settings.Setting('')
    jittering = settings.Setting(0)

    JITTER_SIZES = [0, 1, 3, 6, 9]
    TILE_PROVIDERS = OrderedDict((
        ('Black and white', 'OpenStreetMap.BlackAndWhite'),
        ('OpenStreetMap', 'OpenStreetMap.Mapnik'),
        ('Topographic', 'OpenTopoMap'),
        ('Topographic 2', 'Thunderforest.OpenCycleMap'),
        ('Topographic 3', 'Thunderforest.Outdoors'),
        ('Satellite', 'Esri.WorldImagery'),
        ('Print', 'Stamen.TonerLite'),
        ('Light', 'CartoDB.Positron'),
        ('Dark', 'CartoDB.DarkMatter'),
        ('Railways', 'Thunderforest.Transport'),
        ('Watercolor', 'Stamen.Watercolor'),
    ))

    class Error(widget.OWWidget.Error):
        model_prediction = widget.Msg("Model couldn't predict: {}")

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

        def selectionChanged(indices):
            self.selection = self.data[indices] if self.data is not None and indices else None
            self.commit()

        map.selectionChanged.connect(selectionChanged)

        box = gui.vBox(self.controlArea, 'Map')
        gui.comboBox(box, self, 'tile_provider',
                     orientation=Qt.Horizontal,
                     label='Map:',
                     items=tuple(self.TILE_PROVIDERS.keys()),
                     sendSelectedValue=True,
                     callback=lambda: self.map.set_map_provider(self.TILE_PROVIDERS[self.tile_provider]))

        self._latlon_model = VariableListModel(parent=self)
        self._color_model = VariableListModel(parent=self)
        self._shape_model = VariableListModel(parent=self)
        self._size_model = VariableListModel(parent=self)
        self._label_model = VariableListModel(parent=self)

        def _set_lat_long():
            self.map.set_data(self.data, self.lat_attr, self.lon_attr)

        self._combo_lat = combo = gui.comboBox(
            box, self, 'lat_attr', orientation=Qt.Horizontal,
            label='Latitude:', sendSelectedValue=True, callback=_set_lat_long)
        combo.setModel(self._latlon_model)
        self._combo_lon = combo = gui.comboBox(
            box, self, 'lon_attr', orientation=Qt.Horizontal,
            label='Longitude:', sendSelectedValue=True, callback=_set_lat_long)
        combo.setModel(self._latlon_model)

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
        self._jittering = gui.valueSlider(
            box, self, 'jittering', label='Jittering:', values=(0, .5, 1, 2, 5),
            labelFormat=' %.1f%%', ticks=True,
            callback=lambda: self.map.set_jittering(self.jittering)
        )

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Send Selection')

    autocommit = settings.Setting(True)

    def commit(self):
        self.send('Selected Data', self.selection)

    def set_data(self, data):
        self.data = data

        if data is None:
            return self.clear()

        all_vars = list(chain(self.data.domain, self.data.domain.metas))
        continuous_vars = [var for var in all_vars if var.is_continuous]
        discrete_vars = [var for var in all_vars if var.is_discrete]
        primitive_vars = [var for var in all_vars if var.is_primitive()]
        self._latlon_model.wrap(continuous_vars)
        self._color_model.wrap(['(Same color)'] + primitive_vars)
        self._shape_model.wrap(['(Same shape)'] + discrete_vars)
        self._size_model.wrap(['(Same size)'] + continuous_vars)
        self._label_model.wrap(['(No labels)'] + all_vars)

        def _find_lat_lon():
            lat_attr = next(
                (attr for attr in data.domain
                 if attr.is_continuous and
                    attr.name.lower().startswith(('latitude', 'lat'))), None)
            lon_attr = next(
                (attr for attr in data.domain
                 if attr.is_continuous and
                    attr.name.lower().startswith(('longitude', 'lng', 'long', 'lon'))), None)
            return lat_attr, lon_attr

        lat, lon = _find_lat_lon()
        if lat or lon:
            self._combo_lat.setCurrentIndex(-1 if lat is None else continuous_vars.index(lat))
            self._combo_lon.setCurrentIndex(-1 if lat is None else continuous_vars.index(lon))
            self.map.set_data(self.data, lat, lon)
        self._combo_color.setCurrentIndex(0)
        self._combo_shape.setCurrentIndex(0)
        self._combo_size.setCurrentIndex(0)
        self._combo_label.setCurrentIndex(0)
        self._combo_class.setCurrentIndex(0)

    def set_model(self, model):
        self.map.set_model(model)

    def clear(self):
        self.map.set_data(None, '', '')
        self._latlon_model.wrap([])
        self._color_model.wrap(['(Same color)'])
        self._shape_model.wrap(['(Same shape)'])
        self._size_model.wrap(['(Same size)'])
        self._label_model.wrap(['(No labels)'])
        self.lat_attr = self.lon_attr = self.color_attr = \
        self.label_attr = self.shape_attr = self.size_attr = ''


def test_main():
    from AnyQt.QtWidgets import QApplication
    from Orange.regression import KNNRegressionLearner
    a = QApplication([])

    ow = OWMap()
    ow.show()
    ow.raise_()
    data = Table('philadelphia-crime')
    ow.set_data(data)

    QTimer.singleShot(10, lambda: ow.set_learner(KNNRegressionLearner()))

    ow.handleNewSignals()
    a.exec()
    ow.saveSettings()

if __name__ == "__main__":
    test_main()
