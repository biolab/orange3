"""Pythagorean forest widget for visualizing random forests."""
from math import log, sqrt

import numpy as np
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.base import RandomForest, Tree
from Orange.classification.random_forest import RandomForestClassifier
from Orange.classification.tree import TreeClassifier
from Orange.data import Table
from Orange.regression.random_forest import RandomForestRegressor
from Orange.regression.tree import TreeRegressor
from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from Orange.widgets.visualize.pythagorastreeviewer import PythagorasTreeViewer
from Orange.widgets.visualize.utils.owgrid import (
    OWGrid,
    SelectableGridItem,
    ZoomableGridItem
)
from Orange.widgets.visualize.utils.tree.skltreeadapter import \
    SklTreeAdapter
from Orange.widgets.widget import OWWidget


class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean Forest'
    description = 'Pythagorean forest for visualising random forests.'
    icon = 'icons/PythagoreanForest.svg'

    priority = 1001

    inputs = [('Random forest', RandomForest, 'set_rf')]
    outputs = [('Tree', Tree)]

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    zoom = settings.Setting(50)
    selected_tree_index = settings.ContextSetting(-1)

    CLASSIFICATION, REGRESSION = range(2)

    def __init__(self):
        super().__init__()
        # Instance variables
        self.forest_type = self.CLASSIFICATION
        self.model = None
        self.forest_adapter = None
        self.dataset = None
        self.clf_dataset = None
        # We need to store refernces to the trees and grid items
        self.grid_items, self.ptrees = [], []

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale)),
        ]

        self.REGRESSION_COLOR_CALC = [
            ('None', lambda _, __: QtGui.QColor(255, 255, 255)),
            ('Class mean', self._color_class_mean),
            ('Standard deviation', self._color_stddev),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Forest')
        self.ui_info = gui.widgetLabel(box_info, label='')

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        self.ui_depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            callback=self.max_depth_changed)
        self.ui_target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation='horizontal', items=[], contentsLength=8,
            callback=self.target_colors_changed)
        self.ui_size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.size_calc_changed)
        self.ui_zoom_slider = gui.hSlider(
            box_display, self, 'zoom', label='Zoom', ticks=False, minValue=20,
            maxValue=150, callback=self.zoom_changed, createLabel=False)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # MAIN AREA
        self.scene = QtGui.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.grid = OWGrid()
        self.grid.geometryChanged.connect(self._update_scene_rect)
        self.scene.addItem(self.grid)

        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.mainArea.layout().addWidget(self.view)

        self.resize(800, 500)

        self.clear()

    def set_rf(self, model=None):
        """When a different forest is given."""
        self.clear()
        self.model = model

        if model is not None:
            if isinstance(model, RandomForestClassifier):
                self.forest_type = self.CLASSIFICATION
            elif isinstance(model, RandomForestRegressor):
                self.forest_type = self.REGRESSION
            else:
                raise RuntimeError('Invalid type of forest.')

            self.forest_adapter = self._get_forest_adapter(self.model)
            self.color_palette = self._type_specific('_get_color_palette')()
            self._draw_trees()

            self.dataset = model.instances
            # this bit is important for the regression classifier
            if self.dataset is not None and \
                    self.dataset.domain != model.domain:
                self.clf_dataset = Table.from_table(
                    self.model.domain, self.dataset)
            else:
                self.clf_dataset = self.dataset

            self._update_info_box()
            self._type_specific('_update_target_class_combo')()
            self._update_depth_slider()

            self.selected_tree_index = -1

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.forest_adapter = None
        self.ptrees = []
        self.grid_items = []
        self.grid.clear()

        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    # CONTROL AREA CALLBACKS
    def max_depth_changed(self):
        """When the max depth slider is changed."""
        for tree in self.ptrees:
            tree.set_depth_limit(self.depth_limit)

    def target_colors_changed(self):
        """When the target class or coloring method is changed."""
        for tree in self.ptrees:
            tree.target_class_has_changed()

    def size_calc_changed(self):
        """When the size calculation of the trees is changed."""
        if self.model is not None:
            self.forest_adapter = self._get_forest_adapter(self.model)
            self.grid.clear()
            self._draw_trees()
            # Keep the selected item
            if self.selected_tree_index != -1:
                self.grid_items[self.selected_tree_index].setSelected(True)
            self.max_depth_changed()

    def zoom_changed(self):
        """When we update the "Zoom" slider."""
        for item in self.grid_items:
            item.set_max_size(self._calculate_zoom(self.zoom))

        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

    # MODEL CHANGED METHODS
    def _update_info_box(self):
        self.ui_info.setText(
            'Trees: {}'.format(len(self.forest_adapter.get_trees()))
        )

    def _update_depth_slider(self):
        self.depth_limit = self._get_max_depth()

        self.ui_depth_slider.parent().setEnabled(True)
        self.ui_depth_slider.setMaximum(self.depth_limit)
        self.ui_depth_slider.setValue(self.depth_limit)

    # MODEL CLEARED METHODS
    def _clear_info_box(self):
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.target_class_index = 0
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def _clear_depth_slider(self):
        self.ui_depth_slider.parent().setEnabled(False)
        self.ui_depth_slider.setMaximum(0)

    # HELPFUL METHODS
    def _get_max_depth(self):
        return max([tree.tree_adapter.max_depth for tree in self.ptrees])

    def _get_forest_adapter(self, model):
        return SklRandomForestAdapter(
            model,
            model.domain,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def _draw_trees(self):
        self.grid_items, self.ptrees = [], []

        with self.progressBar(len(self.forest_adapter.get_trees())) as prg:
            for tree in self.forest_adapter.get_trees():
                ptree = PythagorasTreeViewer(
                    None, tree,
                    node_color_func=self._type_specific('_get_node_color'),
                    interactive=False, padding=100)
                self.grid_items.append(GridItem(
                    ptree, self.grid, max_size=self._calculate_zoom(self.zoom)
                ))
                self.ptrees.append(ptree)
                prg.advance()
        self.grid.set_items(self.grid_items)
        # This is necessary when adding items for the first time
        if self.grid:
            width = (self.view.width() -
                     self.view.verticalScrollBar().width())
            self.grid.reflow(width)
            self.grid.setPreferredWidth(width)

    @staticmethod
    def _calculate_zoom(zoom_level):
        """Calculate the max size for grid items from zoom level setting."""
        return zoom_level * 5

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected tree to output."""
        if len(self.scene.selectedItems()) == 0:
            self.send('Tree', None)
            # The selected tree index should only reset when model changes
            if self.model is None:
                self.selected_tree_index = -1
            return

        selected_item = self.scene.selectedItems()[0]
        self.selected_tree_index = self.grid_items.index(selected_item)
        tree = self.model.skl_model.estimators_[self.selected_tree_index]

        if self.forest_type == self.CLASSIFICATION:
            obj = TreeClassifier(tree)
        else:
            obj = TreeRegressor(tree)
        obj.domain = self.model.domain
        obj.instances = self.model.instances

        obj.meta_target_class_index = self.target_class_index
        obj.meta_size_calc_idx = self.size_calc_idx
        obj.meta_size_log_scale = self.size_log_scale
        obj.meta_depth_limit = self.depth_limit

        self.send('Tree', obj)

    def send_report(self):
        """Send report."""
        self.report_plot()

    def _update_scene_rect(self):
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def resizeEvent(self, ev):
        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

        super().resizeEvent(ev)

    def _type_specific(self, method):
        """A best effort method getter that somewhat separates logic specific
        to classification and regression trees.
        This relies on conventional naming of specific methods, e.g.
        a method name _get_tooltip would need to be defined like so:
        _classification_get_tooltip and _regression_get_tooltip, since they are
        both specific.

        Parameters
        ----------
        method : str
            Method name that we would like to call.

        Returns
        -------
        callable or None

        """
        if self.forest_type == self.CLASSIFICATION:
            return getattr(self, '_classification' + method)
        elif self.forest_type == self.REGRESSION:
            return getattr(self, '_regression' + method)
        else:
            return None

    # CLASSIFICATION FOREST SPECIFIC METHODS
    def _classification_update_target_class_combo(self):
        self._clear_target_class_combo()
        self.ui_target_class_combo.addItem('None')
        values = [c.title() for c in
                  self.model.domain.class_vars[0].values]
        self.ui_target_class_combo.addItems(values)

    def _classification_get_color_palette(self):
        return [QtGui.QColor(*c) for c in self.model.domain.class_var.colors]

    def _classification_get_node_color(self, adapter, tree_node):
        # this is taken almost directly from the existing classification tree
        # viewer
        colors = self.color_palette
        distribution = adapter.get_distribution(tree_node.label)[0]
        total = np.sum(distribution)

        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = colors[self.target_class_index - 1].light(200 - 100 * p)
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = colors[int(modus)].light(400 - 300 * p)
        return color

    # REGRESSION FOREST SPECIFIC METHODS
    def _regression_update_target_class_combo(self):
        self._clear_target_class_combo()
        self.ui_target_class_combo.addItems(
            list(zip(*self.REGRESSION_COLOR_CALC))[0])
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def _regression_get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.forest_adapter.domain.class_var.colors)

    def _regression_get_node_color(self, adapter, tree_node):
        return self.REGRESSION_COLOR_CALC[self.target_class_index][1](
            adapter, tree_node
        )

    def _color_class_mean(self, adapter, tree_node):
        # calculate node colors relative to the mean of the node samples
        min_mean = np.min(self.clf_dataset.Y)
        max_mean = np.max(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        mean = np.mean(instances.Y)

        return self.color_palette[(mean - min_mean) / (max_mean - min_mean)]

    def _color_stddev(self, adapter, tree_node):
        # calculate node colors relative to the standard deviation in the node
        # samples
        min_mean, max_mean = 0, np.std(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        std = np.std(instances.Y)

        return self.color_palette[(std - min_mean) / (max_mean - min_mean)]


class GridItem(SelectableGridItem, ZoomableGridItem):
    """The grid item we will use in our grid."""
    pass


class SklRandomForestAdapter:
    """Take a `RandomForest` and wrap all the trees into the `TreeAdapter`
    instances that Pythagorean trees use."""
    def __init__(self, model, domain, adjust_weight=lambda x: x):
        self._adapters = []

        self._domain = domain

        self._trees = model.skl_model.estimators_
        self._domain = model.domain
        self._adjust_weight = adjust_weight

    def get_trees(self):
        """Get the tree adapters in the random forest."""
        if len(self._adapters) > 0:
            return self._adapters
        if len(self._trees) < 1:
            return self._adapters

        self._adapters = [
            SklTreeAdapter(tree.tree_, self._domain, self._adjust_weight)
            for tree in self._trees
        ]
        return self._adapters

    @property
    def domain(self):
        """Get the domain."""
        return self._domain
