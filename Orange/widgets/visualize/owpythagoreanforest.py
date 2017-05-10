"""Pythagorean forest widget for visualizing random forests."""
from contextlib import contextmanager
from math import log, sqrt

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPainter
from AnyQt.QtWidgets import QSizePolicy, QGraphicsScene, QGraphicsView, QLabel

from Orange.base import RandomForestModel, TreeModel
from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTreeViewer,
    ContinuousTreeNode,
)
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

    inputs = [('Random forest', RandomForestModel, 'set_rf')]
    outputs = [('Tree', TreeModel)]

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    zoom = settings.Setting(50)
    selected_tree_index = settings.ContextSetting(-1)

    def __init__(self):
        super().__init__()
        self.model = None
        self.forest_adapter = None
        self.instances = None
        self.clf_dataset = None
        # We need to store refernces to the trees and grid items
        self.grid_items, self.ptrees = [], []
        # In some rare cases, we need to prevent commiting, the only one
        # that this currently helps is that when changing the size calculation
        # the trees are all recomputed, but we don't want to output a new tree
        # to keep things consistent with other ui controls.
        self.__prevent_commit = False

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x + 1)),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Forest')
        self.ui_info = gui.widgetLabel(box_info)

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        self.ui_depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            callback=self.update_depth)
        self.ui_target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation=Qt.Horizontal, items=[], contentsLength=8,
            callback=self.update_colors)
        self.ui_size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation=Qt.Horizontal,
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.update_size_calc)
        self.ui_zoom_slider = gui.hSlider(
            box_display, self, 'zoom', label='Zoom', ticks=False, minValue=20,
            maxValue=150, callback=self.zoom_changed, createLabel=False)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # MAIN AREA
        self.scene = QGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.grid = OWGrid()
        self.grid.geometryChanged.connect(self._update_scene_rect)
        self.scene.addItem(self.grid)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.mainArea.layout().addWidget(self.view)

        self.resize(800, 500)

        self.clear()

    def set_rf(self, model=None):
        """When a different forest is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.forest_adapter = self._get_forest_adapter(self.model)
            self._draw_trees()
            self.color_palette = self.forest_adapter.get_trees()[0]

            self.instances = model.instances
            # this bit is important for the regression classifier
            if self.instances is not None and self.instances.domain != model.domain:
                self.clf_dataset = self.instances.transform(self.model.domain)
            else:
                self.clf_dataset = self.instances

            self._update_info_box()
            self._update_target_class_combo()
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

    def update_depth(self):
        """When the max depth slider is changed."""
        for tree in self.ptrees:
            tree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        """When the target class or coloring method is changed."""
        for tree in self.ptrees:
            tree.target_class_changed(self.target_class_index)

    def update_size_calc(self):
        """When the size calculation of the trees is changed."""
        if self.model is not None:
            with self._prevent_commit():
                self.grid.clear()
                self._draw_trees()
                # Keep the selected item
                if self.selected_tree_index != -1:
                    self.grid_items[self.selected_tree_index].setSelected(True)
                self.update_depth()

    def zoom_changed(self):
        """When we update the "Zoom" slider."""
        for item in self.grid_items:
            item.set_max_size(self._calculate_zoom(self.zoom))

        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

    @contextmanager
    def _prevent_commit(self):
        try:
            self.__prevent_commit = True
            yield
        finally:
            self.__prevent_commit = False

    def _update_info_box(self):
        self.ui_info.setText('Trees: {}'.format(len(self.forest_adapter.get_trees())))

    def _update_depth_slider(self):
        self.depth_limit = self._get_max_depth()

        self.ui_depth_slider.parent().setEnabled(True)
        self.ui_depth_slider.setMaximum(self.depth_limit)
        self.ui_depth_slider.setValue(self.depth_limit)

    def _clear_info_box(self):
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.target_class_index = 0
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def _clear_depth_slider(self):
        self.ui_depth_slider.parent().setEnabled(False)
        self.ui_depth_slider.setMaximum(0)

    def _get_max_depth(self):
        return max(tree.tree_adapter.max_depth for tree in self.ptrees)

    def _get_forest_adapter(self, model):
        return SklRandomForestAdapter(model)

    @contextmanager
    def disable_ui(self):
        """Temporarly disable the UI while trees may be redrawn."""
        try:
            self.ui_size_calc_combo.setEnabled(False)
            self.ui_depth_slider.setEnabled(False)
            self.ui_target_class_combo.setEnabled(False)
            self.ui_zoom_slider.setEnabled(False)
            yield
        finally:
            self.ui_size_calc_combo.setEnabled(True)
            self.ui_depth_slider.setEnabled(True)
            self.ui_target_class_combo.setEnabled(True)
            self.ui_zoom_slider.setEnabled(True)

    def _draw_trees(self):
        self.grid_items, self.ptrees = [], []

        num_trees = len(self.forest_adapter.get_trees())
        with self.progressBar(num_trees) as prg, self.disable_ui():
            for tree in self.forest_adapter.get_trees():
                ptree = PythagorasTreeViewer(
                    None, tree, interactive=False, padding=100,
                    target_class_index=self.target_class_index,
                    weight_adjustment=self.SIZE_CALCULATION[self.size_calc_idx][1]
                )
                grid_item = GridItem(
                    ptree, self.grid, max_size=self._calculate_zoom(self.zoom)
                )
                # We don't want to show flickering while the trees are being
                grid_item.setVisible(False)

                self.grid_items.append(grid_item)
                self.ptrees.append(ptree)
                prg.advance()

            self.grid.set_items(self.grid_items)
            # This is necessary when adding items for the first time
            if self.grid:
                width = (self.view.width() - self.view.verticalScrollBar().width())
                self.grid.reflow(width)
                self.grid.setPreferredWidth(width)
                # After drawing is complete, we show the trees
                for grid_item in self.grid_items:
                    grid_item.setVisible(True)

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
        if self.__prevent_commit:
            return

        if not self.scene.selectedItems():
            self.send('Tree', None)
            # The selected tree index should only reset when model changes
            if self.model is None:
                self.selected_tree_index = -1
            return

        selected_item = self.scene.selectedItems()[0]
        self.selected_tree_index = self.grid_items.index(selected_item)
        tree = self.model.trees[self.selected_tree_index]
        tree.instances = self.instances
        tree.meta_target_class_index = self.target_class_index
        tree.meta_size_calc_idx = self.size_calc_idx
        tree.meta_depth_limit = self.depth_limit

        self.send('Tree', tree)

    def send_report(self):
        """Send report."""
        self.report_plot()

    def _update_scene_rect(self):
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        label = [x for x in self.ui_target_class_combo.parent().children()
                 if isinstance(x, QLabel)][0]

        if self.instances.domain.has_discrete_class:
            label_text = 'Target class'
            values = [c.title() for c in self.instances.domain.class_vars[0].values]
            values.insert(0, 'None')
        else:
            label_text = 'Node color'
            values = list(ContinuousTreeNode.COLOR_METHODS.keys())
        label.setText(label_text)
        self.ui_target_class_combo.addItems(values)
        self.ui_target_class_combo.setCurrentIndex(self.target_class_index)

    def resizeEvent(self, ev):
        width = (self.view.width() - self.view.verticalScrollBar().width())
        self.grid.reflow(width)
        self.grid.setPreferredWidth(width)

        super().resizeEvent(ev)


class GridItem(SelectableGridItem, ZoomableGridItem):
    """The grid item we will use in our grid."""
    pass


class SklRandomForestAdapter:
    """Take a `RandomForest` and wrap all the trees into the `SklTreeAdapter`
    instances that Pythagorean trees use."""
    def __init__(self, model):
        self._adapters = None
        self._domain = model.domain
        self._trees = model.trees

    def get_trees(self):
        """Get the tree adapters in the random forest."""
        if not self._adapters:
            self._adapters = list(map(SklTreeAdapter, self._trees))
        return self._adapters

    @property
    def domain(self):
        """Get the domain."""
        return self._domain


if __name__ == '__main__':
    from Orange.modelling import RandomForestLearner
    from AnyQt.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    ow = OWPythagoreanForest()
    data = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    rf = RandomForestLearner(n_estimators=10)(data)
    rf.instances = data
    ow.set_rf(rf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()
