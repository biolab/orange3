"""Pythagorean forest widget for visualizing random forests."""
from math import log, sqrt
from typing import Any, Callable, Optional

from AnyQt.QtCore import Qt, QRectF, QSize, QPointF, QSizeF, QModelIndex, \
    QItemSelection, QItemSelectionModel, QT_VERSION
from AnyQt.QtGui import QPainter, QPen, QColor, QBrush, QMouseEvent
from AnyQt.QtWidgets import QSizePolicy, QGraphicsScene, QLabel, QSlider, \
    QListView, QStyledItemDelegate, QStyleOptionViewItem, QStyle

from Orange.base import RandomForestModel, TreeModel
from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTreeViewer,
    ContinuousTreeNode,
)
from Orange.widgets.visualize.utils.tree.skltreeadapter import \
    SklTreeAdapter
from Orange.widgets.widget import OWWidget


class PythagoreanForestModel(PyListModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_limit = -1
        self.target_class_idx = None
        self.size_calc_idx = 0
        self.size_adjustment = None
        self.item_scale = 2

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.QDisplayRole) -> Any
        if not index.isValid():
            return None

        idx = index.row()

        if role == Qt.SizeHintRole:
            return self.item_scale * QSize(100, 100)

        if role == Qt.DisplayRole:
            if 'tree' not in self._other_data[idx]:
                scene = QGraphicsScene(parent=self)
                tree = PythagorasTreeViewer(
                    adapter=self._list[idx],
                    weight_adjustment=OWPythagoreanForest.SIZE_CALCULATION[
                        self.size_calc_idx][1],
                    interactive=False,
                    padding=100,
                    depth_limit=self.depth_limit,
                    target_class_index=self.target_class_idx,
                )
                scene.addItem(tree)
                self._other_data[idx]['scene'] = scene
                self._other_data[idx]['tree'] = tree

            return self._other_data[idx]['scene']

        return super().data(index, role)

    @property
    def trees(self):
        """Get the tree adapters."""
        return self._list

    def update_tree_views(self, func):
        # type: (Callable[[PythagorasTreeViewer], None]) -> None
        """Apply `func` to every rendered tree viewer instance."""
        for idx, tree_data in enumerate(self._other_data):
            if 'tree' in tree_data:
                func(tree_data['tree'])
                index = self.index(idx)
                if QT_VERSION < 0x50000:
                    self.dataChanged.emit(index, index)
                else:
                    self.dataChanged.emit(index, index, [Qt.DisplayRole])

    def update_depth(self, depth):
        self.depth_limit = depth
        self.update_tree_views(lambda tree: tree.set_depth_limit(depth))

    def update_target_class(self, idx):
        self.target_class_idx = idx
        self.update_tree_views(lambda tree: tree.target_class_changed(idx))

    def update_item_size(self, scale):
        self.item_scale = scale / 100
        indices = [idx for idx, _ in enumerate(self._other_data)]
        self.emitDataChanged(indices)

    def update_size_calc(self, idx):
        self.size_calc_idx = idx
        _, size_calc = OWPythagoreanForest.SIZE_CALCULATION[idx]
        self.update_tree_views(lambda tree: tree.set_size_calc(size_calc))


class PythagorasTreeDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        scene = index.data(Qt.DisplayRole)  # type: Optional[QGraphicsScene]
        if scene is None:
            super().paint(painter, option, index)
            return

        painter.save()
        rect = QRectF(QPointF(option.rect.topLeft()), QSizeF(option.rect.size()))
        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(QColor(125, 162, 206, 192)))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
        else:
            painter.setPen(QPen(QColor('#ebebeb')))
        painter.drawRoundedRect(rect, 3, 3)
        painter.restore()

        painter.setRenderHint(QPainter.Antialiasing)

        # The sceneRect doesn't automatically shrink to fit contents, so when
        # drawing smaller tree, remove any excess space aroung the tree
        scene.setSceneRect(scene.itemsBoundingRect())

        # Make sure the tree is centered in the item cell
        # First, figure out how much we get the bounding rect to the size of
        # the available painting rect
        scene_rect = scene.itemsBoundingRect()
        w_scale = option.rect.width() / scene_rect.width()
        h_scale = option.rect.height() / scene_rect.height()
        # In order to keep the aspect ratio, we use the same scale
        scale = min(w_scale, h_scale)
        # Figure out the rescaled scene width/height
        scene_w = scale * scene_rect.width()
        scene_h = scale * scene_rect.height()
        # Figure out how much we have to offset the rect so that the scene will
        # be painted in the centre of the rect
        offset_w = (option.rect.width() - scene_w) / 2
        offset_h = (option.rect.height() - scene_h) / 2
        offset = option.rect.topLeft() + QPointF(offset_w, offset_h)
        # Finally, we have all the data for the new rect in which to render
        target_rect = QRectF(offset, QSizeF(scene_w, scene_h))

        scene.render(painter, target=target_rect, mode=Qt.KeepAspectRatio)


class ClickToClearSelectionListView(QListView):
    """Clicking outside any item clears the current selection."""
    def mousePressEvent(self, event):
        # type: (QMouseEvent) -> None
        super().mousePressEvent(event)

        index = self.indexAt(event.pos())
        if index.row() == -1:
            self.clearSelection()


class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean Forest'
    description = 'Pythagorean forest for visualising random forests.'
    icon = 'icons/PythagoreanForest.svg'
    settings_version = 2
    keywords = ["fractal"]

    priority = 1001

    class Inputs:
        random_forest = Input("Random Forest", RandomForestModel, replaces=["Random forest"])

    class Outputs:
        tree = Output("Tree", TreeModel)

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    settingsHandler = settings.ClassValuesContextHandler()

    depth_limit = settings.Setting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    zoom = settings.Setting(200)

    selected_index = settings.ContextSetting(None)

    SIZE_CALCULATION = [
        ('Normal', lambda x: x),
        ('Square root', lambda x: sqrt(x)),
        ('Logarithmic', lambda x: log(x + 1)),
    ]

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            settings.pop('selected_tree_index', None)
            v1_min, v1_max = 20, 150
            v2_min, v2_max = 100, 400
            ratio = (v2_max - v2_min) / (v1_max - v1_min)
            settings['zoom'] = int(ratio * (settings['zoom'] - v1_min) + v2_min)

    def __init__(self):
        super().__init__()
        self.rf_model = None
        self.forest = None
        self.instances = None

        self.color_palette = None

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Forest')
        self.ui_info = gui.widgetLabel(box_info)

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        # maxValue is set to a wide three-digit number to probably ensure the
        # proper label width. The maximum is later set to match the tree depth
        self.ui_depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            maxValue=900
        )  # type: QSlider
        self.ui_target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation=Qt.Horizontal, items=[], contentsLength=8,
            searchable=True
        )
        self.ui_size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation=Qt.Horizontal,
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
        )
        self.ui_zoom_slider = gui.hSlider(
            box_display, self, 'zoom', label='Zoom', ticks=False, minValue=100,
            maxValue=400, createLabel=False, intOnly=False,
        )  # type: QSlider

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # MAIN AREA
        self.forest_model = PythagoreanForestModel(parent=self)
        self.forest_model.update_item_size(self.zoom)
        self.ui_depth_slider.valueChanged.connect(
            self.forest_model.update_depth)
        self.ui_target_class_combo.currentIndexChanged.connect(
            self.forest_model.update_target_class)
        self.ui_zoom_slider.valueChanged.connect(
            self.forest_model.update_item_size)
        self.ui_size_calc_combo.currentIndexChanged.connect(
            self.forest_model.update_size_calc)

        self.list_delegate = PythagorasTreeDelegate(parent=self)
        self.list_view = ClickToClearSelectionListView(parent=self)
        self.list_view.setWrapping(True)
        self.list_view.setFlow(QListView.LeftToRight)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setModel(self.forest_model)
        self.list_view.setItemDelegate(self.list_delegate)
        self.list_view.setSpacing(2)
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.selectionModel().selectionChanged.connect(self.commit)
        self.list_view.setUniformItemSizes(True)
        self.mainArea.layout().addWidget(self.list_view)

        self.resize(800, 500)

        # Clear to set sensible default values
        self.clear()

    @Inputs.random_forest
    def set_rf(self, model=None):
        """When a different forest is given."""
        self.closeContext()
        self.clear()
        self.rf_model = model

        if model is not None:
            self.instances = model.instances
            self._update_target_class_combo()

            self.forest = self._get_forest_adapter(self.rf_model)
            self.forest_model[:] = self.forest.trees

            self._update_info_box()
            self._update_depth_slider()

            self.openContext(
                model.domain.class_var if model.domain is not None else None
            )
        # Restore item selection
        if self.selected_index is not None:
            index = self.list_view.model().index(self.selected_index)
            selection = QItemSelection(index, index)
            self.list_view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def clear(self):
        """Clear all relevant data from the widget."""
        self.rf_model = None
        self.forest = None
        self.forest_model.clear()
        self.selected_index = None

        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    def _update_info_box(self):
        self.ui_info.setText('Trees: {}'.format(len(self.forest.trees)))

    def _update_depth_slider(self):
        self.depth_limit = self._get_max_depth()

        self.ui_depth_slider.parent().setEnabled(True)
        self.ui_depth_slider.setMaximum(self.depth_limit)
        self.ui_depth_slider.setValue(self.depth_limit)

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
        # set it to 0, context will change if required
        self.target_class_index = 0

    def _clear_info_box(self):
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        self.ui_target_class_combo.clear()
        self.target_class_index = -1

    def _clear_depth_slider(self):
        self.ui_depth_slider.parent().setEnabled(False)
        self.ui_depth_slider.setMaximum(0)

    def _get_max_depth(self):
        return max(tree.max_depth for tree in self.forest.trees)

    def _get_forest_adapter(self, model):
        return SklRandomForestAdapter(model)

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self, selection: QItemSelection) -> None:
        """Commit the selected tree to output."""
        selected_indices = selection.indexes()

        if not len(selected_indices):
            self.selected_index = None
            self.Outputs.tree.send(None)
            return

        # We only allow selecting a single tree so there will always be one index
        self.selected_index = selected_indices[0].row()

        tree = self.rf_model.trees[self.selected_index]
        tree.instances = self.instances
        tree.meta_target_class_index = self.target_class_index
        tree.meta_size_calc_idx = self.size_calc_idx
        tree.meta_depth_limit = self.depth_limit

        self.Outputs.tree.send(tree)

    def send_report(self):
        """Send report."""
        self.report_plot()


class SklRandomForestAdapter:
    """Take a `RandomForest` and wrap all the trees into the `SklTreeAdapter`
    instances that Pythagorean trees use."""
    def __init__(self, model):
        self._adapters = None
        self._domain = model.domain
        self._trees = model.trees

    @property
    def trees(self):
        """Get the tree adapters in the random forest."""
        if not self._adapters:
            self._adapters = list(map(SklTreeAdapter, self._trees))
        return self._adapters

    @property
    def domain(self):
        """Get the domain."""
        return self._domain


if __name__ == '__main__':  # pragma: no cover
    from Orange.modelling import RandomForestLearner
    data = Table('iris')
    rf = RandomForestLearner(n_estimators=10)(data)
    rf.instances = data
    WidgetPreview(OWPythagoreanForest).run(rf)
