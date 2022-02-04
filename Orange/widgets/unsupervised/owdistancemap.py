import itertools
from functools import reduce
from operator import iadd

import numpy

from AnyQt.QtWidgets import (
    QGraphicsRectItem, QGraphicsGridLayout, QApplication, QSizePolicy
)
from AnyQt.QtGui import QFontMetrics, QPen, QTransform, QFont
from AnyQt.QtCore import Qt, QRect, QRectF, QPointF
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data
import Orange.misc
from Orange.clustering import hierarchical
from Orange.data.domain import filter_visible

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.graphicsscene import graphicsscene_help_event
from Orange.widgets.utils.graphicstextlist import TextListWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.dendrogram import DendrogramWidget
from Orange.widgets.visualize.utils.plotutils import GraphicsView
from Orange.widgets.visualize.utils.heatmap import (
    GradientColorMap, GradientLegendWidget,
)
from Orange.widgets.utils.colorgradientselection import ColorGradientSelection


def _remove_item(item):
    item.setParentItem(None)
    scene = item.scene()
    if scene is not None:
        scene.removeItem(item)


class DistanceMapItem(pg.ImageItem):
    """A distance matrix image with user selectable regions.
    """
    class SelectionRect(QGraphicsRectItem):
        def boundingRect(self):
            return super().boundingRect().adjusted(-1, -1, 1, 1)

        def paint(self, painter, option, widget=None):
            t = painter.transform()

            rect = t.mapRect(self.rect())

            painter.save()
            painter.setTransform(QTransform())
            pwidth = self.pen().widthF()
            painter.setPen(self.pen())
            painter.drawRect(rect.adjusted(pwidth, -pwidth, -pwidth, pwidth))
            painter.restore()

    selectionChanged = Signal()

    Clear, Select, Commit = 1, 2, 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setAcceptHoverEvents(True)

        self.__selections = []
        #: (QGraphicsRectItem, QRectF) | None
        self.__dragging = None

    def __select(self, area, command):
        if command & self.Clear:
            self.__clearSelections()

        if command & self.Select:
            area = area.normalized()

            def partition(predicate, iterable):
                t1, t2 = itertools.tee(iterable)
                return (itertools.filterfalse(predicate, t1),
                        filter(predicate, t2))

            def intersects(selection):
                _, selarea = selection
                return selarea.intersects(area)

            disjoint, intersection = partition(intersects, self.__selections)
            disjoint = list(disjoint)
            intersection = list(intersection)

            # merge intersecting selections into a single area
            area = reduce(QRect.united, (area for _, area in intersection),
                          area)

            visualarea = self.__visualRectForSelection(area)
            item = DistanceMapItem.SelectionRect(visualarea, self)
            item.setPen(QPen(Qt.red, 0))

            selection = disjoint + [(item, area)]

            for item, _ in intersection:
                _remove_item(item)

            self.__selections = selection

        self.selectionChanged.emit()

    def __elastic_band_select(self, area, command):
        if command & self.Clear and self.__dragging:
            item, area = self.__dragging
            _remove_item(item)
            self.__dragging = None

        if command & self.Select:
            if self.__dragging:
                item, _ = self.__dragging
            else:
                item = DistanceMapItem.SelectionRect(self)
                item.setPen(QPen(Qt.red, 0))

            # intersection with existing regions
            intersection = [(item, selarea)
                            for item, selarea in self.__selections
                            if area.intersects(selarea)]
            fullarea = reduce(
                QRect.united, (selarea for _, selarea in intersection),
                area
            )
            visualarea = self.__visualRectForSelection(fullarea)
            item.setRect(visualarea)

            self.__dragging = item, area

        if command & self.Commit and self.__dragging:
            item, area = self.__dragging
            self.__select(area, self.Select)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            r, c = self._cellAt(event.pos())
            if r != -1 and c != -1:
                # Clear existing selection
                # TODO: Fix extended selection.
                self.__select(QRect(), self.Clear)
                selrange = QRect(c, r, 1, 1)
                self.__elastic_band_select(selrange, self.Select | self.Clear)
        elif event.button() == Qt.RightButton:
            self.__select(QRect(), self.Clear)

        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.__dragging:
            r1, c1 = self._cellAt(event.buttonDownPos(Qt.LeftButton))
            r2, c2 = self._cellCloseTo(event.pos())
            selrange = QRect(c1, r1, 1, 1).united(QRect(c2, r2, 1, 1))
            self.__elastic_band_select(selrange, self.Select)

        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.__dragging:
            r1, c1 = self._cellAt(event.buttonDownPos(Qt.LeftButton))
            r2, c2 = self._cellCloseTo(event.pos())
            selrange = QRect(c1, r1, 1, 1).united(QRect(c2, r2, 1, 1))
            self.__elastic_band_select(selrange, self.Select | self.Commit)

            self.__elastic_band_select(QRect(), self.Clear)

        super().mouseReleaseEvent(event)
        event.accept()

    def _cellAt(self, pos):
        """Return the i, j cell index at `pos` in local coordinates."""
        if self.image is None:
            return -1, -1
        else:
            h, w = self.image.shape
            i, j = numpy.floor([pos.y(), pos.x()])
            if 0 <= i < h and 0 <= j < w:
                return int(i), int(j)
            else:
                return -1, -1

    def _cellCloseTo(self, pos):
        """Return the i, j cell index closest to `pos` in local coordinates."""
        if self.image is None:
            return -1, -1
        else:
            h, w = self.image.shape
            i, j = numpy.floor([pos.y(), pos.x()])
            i = numpy.clip(i, 0, h - 1)
            j = numpy.clip(j, 0, w - 1)
            return int(i), int(j)

    def __clearSelections(self):
        for item, _ in self.__selections:
            _remove_item(item)

        self.__selections = []

    def __visualRectForSelection(self, rect):
        h, w = self.image.shape
        rect = rect.normalized()
        rect = rect.intersected(QRect(0, 0, w, h))
        r1, r2 = rect.top(), rect.bottom() + 1
        c1, c2 = rect.left(), rect.right() + 1
        return QRectF(QPointF(c1, r1), QPointF(c2, r2))

    def __selectionForArea(self, area):
        r1, c1 = self._cellAt(area.topLeft())
        r2, c2 = self._cellAt(area.bottomRight())
        selarea = QRect(c1, r1, c2 - c1 + 1, r2 - r1 + 1)
        return selarea.normalized()

    def selections(self):
        selections = [self.__selectionForArea(area)
                      for _, area in self.__selections]
        return [(range(r.top(), r.bottom() + 1),
                 range(r.left(), r.right() + 1))
                for r in selections]

    def set_selections(self, ranges):
        self.__clearSelections()
        for y, x in ranges:
            area = QRectF(x.start, y.start,
                          x.stop - x.start - 1, y.stop - y.start - 1)
            item = DistanceMapItem.SelectionRect(area, self)
            item.setPen(QPen(Qt.red, 0))
            self.__selections.append((item, area))
        self.selectionChanged.emit()

    def hoverMoveEvent(self, event):
        super().hoverMoveEvent(event)
        i, j = self._cellAt(event.pos())
        if i != -1 and j != -1:
            d = self.image[i, j]
            self.setToolTip("{}, {}: {:.3f}".format(i, j, d))
        else:
            self.setToolTip("")


class GraphicsView(pg.GraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scene = self.scene()
        delegate = HelpEventDelegate(self.__helpEvent, parent=self)
        scene.installEventFilter(delegate)

    def __helpEvent(self, event):
        graphicsscene_help_event(self.scene(), event)
        return event.isAccepted()


class OWDistanceMap(widget.OWWidget):
    name = "Distance Map"
    description = "Visualize a distance matrix."
    icon = "icons/DistanceMap.svg"
    priority = 1200
    keywords = []

    class Inputs:
        distances = Input("Distances", Orange.misc.DistMatrix)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)
        features = Output("Features", widget.AttributeList, dynamic=False)

    settingsHandler = settings.PerfectDomainContextHandler()

    #: type of ordering to apply to matrix rows/columns
    NoOrdering, Clustering, OrderedClustering = 0, 1, 2

    sorting = settings.Setting(NoOrdering)

    palette_name = settings.Setting(colorpalettes.DefaultContinuousPaletteName)
    color_gamma = settings.Setting(0.0)
    color_low = settings.Setting(0.0)
    color_high = settings.Setting(1.0)

    annotation_idx = settings.ContextSetting(0)
    pending_selection = settings.Setting(None, schema_only=True)

    autocommit = settings.Setting(True)

    graph_name = "grid_widget"

    # Disable clustering for inputs bigger than this
    _MaxClustering = 25000
    # Disable cluster leaf ordering for inputs bigger than this
    _MaxOrderedClustering = 2000

    def __init__(self):
        super().__init__()

        self.matrix = None
        self._matrix_range = 0.
        self._tree = None
        self._ordered_tree = None
        self._sorted_matrix = None
        self._sort_indices = None
        self._selection = None

        self.sorting_cb = gui.comboBox(
            self.controlArea, self, "sorting", box="Element Sorting",
            items=["None", "Clustering", "Clustering with ordered leaves"],
            callback=self._invalidate_ordering)

        box = gui.vBox(self.controlArea, "Colors")
        self.color_map_widget = cmw = ColorGradientSelection(
            thresholds=(self.color_low, self.color_high),
        )
        model = itemmodels.ContinuousPalettesModel(parent=self)
        cmw.setModel(model)
        idx = cmw.findData(self.palette_name, model.KeyRole)
        if idx != -1:
            cmw.setCurrentIndex(idx)

        cmw.activated.connect(self._update_color)

        def _set_thresholds(low, high):
            self.color_low, self.color_high = low, high
            self._update_color()

        cmw.thresholdsChanged.connect(_set_thresholds)
        box.layout().addWidget(self.color_map_widget)

        self.annot_combo = gui.comboBox(
            self.controlArea, self, "annotation_idx", box="Annotations",
            contentsLength=12, searchable=True,
            callback=self._invalidate_annotations
        )
        self.annot_combo.setModel(itemmodels.VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        gui.rubber(self.controlArea)

        gui.auto_send(self.buttonsArea, self, "autocommit")

        self.view = GraphicsView(background=None)
        self.mainArea.layout().addWidget(self.view)

        self.grid_widget = pg.GraphicsWidget()
        self.grid = QGraphicsGridLayout()
        self.grid_widget.setLayout(self.grid)

        self.gradient_legend = GradientLegendWidget(0, 1, self._color_map())
        self.gradient_legend.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.gradient_legend.setMaximumWidth(250)
        self.grid.addItem(self.gradient_legend, 0, 1)
        self.viewbox = pg.ViewBox(enableMouse=False, enableMenu=False)
        self.viewbox.setAcceptedMouseButtons(Qt.NoButton)
        self.viewbox.setAcceptHoverEvents(False)
        self.grid.addItem(self.viewbox, 2, 1)

        self.left_dendrogram = DendrogramWidget(
            self.grid_widget, orientation=DendrogramWidget.Left,
            selectionMode=DendrogramWidget.NoSelection,
            hoverHighlightEnabled=False
        )
        self.left_dendrogram.setAcceptedMouseButtons(Qt.NoButton)
        self.left_dendrogram.setAcceptHoverEvents(False)

        self.top_dendrogram = DendrogramWidget(
            self.grid_widget, orientation=DendrogramWidget.Top,
            selectionMode=DendrogramWidget.NoSelection,
            hoverHighlightEnabled=False
        )
        self.top_dendrogram.setAcceptedMouseButtons(Qt.NoButton)
        self.top_dendrogram.setAcceptHoverEvents(False)

        self.grid.addItem(self.left_dendrogram, 2, 0)
        self.grid.addItem(self.top_dendrogram, 1, 1)

        self.right_labels = TextList(
            alignment=Qt.AlignLeft | Qt.AlignVCenter,
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        )
        self.bottom_labels = TextList(
            orientation=Qt.Horizontal,
            alignment=Qt.AlignRight | Qt.AlignVCenter,
            sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        )

        self.grid.addItem(self.right_labels, 2, 2)
        self.grid.addItem(self.bottom_labels, 3, 1)

        self.view.setCentralItem(self.grid_widget)

        self.gradient_legend.hide()
        self.left_dendrogram.hide()
        self.top_dendrogram.hide()
        self.right_labels.hide()
        self.bottom_labels.hide()

        self.matrix_item = None
        self.dendrogram = None

        self.settingsAboutToBePacked.connect(self.pack_settings)

    def pack_settings(self):
        if self.matrix_item is not None:
            self.pending_selection = self.matrix_item.selections()
        else:
            self.pending_selection = None

    @Inputs.distances
    def set_distances(self, matrix):
        self.closeContext()
        self.clear()
        self.error()
        if matrix is not None:
            N, _ = matrix.shape
            if N < 2:
                self.error("Empty distance matrix.")
                matrix = None

        self.matrix = matrix
        if matrix is not None:
            self._matrix_range = numpy.nanmax(matrix)
            self.set_items(matrix.row_items, matrix.axis)
        else:
            self._matrix_range = 0.
            self.set_items(None)

        if matrix is not None:
            N, _ = matrix.shape
        else:
            N = 0

        model = self.sorting_cb.model()
        item = model.item(2)

        msg = None
        if N > OWDistanceMap._MaxOrderedClustering:
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            if self.sorting == OWDistanceMap.OrderedClustering:
                self.sorting = OWDistanceMap.Clustering
                msg = "Cluster ordering was disabled due to the input " \
                      "matrix being to big"
        else:
            item.setFlags(item.flags() | Qt.ItemIsEnabled)

        item = model.item(1)
        if N > OWDistanceMap._MaxClustering:
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            if self.sorting == OWDistanceMap.Clustering:
                self.sorting = OWDistanceMap.NoOrdering
            msg = "Clustering was disabled due to the input " \
                  "matrix being to big"
        else:
            item.setFlags(item.flags() | Qt.ItemIsEnabled)

        self.information(msg)

    def set_items(self, items, axis=1):
        self.items = items
        model = self.annot_combo.model()
        if items is None:
            model[:] = ["None", "Enumeration"]
        elif not axis:
            model[:] = ["None", "Enumeration", "Attribute names"]
        elif isinstance(items, Orange.data.Table):
            annot_vars = list(filter_visible(items.domain.variables)) + list(items.domain.metas)
            model[:] = ["None", "Enumeration"] + annot_vars
            self.annotation_idx = 0
            self.openContext(items.domain)
        elif isinstance(items, list) and \
                all(isinstance(item, Orange.data.Variable) for item in items):
            model[:] = ["None", "Enumeration", "Name"]
        else:
            model[:] = ["None", "Enumeration"]
        self.annotation_idx = min(self.annotation_idx, len(model) - 1)

    def clear(self):
        self.matrix = None
        self._tree = None
        self._ordered_tree = None
        self._sorted_matrix = None
        self._selection = []
        self._clear_plot()

    def handleNewSignals(self):
        if self.matrix is not None:
            self._update_ordering()
            self._setup_scene()
            self._update_labels()
            if self.pending_selection is not None:
                self.matrix_item.set_selections(self.pending_selection)
                self.pending_selection = None
        self.commit.now()

    def _clear_plot(self):
        def remove(item):
            item.setParentItem(None)
            item.scene().removeItem(item)

        if self.matrix_item is not None:
            self.matrix_item.selectionChanged.disconnect(
                self._invalidate_selection)
            remove(self.matrix_item)
            self.matrix_item = None

        self._set_displayed_dendrogram(None)
        self._set_labels(None)
        self.gradient_legend.hide()

    def _cluster_tree(self):
        if self._tree is None:
            self._tree = hierarchical.dist_matrix_clustering(self.matrix)
        return self._tree

    def _ordered_cluster_tree(self):
        if self._ordered_tree is None:
            tree = self._cluster_tree()
            self._ordered_tree = \
                hierarchical.optimal_leaf_ordering(tree, self.matrix)
        return self._ordered_tree

    def _setup_scene(self):
        self._clear_plot()
        self.matrix_item = DistanceMapItem(self._sorted_matrix)
        # Scale the y axis to compensate for pg.ViewBox's y axis invert
        self.matrix_item.setTransform(QTransform.fromScale(1, -1), )
        self.viewbox.addItem(self.matrix_item)
        # Set fixed view box range.
        h, w = self._sorted_matrix.shape
        self.viewbox.setRange(QRectF(0, -h, w, h), padding=0)

        self.matrix_item.selectionChanged.connect(self._invalidate_selection)

        if self.sorting == OWDistanceMap.NoOrdering:
            tree = None
        elif self.sorting == OWDistanceMap.Clustering:
            tree = self._cluster_tree()
        elif self.sorting == OWDistanceMap.OrderedClustering:
            tree = self._ordered_cluster_tree()

        self._set_displayed_dendrogram(tree)

        self._update_color()

    def _set_displayed_dendrogram(self, root):
        self.left_dendrogram.set_root(root)
        self.top_dendrogram.set_root(root)
        self.left_dendrogram.setVisible(root is not None)
        self.top_dendrogram.setVisible(root is not None)

        constraint = 0 if root is None else -1  # 150
        self.left_dendrogram.setMaximumWidth(constraint)
        self.top_dendrogram.setMaximumHeight(constraint)

    def _invalidate_ordering(self):
        self._sorted_matrix = None
        if self.matrix is not None:
            self._update_ordering()
            self._setup_scene()
            self._update_labels()
            self._invalidate_selection()

    def _update_ordering(self):
        if self.sorting == OWDistanceMap.NoOrdering:
            self._sorted_matrix = self.matrix
            self._sort_indices = None
        else:
            if self.sorting == OWDistanceMap.Clustering:
                tree = self._cluster_tree()
            elif self.sorting == OWDistanceMap.OrderedClustering:
                tree = self._ordered_cluster_tree()

            leaves = hierarchical.leaves(tree)
            indices = numpy.array([leaf.value.index for leaf in leaves])
            X = self.matrix
            self._sorted_matrix = X[indices[:, numpy.newaxis],
                                    indices[numpy.newaxis, :]]
            self._sort_indices = indices

    def _invalidate_annotations(self):
        if self.matrix is not None:
            self._update_labels()

    def _update_labels(self, ):
        if self.annotation_idx == 0:  # None
            labels = None
        elif self.annotation_idx == 1:  # Enumeration
            labels = [str(i + 1) for i in range(self.matrix.shape[0])]
        elif self.annot_combo.model()[self.annotation_idx] == "Attribute names":
            attr = self.matrix.row_items.domain.attributes
            labels = [str(attr[i]) for i in range(self.matrix.shape[0])]
        elif self.annotation_idx == 2 and \
                isinstance(self.items, widget.AttributeList):
            labels = [v.name for v in self.items]
        elif isinstance(self.items, Orange.data.Table):
            var = self.annot_combo.model()[self.annotation_idx]
            column, _ = self.items.get_column_view(var)
            labels = [var.str_val(value) for value in column]

        self._set_labels(labels)

    def _set_labels(self, labels):
        self._labels = labels

        if labels and self.sorting != OWDistanceMap.NoOrdering:
            sortind = self._sort_indices
            labels = [labels[i] for i in sortind]

        for textlist in [self.right_labels, self.bottom_labels]:
            textlist.setItems(labels or [])
            textlist.setVisible(bool(labels))

        constraint = -1 if labels else 0
        self.right_labels.setMaximumWidth(constraint)
        self.bottom_labels.setMaximumHeight(constraint)

    def _color_map(self) -> GradientColorMap:
        palette = self.color_map_widget.currentData()
        return GradientColorMap(
            palette.lookup_table(),
            thresholds=(self.color_low, max(self.color_high, self.color_low)),
            span=(0., self._matrix_range))

    def _update_color(self):
        palette = self.color_map_widget.currentData()
        self.palette_name = palette.name
        if self.matrix_item:
            cmap = self._color_map().replace(span=(0., 1.))
            colors = cmap.apply(numpy.arange(256) / 255.)
            self.matrix_item.setLookupTable(colors)
            self.gradient_legend.show()
            self.gradient_legend.setRange(0, self._matrix_range)
            self.gradient_legend.setColorMap(self._color_map())

    def _invalidate_selection(self):
        ranges = self.matrix_item.selections()
        ranges = reduce(iadd, ranges, [])
        indices = reduce(iadd, ranges, [])
        if self.sorting != OWDistanceMap.NoOrdering:
            sortind = self._sort_indices
            indices = [sortind[i] for i in indices]
        self._selection = list(sorted(set(indices)))
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        datasubset = None
        featuresubset = None

        if not self._selection:
            pass
        elif isinstance(self.items, Orange.data.Table):
            indices = self._selection
            if self.matrix.axis == 1:
                datasubset = self.items.from_table_rows(self.items, indices)
            elif self.matrix.axis == 0:
                domain = Orange.data.Domain(
                    [self.items.domain[i] for i in indices],
                    self.items.domain.class_vars,
                    self.items.domain.metas)
                datasubset = self.items.transform(domain)
        elif isinstance(self.items, widget.AttributeList):
            subset = [self.items[i] for i in self._selection]
            featuresubset = widget.AttributeList(subset)

        self.Outputs.selected_data.send(datasubset)
        self.Outputs.annotated_data.send(create_annotated_table(self.items, self._selection))
        self.Outputs.features.send(featuresubset)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.clear()

    def send_report(self):
        annot = self.annot_combo.currentText()
        if self.annotation_idx <= 1:
            annot = annot.lower()
        self.report_items((
            ("Sorting", self.sorting_cb.currentText().lower()),
            ("Annotations", annot)
        ))
        if self.matrix is not None:
            self.report_plot()


class TextList(TextListWidget):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._updateFontSize()

    def _updateFontSize(self):
        crect = self.contentsRect()
        if self.orientation() == Qt.Vertical:
            h = crect.height()
        else:
            h = crect.width()
        n = self.count()
        if n == 0:
            return

        if self.scene() is not None:
            maxfontsize = self.scene().font().pointSize()
        else:
            maxfontsize = QApplication.instance().font().pointSize()

        lineheight = max(1., h / n)
        fontsize = min(self._point_size(lineheight), maxfontsize)

        font_ = QFont()
        font_.setPointSize(fontsize)
        self.setFont(font_)

    def _point_size(self, height):
        font = self.font()
        font.setPointSize(height)
        fix = 0
        while QFontMetrics(font).lineSpacing() > height and height - fix > 1:
            fix += 1
            font.setPointSize(height - fix)
        return height - fix


# run widget with `python -m Orange.widgets.unsupervised.owdistancemap`
if __name__ == "__main__":  # pragma: no cover
    import Orange.distance
    data = Orange.data.Table("iris")
    dist = Orange.distance.Euclidean(data)
    WidgetPreview(OWDistanceMap).run(dist)
