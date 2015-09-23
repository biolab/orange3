import sys
import itertools
from functools import reduce
from operator import iadd

import numpy

from PyQt4.QtGui import (
    QFormLayout, QGraphicsRectItem, QGraphicsGridLayout,
    QFontMetrics, QPen, QIcon, QPixmap, QLinearGradient, QPainter, QColor,
    QBrush, QTransform, QGraphicsWidget, QApplication
)

from PyQt4.QtCore import Qt, QRect, QRectF, QSize, QPointF
from PyQt4.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data
import Orange.misc
from Orange.clustering import hierarchical

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorbrewer
from .owhierarchicalclustering import DendrogramWidget, GraphicsSimpleTextList
from Orange.widgets.io import FileFormats


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
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)

        self.__selections = []
        #: (QGraphicsRectItem, QRectF) | None
        self.__dragging = None

    def __select(self, area, command):
        if command & self.Clear:
            self.__clearSelections()

        if command & self.Select:
            area = area.normalized()
            intersects = [rect.intersects(area)
                          for item, rect in self.__selections]

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

    def hoverMoveEvent(self, event):
        super().hoverMoveEvent(event)
        i, j = self._cellAt(event.pos())
        if i != -1 and j != -1:
            d = self.image[i, j]
            self.setToolTip("{}, {}: {:.3f}".format(i, j, d))
        else:
            self.setToolTip("")


class OWDistanceMap(widget.OWWidget):
    name = "Distance Map"
    description = "Visualize a distance matrix."
    icon = "icons/DistanceMatrix.svg"
    priority = 1200

    inputs = [("Distances", Orange.misc.DistMatrix, "set_distances")]
    outputs = [("Data", Orange.data.Table), ("Features", widget.AttributeList)]

    sorting = settings.Setting(0)

    colormap = settings.Setting(0)
    color_gamma = settings.Setting(0.0)
    color_low = settings.Setting(0.0)
    color_high = settings.Setting(1.0)

    annotation_idx = settings.Setting(0)

    autocommit = settings.Setting(True)

    want_graph = True

    def __init__(self, parent=None):
        super().__init__(parent)

        self.matrix = None
        self._tree = None
        self._ordered_tree = None
        self._sorted_matrix = None
        self._sort_indices = None
        self._selection = None

        box = gui.widgetBox(self.controlArea, "Element sorting", margin=0)
        gui.comboBox(box, self, "sorting",
                     items=["None", "Clustering",
                            "Clustering with ordered leaves"
                            ],
                     callback=self._invalidate_ordering)

        box = gui.widgetBox(self.controlArea, "Colors")

        self.colormap_cb = gui.comboBox(
            box, self, "colormap", callback=self._update_color
        )
        self.colormap_cb.setIconSize(QSize(64, 16))
        self.palettes = list(sorted(load_default_palettes()))
        init_color_combo(self.colormap_cb, self.palettes, QSize(64, 16))
        self.colormap_cb.setCurrentIndex(self.colormap)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
#         form.addRow(
#             "Gamma",
#             gui.hSlider(box, self, "color_gamma", minValue=0.0, maxValue=1.0,
#                         step=0.05, ticks=True, intOnly=False,
#                         createLabel=False, callback=self._update_color)
#         )
        form.addRow(
            "Low",
            gui.hSlider(box, self, "color_low", minValue=0.0, maxValue=1.0,
                        step=0.05, ticks=True, intOnly=False,
                        createLabel=False, callback=self._update_color)
        )
        form.addRow(
            "High",
            gui.hSlider(box, self, "color_high", minValue=0.0, maxValue=1.0,
                        step=0.05, ticks=True, intOnly=False,
                        createLabel=False, callback=self._update_color)
        )
        box.layout().addLayout(form)

        box = gui.widgetBox(self.controlArea, "Annotations")
        self.annot_combo = gui.comboBox(box, self, "annotation_idx",
                                        callback=self._invalidate_annotations,
                                        contentsLength=12)
        self.annot_combo.setModel(itemmodels.VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        self.controlArea.layout().addStretch()

        gui.auto_commit(self.controlArea, self, "autocommit",
                        "Send data", "Auto send is on")

        self.view = pg.GraphicsView(background="w")
        self.mainArea.layout().addWidget(self.view)

        self.grid_widget = pg.GraphicsWidget()
        self.grid = QGraphicsGridLayout()
        self.grid_widget.setLayout(self.grid)

        self.viewbox = pg.ViewBox(enableMouse=False)
        self.viewbox.setAcceptedMouseButtons(Qt.NoButton)
        self.viewbox.setAcceptHoverEvents(False)
        self.grid.addItem(self.viewbox, 1, 1)

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

        self.grid.addItem(self.left_dendrogram, 1, 0)
        self.grid.addItem(self.top_dendrogram, 0, 1)

        self.right_labels = TextList(
            alignment=Qt.AlignLeft)

        self.bottom_labels = TextList(
            orientation=Qt.Horizontal, alignment=Qt.AlignRight)

        self.grid.addItem(self.right_labels, 1, 2)
        self.grid.addItem(self.bottom_labels, 2, 1)

        self.view.setCentralItem(self.grid_widget)

        self.left_dendrogram.hide()
        self.top_dendrogram.hide()
        self.right_labels.hide()
        self.bottom_labels.hide()

        self.matrix_item = None
        self.dendrogram = None

        self.grid_widget.scene().installEventFilter(self)
        self.graphButton.clicked.connect(self.save_graph)

    def set_distances(self, matrix):
        self.clear()
        self.error(0)
        if matrix is not None:
            N, _ = matrix.shape
            if N < 2:
                self.error(0, "Empty distance matrix.")
                matrix = None

        self.matrix = matrix
        if matrix is not None:
            self.set_items(matrix.row_items, matrix.axis)
        else:
            self.set_items(None)

    def set_items(self, items, axis=1):
        self.items = items
        model = self.annot_combo.model()
        if items is None:
            model[:] = ["None", "Enumeration"]
        elif not axis:
            model[:] = ["None", "Enumeration", "Attribute names"]
            self.annotation_idx = 2
        elif isinstance(items, Orange.data.Table):
            model[:] = ["None", "Enumeration"] + list(items.domain)
        elif isinstance(items, list) and \
                all(isinstance(item, Orange.data.Variable) for item in items):
            model[:] = ["None", "Enumeration", "Name"]
        else:
            model[:] = ["None", "Enumeration"]
        self.annotation_idx = min(self.annotation_idx, len(model) - 1)

    def clear(self):
        self.matrix = None
        self.cluster = None
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
        self.unconditional_commit()

    def _clear_plot(self):
        def remove(item):
            item.setParentItem(None)
            item.scene().removeItem(item)

        if self.matrix_item:
            remove(self.matrix_item)
            self.matrix_item = None

        self.top_dendrogram.hide()
        self.left_dendrogram.hide()

        self._set_labels(None)

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
        self.matrix_item = DistanceMapItem(self._sorted_matrix)
        # Scale the y axis to compensate for pg.ViewBox's y axis invert
        self.matrix_item.scale(1, -1)
        self.viewbox.addItem(self.matrix_item)
        # Set fixed view box range.
        h, w = self._sorted_matrix.shape
        self.viewbox.setRange(QRectF(0, -h, w, h), padding=0)

        self.matrix_item.selectionChanged.connect(self._invalidate_selection)

        if self.sorting == 0:
            tree = None
        elif self.sorting == 1:
            tree = self._cluster_tree()
        else:
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

    def _update_ordering(self):
        if self.sorting == 0:
            self._sorted_matrix = self.matrix
            self._sort_indices = None
        else:
            if self.sorting == 1:
                tree = self._cluster_tree()
            elif self.sorting == 2:
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
        if self.annotation_idx == 0:
            labels = None
        elif self.annotation_idx == 1:
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
            labels = [var.repr_val(value) for value in column]

        self._set_labels(labels)

    def _set_labels(self, labels):
        self._labels = labels

        if labels and self.sorting:
            sortind = self._sort_indices
            labels = [labels[i] for i in sortind]

        for textlist in [self.right_labels, self.bottom_labels]:
            textlist.set_labels(labels or [])
            textlist.setVisible(bool(labels))

        constraint = -1 if labels else 0
        self.right_labels.setMaximumWidth(constraint)
        self.bottom_labels.setMaximumHeight(constraint)

    def _update_color(self):
        if self.matrix_item:
            name, colors = self.palettes[self.colormap]
            n, colors = max(colors.items())
            colors = numpy.array(colors, dtype=numpy.ubyte)
            low, high = self.color_low * 255, self.color_high * 255
            points = numpy.linspace(low, high, n)
            space = numpy.linspace(0, 255, 255)

            r = numpy.interp(space, points, colors[:, 0], left=255, right=0)
            g = numpy.interp(space, points, colors[:, 1], left=255, right=0)
            b = numpy.interp(space, points, colors[:, 2], left=255, right=0)
            colortable = numpy.c_[r, g, b]
            self.matrix_item.setLookupTable(colortable)

    def _invalidate_selection(self):
        ranges = self.matrix_item.selections()
        ranges = reduce(iadd, ranges, [])
        indices = reduce(iadd, ranges, [])
        if self.sorting:
            sortind = self._sort_indices
            indices = [sortind[i] for i in indices]
        self._selection = list(sorted(set(indices)))
        self.commit()

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
                datasubset = Orange.data.Table.from_table(domain, self.items)
        elif isinstance(self.items, widget.AttributeList):
            subset = [self.items[i] for i in self._selection]
            featuresubset = widget.AttributeList(subset)

        self.send("Data", datasubset)
        self.send("Features", featuresubset)

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.grid_widget,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


class TextList(GraphicsSimpleTextList):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._updateFontSize()

    def _updateFontSize(self):
        crect = self.contentsRect()
        if self.orientation == Qt.Vertical:
            h = crect.height()
        else:
            h = crect.width()
        n = len(getattr(self, "label_items", []))
        if n == 0:
            return

        if self.scene() is not None:
            maxfontsize = self.scene().font().pointSize()
        else:
            maxfontsize = QApplication.instance().font().pointSize()

        lineheight = max(1, h / n)
        fontsize = min(self._point_size(lineheight), maxfontsize)

        font = self.font()
        font.setPointSize(fontsize)

        self.setFont(font)
        self.layout().invalidate()
        self.layout().activate()

    def _point_size(self, height):
        font = self.font()
        font.setPointSize(height)
        fix = 0
        while QFontMetrics(font).lineSpacing() > height and height - fix > 1:
            fix += 1
            font.setPointSize(height - fix)
        return height - fix


##########################
# Color palette management
##########################


def palette_gradient(colors, discrete=False):
    n = len(colors)
    stops = numpy.linspace(0.0, 1.0, n, endpoint=True)
    gradstops = [(float(stop), color) for stop, color in zip(stops, colors)]
    grad = QLinearGradient(QPointF(0, 0), QPointF(1, 0))
    grad.setStops(gradstops)
    return grad


def palette_pixmap(colors, size):
    img = QPixmap(size)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    grad = palette_gradient(colors)
    grad.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawRect(0, 0, size.width(), size.height())
    painter.end()
    return img


def init_color_combo(cb, palettes, iconsize):
    cb.clear()
    iconsize = cb.iconSize()

    for name, palette in palettes:
        n, colors = max(palette.items())
        colors = [QColor(*c) for c in colors]
        cb.addItem(QIcon(palette_pixmap(colors, iconsize)), name,
                   palette)


def load_default_palettes():
    palettes = colorbrewer.colorSchemes["sequential"]
    return list(palettes.items())


def test(argv=sys.argv):
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    import sip
    import Orange.distance
    app = QApplication(argv)
    w = OWDistanceMap()
    w.show()
    w.raise_()
    data = Orange.data.Table(filename)
    dist = Orange.distance.Euclidean(data)
    w.set_distances(dist)
    w.handleNewSignals()
    rval = app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    sip.delete(w)
    del w
    return rval

if __name__ == "__main__":
    sys.exit(test())
