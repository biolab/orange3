
import numpy
from functools import reduce
from operator import iadd

from PyQt4.QtGui import (
    QSlider, QLabel, QFormLayout, QGraphicsRectItem, QGraphicsGridLayout,
    QFontMetrics, QPen, QIcon, QPixmap, QLinearGradient, QPainter, QColor,
    QBrush
)

from PyQt4.QtCore import (
    Qt, QEvent, QRect, QRectF, QSize, QSizeF, QPoint, QPointF
)
from PyQt4.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data
import Orange.misc
from Orange.clustering import hierarchical

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorbrewer
from .owhierarchicalclustering import DendrogramWidget, GraphicsSimpleTextList


class DistanceMapItem(pg.ImageItem):
    selectionChanged = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.__selections = []
        #: (QGraphicsRectItem, QRectF) | None
        self.__dragging = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            i, j = self._cellAt(event.pos())
            if i != -1 and j != -1:
                if not event.modifiers() & Qt.ControlModifier:
                    self.__clearSelections()

                area = QRectF(event.pos(), QSizeF(0, 0))
                selrange = self._selectionForArea(area)
                rect = self._visualRectForSelection(selrange)
                item = QGraphicsRectItem(rect, self)
                pen = QPen(Qt.red, 0)
                pen.setCosmetic(True)
                item.setPen(pen)
                self.__dragging = item, area

        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.__dragging:
            i, j = self._cellAt(event.pos())
            item, area = self.__dragging
            area = QRectF(area.topLeft(), event.pos())
            selrange = self._selectionForArea(area)
            rect = self._visualRectForSelection(selrange)
            item.setRect(rect.normalized())
            self.__dragging = (item, area)

        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.__dragging:
            i, j = self._cellAt(event.pos())
            item, area = self.__dragging
            area = QRectF(area.topLeft(), event.pos())
            selrange = self._selectionForArea(area)
            rect = self._visualRectForSelection(selrange)
            item.setRect(rect)

            self.__selections.append((item, area))
            self.__dragging = None

            self.selectionChanged.emit()

        super().mouseReleaseEvent(event)
        event.accept()

    def _cellAt(self, pos):
        """Return the i, j cell index at `pos` in local coordinates."""
        if self.image is None:
            return -1, -1
        else:
            h, w = self.image.shape
            i, j = numpy.floor([h - pos.y(), pos.x()])
            if 0 < i >= h or 0 < j >= w:
                return -1, -1
            else:
                return int(i), int(j)

    def __clearSelections(self):
        for item, _ in self.__selections:
            item.setParentItem(None)
            if item.scene():
                item.scene().removeItem(item)

        self.__selections = []

    def _visualRectForSelection(self, rect):
        h, _ = self.image.shape
        r1, r2 = rect.top(), rect.bottom()
        c1, c2 = rect.left(), rect.right()

        return QRectF(QPointF(c1, h - r1), QPointF(c2, h - r2))

    def _selectionForArea(self, area):
        r1, c1 = self._cellAt(area.topLeft())
        r2, c2 = self._cellAt(area.bottomRight())
        return QRect(QPoint(c1, r1), QPoint(c2, r2)).normalized()

    def selections(self):
        selections = [self._selectionForArea(area)
                      for _, area in self.__selections]
        return [(range(r.top(), r.bottom()), range(r.left(), r.right()))
                for r in selections]

    def hoverMoveEvent(self, event):
        super().hoverMoveEvent(event)
        i, j = self._cellAt(event.pos())
        if i != -1 and j != -1:
            d = self.image[i, self.image.shape[1] - j - 1]
            self.setToolTip("{}, {}: {:.3f}".format(i, j, d))
        else:
            self.setToolTip("")


class OWDistanceMap(widget.OWWidget):
    name = "Distance Map"

    description = "Visualize a distance matrix"
    icon = "icons/DistanceMatrix.svg"
    priority = 1200

    inputs = [("Distances", Orange.misc.DistMatrix, "set_distances")]
    outputs = [("Data", Orange.data.Table), ("Features", widget.AttributeList)]

    display_grid = settings.Setting(False)
    sorting = settings.Setting(0)

    colormap = settings.Setting(0)
    color_gamma = settings.Setting(0.0)
    color_low = settings.Setting(0.0)
    color_high = settings.Setting(1.0)

    annotation_idx = settings.Setting(0)

    autocommit = settings.Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.matrix = None
        self._tree = None
        self._ordered_tree = None
        self._sorted_matrix = None
        self._sort_indices = None
        self._selection = None

        self._output_invalidated = False

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
                                        callback=self._invalidate_annotations)
        self.annot_combo.setModel(itemmodels.VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        self.controlArea.layout().addStretch()

        box = gui.widgetBox(self.controlArea, "Output")
        cb = gui.checkBox(box, self, "autocommit", "Commit on any change")
        b = gui.button(box, self, "Commit", callback=self.commit)
        gui.setStopper(self, b, cb, "_output_invalidated", callback=self.commit)

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
            self.grid_widget, orientation=DendrogramWidget.Left)
        self.left_dendrogram.setAcceptedMouseButtons(Qt.NoButton)
        self.left_dendrogram.setAcceptHoverEvents(False)

        self.top_dendrogram = DendrogramWidget(
            self.grid_widget, orientation=DendrogramWidget.Top)
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

    def set_distances(self, matrix):
        self.clear()
        self.matrix = matrix
        if matrix is not None:
            self.set_items(matrix.row_items)
        else:
            self.set_items(None)

    def set_items(self, items):
        self.items = items
        model = self.annot_combo.model()
        if items is None:
            model[:] = ["None", "Enumeration"]
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
        self.commit()

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
        self.matrix_item = DistanceMapItem(self._sorted_matrix[:, ::-1])

        self.viewbox.addItem(self.matrix_item)
        self.viewbox.setRange(QRectF(0, 0, *self._sorted_matrix.shape),
                              padding=0)
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
            self._sorted_matrix = self.matrix.X
            self._sort_indices = None
        else:
            if self.sorting == 1:
                tree = self._cluster_tree()
            elif self.sorting == 2:
                tree = self._ordered_cluster_tree()

            leaves = hierarchical.leaves(tree)
            indices = numpy.array([leaf.value.index for leaf in leaves])
            X = self.matrix.X
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
            labels = [str(i + 1) for i in range(self.matrix.dim[0])]
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

    def _update_color(self):
        if self.matrix_item:
            name, colors = self.palettes[self.colormap]
            n, colors = max(colors.items())
            colors = numpy.array(colors, dtype=numpy.ubyte)
            low, high = self.color_low * 255, self.color_high * 255
            points = numpy.linspace(low, high, n, dtype=numpy.float)
            space = numpy.linspace(0, 255, 255, dtype=numpy.float)

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

        if self.autocommit:
            self.commit()
        else:
            self._output_invalidated = True

    def commit(self):
        datasubset = None
        featuresubset = None

        if not self._selection:
            pass
        elif isinstance(self.items, Orange.data.Table):
            indices = self._selection
            datasubset = self.items.from_table_rows(self.items, indices)
        elif isinstance(self.items, widget.AttributeList):
            subset = [self.items[i] for i in self._selection]
            featuresubset = widget.AttributeList(subset)

        self.send("Data", datasubset)
        self.send("Features", featuresubset)
        self._output_invalidated = False


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

        lineheight = max(1, h / n)
        fontsize = self._point_size(lineheight)

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


class ImageWidget(pg.GraphicsWidget):
    def __init__(self, image=None, parent=None):
        super().__init__(parent)
        self._image = image
        self._image.setParentItem(self)

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            w, h = self._image.image.shape
            # Take into account the constraint, keep aspect ratio, ...
            return QSizeF(w, h)
        else:
            return super().sizeHint(which, constraint)

    def setGeometry(self, geom):
        super().setGeometry(geom)
        geom = self.geometry()
        self._image.setRect(self.contentsRect())

    def changeEvent(self, event):
        if event.type() == QEvent.ContentsRectChange:
            self._image.setRect(self.contentsRect())
        super().changeEvent(event)


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


def test():
    from PyQt4.QtGui import QApplication
    import sip
    import Orange.distance
    app = QApplication([])
    w = OWDistanceMap()
    w.show()
    w.raise_()
#     data = Orange.data.Table("iris")
    data = Orange.data.Table("housing")
    dist = Orange.distance.Euclidean(data)
    w.set_distances(dist)
    w.handleNewSignals()
    rval = app.exec_()
    w.onDeleteWidget()
    sip.delete(w)
    del w
    return rval

if __name__ == "__main__":
    import sys
    sys.exit(test())
