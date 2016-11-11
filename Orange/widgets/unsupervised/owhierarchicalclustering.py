# -*- coding: utf-8 -*-
import sys

from collections import namedtuple, OrderedDict
from itertools import chain
from contextlib import contextmanager

import numpy

from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsObject, QGraphicsLinearLayout, QGraphicsPathItem,
    QGraphicsScene, QGraphicsView, QGridLayout, QFormLayout, QSizePolicy,
    QGraphicsSimpleTextItem,
    QGraphicsLayoutItem, QAction,
)
from AnyQt.QtGui import (
    QTransform, QPainterPath, QPainterPathStroker, QColor, QBrush, QPen,
    QFont, QFontMetrics, QPolygonF, QKeySequence
)
from AnyQt.QtCore import Qt,  QSize, QSizeF, QPointF, QRectF, QLineF, QEvent
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data
from Orange.data.domain import filter_visible
from Orange.data import Domain
import Orange.misc
from Orange.clustering.hierarchical import \
    postorder, preorder, Tree, tree_from_linkage, dist_matrix_linkage, \
    leaves, prune, top_clusters

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette, itemmodels
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.io import FileFormat

__all__ = ["OWHierarchicalClustering"]


LINKAGE = ["Single", "Average", "Weighted", "Complete", "Ward"]


def dendrogram_layout(tree, expand_leaves=False):
    coords = []
    cluster_geometry = {}
    leaf_idx = 0
    for node in postorder(tree):
        cluster = node.value
        if node.is_leaf:
            if expand_leaves:
                start = float(cluster.first) + 0.5
                end = float(cluster.last - 1) + 0.5
            else:
                start = end = leaf_idx + 0.5
                leaf_idx += 1
            center = (start + end) / 2.0
            cluster_geometry[node] = (start, center, end)
            coords.append((node, (start, center, end)))
        else:
            left = node.left
            right = node.right
            left_center = cluster_geometry[left][1]
            right_center = cluster_geometry[right][1]
            start, end = left_center, right_center
            center = (start + end) / 2.0
            cluster_geometry[node] = (start, center, end)
            coords.append((node, (start, center, end)))

    return coords


Point = namedtuple("Point", ["x", "y"])
Element = namedtuple("Element", ["anchor", "path"])


def Path_toQtPath(geom):
    p = QPainterPath()
    anchor, points = geom
    if len(points) > 1:
        p.moveTo(*points[0])
        for (x, y) in points[1:]:
            p.lineTo(x, y)
    elif len(points) == 1:
        r = QRectF(0, 0, 1e-0, 1e-9)
        r.moveCenter(*points[0])
        p.addRect(r)
    elif len(points) == 0:
        r = QRectF(0, 0, 1e-16, 1e-16)
        r.moveCenter(QPointF(*anchor))
        p.addRect(r)
    return p


#: Dendrogram orientation flags
Left, Top, Right, Bottom = 1, 2, 3, 4


def dendrogram_path(tree, orientation=Left):
    layout = dendrogram_layout(tree)
    T = {}
    paths = {}
    rootdata = tree.value
    base = rootdata.height

    if orientation == Bottom:
        transform = lambda x, y: (x, y)
    if orientation == Top:
        transform = lambda x, y: (x, base - y)
    elif orientation == Left:
        transform = lambda x, y: (base - y, x)
    elif orientation == Right:
        transform = lambda x, y: (y, x)

    for node, (start, center, end) in layout:
        if node.is_leaf:
            x, y = transform(center, 0)
            anchor = Point(x, y)
            paths[node] = Element(anchor, ())
        else:
            left, right = paths[node.left], paths[node.right]
            lines = (left.anchor,
                     Point(*transform(start, node.value.height)),
                     Point(*transform(end, node.value.height)),
                     right.anchor)
            anchor = Point(*transform(center, node.value.height))
            paths[node] = Element(anchor, lines)

        T[node] = Tree((node, paths[node]),
                       tuple(T[ch] for ch in node.branches))
    return T[tree]


def make_pen(brush=Qt.black, width=1, style=Qt.SolidLine,
             cap_style=Qt.SquareCap, join_style=Qt.BevelJoin,
             cosmetic=False):
    pen = QPen(brush)
    pen.setWidth(width)
    pen.setStyle(style)
    pen.setCapStyle(cap_style)
    pen.setJoinStyle(join_style)
    pen.setCosmetic(cosmetic)
    return pen


def update_pen(pen, brush=None, width=None, style=None,
               cap_style=None, join_style=None,
               cosmetic=None):
    pen = QPen(pen)
    if brush is not None:
        pen.setBrush(QBrush(brush))
    if width is not None:
        pen.setWidth(width)
    if style is not None:
        pen.setStyle(style)
    if cap_style is not None:
        pen.setCapStyle(cap_style)
    if join_style is not None:
        pen.setJoinStyle(join_style)
    if cosmetic is not None:
        pen.setCosmetic(cosmetic)
    return pen


def path_stroke(path, width=1, join_style=Qt.MiterJoin):
    stroke = QPainterPathStroker()
    stroke.setWidth(width)
    stroke.setJoinStyle(join_style)
    stroke.setMiterLimit(1.0)
    return stroke.createStroke(path)


def path_outline(path, width=1, join_style=Qt.MiterJoin):
    stroke = path_stroke(path, width, join_style)
    return stroke.united(path)


@contextmanager
def blocked(obj):
    old = obj.signalsBlocked()
    obj.blockSignals(True)
    try:
        yield obj
    finally:
        obj.blockSignals(old)


class DendrogramWidget(QGraphicsWidget):
    """A Graphics Widget displaying a dendrogram."""

    class ClusterGraphicsItem(QGraphicsPathItem):
        _rect = None

        def shape(self):
            if self._rect is not None:
                p = QPainterPath()
                p.addRect(self.boundingRect())
                return p
            else:
                return super().shape()

        def setRect(self, rect):
            self.prepareGeometryChange()
            self._rect = QRectF(rect)

        def boundingRect(self):
            if self._rect is not None:
                return QRectF(self._rect)
            else:
                return super().boundingRect()

    #: Orientation
    Left, Top, Right, Bottom = 1, 2, 3, 4

    #: Selection flags
    NoSelection, SingleSelection, ExtendedSelection = 0, 1, 2

    #: Emitted when a user clicks on the cluster item.
    itemClicked = Signal(ClusterGraphicsItem)
    selectionChanged = Signal()
    selectionEdited = Signal()

    def __init__(self, parent=None, root=None, orientation=Left,
                 hoverHighlightEnabled=True, selectionMode=ExtendedSelection):

        QGraphicsWidget.__init__(self, parent)
        self.orientation = orientation
        self._root = None
        self._highlighted_item = None
        #: a list of selected items
        self._selection = OrderedDict()
        #: a {node: item} mapping
        self._items = {}
        #: container for all cluster items.
        self._itemgroup = QGraphicsWidget(self)
        self._itemgroup.setGeometry(self.contentsRect())
        self._cluster_parent = {}
        self.__hoverHighlightEnabled = hoverHighlightEnabled
        self.__selectionMode = selectionMode
        self.setContentsMargins(0, 0, 0, 0)
        self.set_root(root)

    def clear(self):
        for item in self._items.values():
            item.setParentItem(None)
            if item.scene() is self.scene() and self.scene() is not None:
                self.scene().removeItem(item)

        for item in self._selection.values():
            item.setParentItem(None)
            if item.scene():
                item.scene().removeItem(item)

        self._root = None
        self._items = {}
        self._selection = OrderedDict()
        self._highlighted_item = None
        self._cluster_parent = {}

    def set_root(self, root):
        """Set the root cluster.

        :param Tree root: Root tree.
        """
        self.clear()
        self._root = root
        if root:
            pen = make_pen(Qt.blue, width=1, cosmetic=True,
                           join_style=Qt.MiterJoin)
            for node in postorder(root):
                item = DendrogramWidget.ClusterGraphicsItem(self._itemgroup)
                item.setAcceptHoverEvents(True)
                item.setPen(pen)
                item.node = node
                item.installSceneEventFilter(self)
                for branch in node.branches:
                    assert branch in self._items
                    self._cluster_parent[branch] = node
                self._items[node] = item

            self.updateGeometry()
            self._relayout()
            self._rescale()

    def item(self, node):
        """Return the DendrogramNode instance representing the cluster.

        :type cluster: :class:`Tree`

        """
        return self._items.get(node)

    def height_at(self, point):
        """Return the cluster height at the point in widget local coordinates.
        """
        if not self._root:
            return 0

        tpoint = self.mapToItem(self._itemgroup, point)
        if self.orientation in [self.Left, self.Right]:
            height = tpoint.x()
        else:
            height = tpoint.y()

        if self.orientation in [self.Left, self.Bottom]:
            base = self._root.value.height
            height = base - height
        return height

    def pos_at_height(self, height):
        """Return a point in local coordinates for `height` (in cluster
        height scale).
        """
        if not self._root:
            return QPointF()

        if self.orientation in [self.Left, self.Bottom]:
            base = self._root.value.height
            height = base - height

        if self.orientation in [self.Left, self.Right]:
            p = QPointF(height, 0)
        else:
            p = QPointF(0, height)
        return self.mapFromItem(self._itemgroup, p)

    def _set_hover_item(self, item):
        """Set the currently highlighted item."""
        if self._highlighted_item is item:
            return

        def branches(item):
            return [self._items[ch] for ch in item.node.branches]

        if self._highlighted_item:
            pen = make_pen(Qt.blue, width=1, cosmetic=True)
            for it in postorder(self._highlighted_item, branches):
                it.setPen(pen)

        self._highlighted_item = item
        if item:
            hpen = make_pen(Qt.blue, width=2, cosmetic=True)
            for it in postorder(item, branches):
                it.setPen(hpen)

    def leaf_items(self):
        """Iterate over the dendrogram leaf items (:class:`QGraphicsItem`).
        """
        if self._root:
            return (self._items[leaf] for leaf in leaves(self._root))
        else:
            return iter(())

    def leaf_anchors(self):
        """Iterate over the dendrogram leaf anchor points (:class:`QPointF`).

        The points are in the widget local coordinates.
        """
        for item in self.leaf_items():
            anchor = QPointF(item.element.anchor)
            yield self.mapFromItem(item, anchor)

    def selected_nodes(self):
        """Return the selected clusters."""
        return [item.node for item in self._selection]

    def set_selected_items(self, items):
        """Set the item selection.

        :param items: List of `GraphicsItems`s to select.
        """
        to_remove = set(self._selection) - set(items)
        to_add = set(items) - set(self._selection)

        for sel in to_remove:
            self._remove_selection(sel)
        for sel in to_add:
            self._add_selection(sel)

        if to_add or to_remove:
            self._re_enumerate_selections()
            self.selectionChanged.emit()

    def set_selected_clusters(self, clusters):
        """Set the selected clusters.

        :param Tree items: List of cluster nodes to select .
        """
        self.set_selected_items(list(map(self.item, clusters)))

    def is_selected(self, item):
        return item in self._selection

    def is_included(self, item):
        return self._selected_super_item(item) is not None

    def select_item(self, item, state):
        """Set the `item`s selection state to `select_state`

        :param item: QGraphicsItem.
        :param bool state: New selection state for item.

        """
        if state is False and item not in self._selection or \
                state == True and item in self._selection:
            return  # State unchanged

        if item in self._selection:
            if state == False:
                self._remove_selection(item)
                self.selectionChanged.emit()
        else:
            # If item is already inside another selected item,
            # remove that selection
            super_selection = self._selected_super_item(item)

            if super_selection:
                self._remove_selection(super_selection)
            # Remove selections this selection will override.
            sub_selections = self._selected_sub_items(item)

            for sub in sub_selections:
                self._remove_selection(sub)

            if state:
                self._add_selection(item)
                self._re_enumerate_selections()

            elif item in self._selection:
                self._remove_selection(item)

            self.selectionChanged.emit()

    def _add_selection(self, item):
        """Add selection rooted at item
        """
        outline = self._selection_poly(item)
        selection_item = QGraphicsPathItem(self)
        selection_item.setPos(self.contentsRect().topLeft())
        selection_item.setPen(make_pen(width=1, cosmetic=True))

        transform = self._itemgroup.transform()
        path = transform.map(outline)
        margin = 4

        if item.node.is_leaf:
            ppath = QPainterPath()
            ppath.addRect(path.boundingRect()
                          .adjusted(-margin, -margin, margin, margin))
        else:
            ppath = QPainterPath()
            ppath.addPolygon(path)
            ppath = path_outline(ppath, width=margin * 2,)

        selection_item.setPath(ppath)
        selection_item.unscaled_path = outline
        self._selection[item] = selection_item

    def _remove_selection(self, item):
        """Remove selection rooted at item."""

        selection_item = self._selection[item]

        selection_item.hide()
        selection_item.setParentItem(None)
        if self.scene():
            self.scene().removeItem(selection_item)

        del self._selection[item]

        self._re_enumerate_selections()

    def _selected_sub_items(self, item):
        """Return all selected subclusters under item."""
        def branches(item):
            return [self._items[ch] for ch in item.node.branches]

        res = []
        for item in list(preorder(item, branches))[1:]:
            if item in self._selection:
                res.append(item)
        return res

    def _selected_super_item(self, item):
        """Return the selected super item if it exists."""
        def branches(item):
            return [self._items[ch] for ch in item.node.branches]

        for selected_item in self._selection:
            if item in set(preorder(selected_item, branches)):
                return selected_item
        return None

    def _re_enumerate_selections(self):
        """Re enumerate the selection items and update the colors."""
        # Order the clusters
        items = sorted(self._selection.items(),
                       key=lambda item: item[0].node.value.first)

        palette = colorpalette.ColorPaletteGenerator(len(items))
        for i, (item, selection_item) in enumerate(items):
            # delete and then reinsert to update the ordering
            del self._selection[item]
            self._selection[item] = selection_item
            color = palette[i]
            color.setAlpha(150)
            selection_item.setBrush(QColor(color))

    def _selection_poly(self, item):
        """Return an selection item covering the selection rooted at item.
        """
        def left(item):
            return [self._items[ch] for ch in item.node.branches[:1]]

        def right(item):
            return [self._items[ch] for ch in item.node.branches[-1:]]

        itemsleft = list(preorder(item, left))[::-1]
        itemsright = list(preorder(item, right))
        # itemsleft + itemsright walks from the leftmost leaf up to the root
        # and down to the rightmost leaf
        assert itemsleft[0].node.is_leaf
        assert itemsright[-1].node.is_leaf

        if item.node.is_leaf:
            # a single anchor point
            vert = [itemsleft[0].element.anchor]
        else:
            vert = []
            for it in itemsleft[1:]:
                vert.extend([it.element.path[0], it.element.path[1],
                             it.element.anchor])
            for it in itemsright[:-1]:
                vert.extend([it.element.anchor,
                             it.element.path[-2], it.element.path[-1]])
            # close the polygon
            vert.append(vert[0])

            def isclose(a, b, rel_tol=1e-6):
                return abs(a - b) < rel_tol * max(abs(a), abs(b))

            def isclose_p(p1, p2, rel_tol=1e-6):
                return isclose(p1.x, p2.x, rel_tol) and \
                       isclose(p1.y, p2.y, rel_tol)

            # merge consecutive vertices that are (too) close
            acc = [vert[0]]
            for v in vert[1:]:
                if not isclose_p(v, acc[-1]):
                    acc.append(v)
            vert = acc

        return QPolygonF([QPointF(*p) for p in vert])

    def _update_selection_items(self):
        """Update the shapes of selection items after a scale change.
        """
        transform = self._itemgroup.transform()
        for item, selection in self._selection.items():
            path = transform.map(selection.unscaled_path)
            ppath = QPainterPath()
            margin = 4
            if item.node.is_leaf:
                ppath.addRect(path.boundingRect()
                              .adjusted(-margin, -margin, margin, margin))
            else:
                ppath.addPolygon(path)
                ppath = path_outline(ppath, width=margin * 2)
            selection.setPath(ppath)

    def _relayout(self):
        if not self._root:
            return

        self._layout = dendrogram_path(self._root, self.orientation)
        for node_geom in postorder(self._layout):
            node, geom = node_geom.value
            item = self._items[node]
            item.element = geom

            item.setPath(Path_toQtPath(geom))
            item.setZValue(-node.value.height)
            r = item.path().boundingRect()
            base = self._root.value.height

            if self.orientation == Left:
                r.setRight(base)
            elif self.orientation == Right:
                r.setLeft(0)
            elif self.orientation == Top:
                r.setBottom(base)
            else:
                r.setTop(0)
            item.setRect(r)

    def _rescale(self):
        if self._root is None:
            return

        crect = self.contentsRect()
        leaf_count = len(list(leaves(self._root)))
        if self.orientation in [Left, Right]:
            drect = QSizeF(self._root.value.height, leaf_count)
        else:
            drect = QSizeF(leaf_count, self._root.value.height)

        eps = numpy.finfo(numpy.float64).eps

        if abs(drect.width()) < eps:
            sx = 1.0
        else:
            sx = crect.width() / drect.width()

        if abs(drect.height()) < eps:
            sy = 1.0
        else:
            sy = crect.height() / drect.height()

        transform = QTransform().scale(sx, sy)

        self._itemgroup.setPos(crect.topLeft())
        self._itemgroup.setTransform(transform)
        self._selection_items = None
        self._update_selection_items()

    def sizeHint(self, which, constraint=QSizeF()):
        fm = QFontMetrics(self.font())
        spacing = fm.lineSpacing()
        mleft, mtop, mright, mbottom = self.getContentsMargins()

        if self._root and which == Qt.PreferredSize:
            nleaves = len([node for node in self._items.keys()
                           if not node.branches])

            if self.orientation in [self.Left, self.Right]:
                return QSizeF(250, spacing * nleaves + mleft + mright)
            else:
                return QSizeF(spacing * nleaves + mtop + mbottom, 250)

        elif which == Qt.MinimumSize:
            return QSizeF(mleft + mright + 10, mtop + mbottom + 10)
        else:
            return QSizeF()

    def sceneEventFilter(self, obj, event):
        if isinstance(obj, DendrogramWidget.ClusterGraphicsItem):
            if event.type() == QEvent.GraphicsSceneHoverEnter and \
                    self.__hoverHighlightEnabled:
                self._set_hover_item(obj)
                event.accept()
                return True
            elif event.type() == QEvent.GraphicsSceneMousePress and \
                    event.button() == Qt.LeftButton:

                is_selected = self.is_selected(obj)
                is_included = self.is_included(obj)
                current_selection = list(self._selection)

                if self.__selectionMode == DendrogramWidget.SingleSelection:
                    if event.modifiers() & Qt.ControlModifier:
                        self.set_selected_items(
                            [obj] if not is_selected else [])
                    elif event.modifiers() & Qt.AltModifier:
                        self.set_selected_items([])
                    elif event.modifiers() & Qt.ShiftModifier:
                        if not is_included:
                            self.set_selected_items([obj])
                    elif current_selection != [obj]:
                        self.set_selected_items([obj])
                elif self.__selectionMode == DendrogramWidget.ExtendedSelection:
                    if event.modifiers() & Qt.ControlModifier:
                        self.select_item(obj, not is_selected)
                    elif event.modifiers() & Qt.AltModifier:
                        self.select_item(self._selected_super_item(obj), False)
                    elif event.modifiers() & Qt.ShiftModifier:
                        if not is_included:
                            self.select_item(obj, True)
                    elif current_selection != [obj]:
                        self.set_selected_items([obj])

                if current_selection != self._selection:
                    self.selectionEdited.emit()
                self.itemClicked.emit(obj)
                event.accept()
                return True

        if event.type() == QEvent.GraphicsSceneHoverLeave:
            self._set_hover_item(None)

        return super().sceneEventFilter(obj, event)

    def changeEvent(self, event):
        super().changeEvent(event)

        if event.type() == QEvent.FontChange:
            self.updateGeometry()

        # QEvent.ContentsRectChange is missing in PyQt4 <= 4.11.3
        if event.type() == 178:  # QEvent.ContentsRectChange:
            self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event):
        QGraphicsWidget.mousePressEvent(self, event)
        # A mouse press on an empty widget part
        if event.modifiers() == Qt.NoModifier and self._selection:
            self.set_selected_clusters([])


class OWHierarchicalClustering(widget.OWWidget):
    name = "Hierarchical Clustering"
    description = "Display a dendrogram of a hierarchical clustering " \
                  "constructed from the input distance matrix."
    icon = "icons/HierarchicalClustering.svg"
    priority = 2100

    inputs = [("Distances", Orange.misc.DistMatrix, "set_distances")]

    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)]

    settingsHandler = settings.DomainContextHandler()

    #: Selected linkage
    linkage = settings.Setting(1)
    #: Index of the selected annotation item (variable, ...)
    annotation = settings.ContextSetting("Enumeration")
    #: Out-of-context setting for the case when the "Name" option is available
    annotation_if_names = settings.Setting("Name")
    #: Out-of-context setting for the case with just "Enumerate" and "None"
    annotation_if_enumerate = settings.Setting("Enumerate")
    #: Selected tree pruning (none/max depth)
    pruning = settings.Setting(0)
    #: Maximum depth when max depth pruning is selected
    max_depth = settings.Setting(10)

    #: Selected cluster selection method (none, cut distance, top n)
    selection_method = settings.Setting(0)
    #: Cut height ratio wrt root height
    cut_ratio = settings.Setting(75.0)
    #: Number of top clusters to select
    top_n = settings.Setting(3)

    #: Dendrogram zoom factor
    zoom_factor = settings.Setting(0)

    append_clusters = settings.Setting(True)
    cluster_role = settings.Setting(2)
    cluster_name = settings.Setting("Cluster")
    autocommit = settings.Setting(True)

    graph_name = "scene"

    #: Cluster variable domain role
    AttributeRole, ClassRole, MetaRole = 0, 1, 2

    cluster_roles = ["Attribute", "Class variable", "Meta variable"]
    basic_annotations = ["None", "Enumeration"]

    def __init__(self):
        super().__init__()

        self.matrix = None
        self.items = None
        self.linkmatrix = None
        self.root = None
        self._displayed_root = None
        self.cutoff_height = 0.0

        gui.comboBox(
            self.controlArea, self, "linkage", items=LINKAGE, box="Linkage",
            callback=self._invalidate_clustering)

        model = itemmodels.VariableListModel()
        model[:] = self.basic_annotations
        self.label_cb = gui.comboBox(
            self.controlArea, self, "annotation", box="Annotation",
            model=model, callback=self._update_labels, contentsLength=12)

        box = gui.radioButtons(
            self.controlArea, self, "pruning", box="Pruning",
            callback=self._invalidate_pruning)
        grid = QGridLayout()
        box.layout().addLayout(grid)
        grid.addWidget(
            gui.appendRadioButton(box, "None", addToLayout=False),
            0, 0
        )
        self.max_depth_spin = gui.spin(
            box, self, "max_depth", minv=1, maxv=100,
            callback=self._invalidate_pruning,
            keyboardTracking=False
        )

        grid.addWidget(
            gui.appendRadioButton(box, "Max depth:", addToLayout=False),
            1, 0)
        grid.addWidget(self.max_depth_spin, 1, 1)

        self.selection_box = gui.radioButtons(
            self.controlArea, self, "selection_method",
            box="Selection",
            callback=self._selection_method_changed)

        grid = QGridLayout()
        self.selection_box.layout().addLayout(grid)
        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Manual", addToLayout=False),
            0, 0
        )
        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Height ratio:", addToLayout=False),
            1, 0
        )
        self.cut_ratio_spin = gui.spin(
            self.selection_box, self, "cut_ratio", 0, 100, step=1e-1,
            spinType=float, callback=self._selection_method_changed
        )
        self.cut_ratio_spin.setSuffix("%")

        grid.addWidget(self.cut_ratio_spin, 1, 1)

        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Top N:", addToLayout=False),
            2, 0
        )
        self.top_n_spin = gui.spin(self.selection_box, self, "top_n", 1, 20,
                                   callback=self._selection_method_changed)
        grid.addWidget(self.top_n_spin, 2, 1)

        self.zoom_slider = gui.hSlider(
            self.controlArea, self, "zoom_factor", box="Zoom",
            minValue=-6, maxValue=3, step=1, ticks=True, createLabel=False,
            callback=self.__zoom_factor_changed)

        zoom_in = QAction(
            "Zoom in", self, shortcut=QKeySequence.ZoomIn,
            triggered=self.__zoom_in
        )
        zoom_out = QAction(
            "Zoom out", self, shortcut=QKeySequence.ZoomOut,
            triggered=self.__zoom_out
        )
        zoom_reset = QAction(
            "Reset zoom", self,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=self.__zoom_reset
        )
        self.addActions([zoom_in, zoom_out, zoom_reset])

        self.controlArea.layout().addStretch()

        box = gui.vBox(self.controlArea, "Output")
        gui.checkBox(box, self, "append_clusters", "Append cluster IDs",
                     callback=self._invalidate_output)

        ibox = gui.indentedBox(box)
        name_edit = gui.lineEdit(ibox, self, "cluster_name")
        name_edit.editingFinished.connect(self._invalidate_output)

        cb = gui.comboBox(
            ibox, self, "cluster_role", callback=self._invalidate_output,
            items=self.cluster_roles
        )
        form = QFormLayout(
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            labelAlignment=Qt.AlignLeft,
            spacing=8
        )
        form.addRow("Name:", name_edit)
        form.addRow("Place:", cb)

        ibox.layout().addSpacing(5)
        ibox.layout().addLayout(form)
        ibox.layout().addSpacing(5)

        gui.auto_commit(box, self, "autocommit", "Send Selected", "Send Automatically",
                        box=False)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(
            self.scene,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            alignment=Qt.AlignLeft | Qt.AlignVCenter
        )

        def axis_view(orientation):
            ax = pg.AxisItem(orientation=orientation, maxTickLength=7)
            scene = QGraphicsScene()
            scene.addItem(ax)
            view = QGraphicsView(
                scene,
                horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
                verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                alignment=Qt.AlignLeft | Qt.AlignVCenter
            )
            view.setFixedHeight(ax.size().height())
            ax.line = SliderLine(orientation=Qt.Horizontal,
                                 length=ax.size().height())
            scene.addItem(ax.line)
            return view, ax

        self.top_axis_view, self.top_axis = axis_view("top")
        self.mainArea.layout().setSpacing(1)
        self.mainArea.layout().addWidget(self.top_axis_view)
        self.mainArea.layout().addWidget(self.view)
        self.bottom_axis_view, self.bottom_axis = axis_view("bottom")
        self.mainArea.layout().addWidget(self.bottom_axis_view)

        self._main_graphics = QGraphicsWidget()
        self._main_layout = QGraphicsLinearLayout(Qt.Horizontal)
        self._main_layout.setSpacing(10)

        self._main_graphics.setLayout(self._main_layout)
        self.scene.addItem(self._main_graphics)

        self.dendrogram = DendrogramWidget()
        self.dendrogram.setSizePolicy(QSizePolicy.MinimumExpanding,
                                      QSizePolicy.MinimumExpanding)
        self.dendrogram.selectionChanged.connect(self._invalidate_output)
        self.dendrogram.selectionEdited.connect(self._selection_edited)

        self.labels = GraphicsSimpleTextList()
        self.labels.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.labels.setAlignment(Qt.AlignLeft)
        self.labels.setMaximumWidth(200)
        self.labels.layout().setSpacing(0)

        self._main_layout.addItem(self.dendrogram)
        self._main_layout.addItem(self.labels)

        self._main_layout.setAlignment(
            self.dendrogram, Qt.AlignLeft | Qt.AlignVCenter)
        self._main_layout.setAlignment(
            self.labels, Qt.AlignLeft | Qt.AlignVCenter)

        self.view.viewport().installEventFilter(self)
        self.top_axis_view.viewport().installEventFilter(self)
        self.bottom_axis_view.viewport().installEventFilter(self)
        self._main_graphics.installEventFilter(self)

        self.cut_line = SliderLine(self.dendrogram,
                                   orientation=Qt.Horizontal)
        self.cut_line.valueChanged.connect(self._dendrogram_slider_changed)
        self.cut_line.hide()

        self.bottom_axis.line.valueChanged.connect(self._axis_slider_changed)
        self.top_axis.line.valueChanged.connect(self._axis_slider_changed)
        self.dendrogram.geometryChanged.connect(self._dendrogram_geom_changed)
        self._set_cut_line_visible(self.selection_method == 1)

    def set_distances(self, matrix):
        self.error()
        if matrix is not None:
            N, _ = matrix.shape
            if N < 2:
                self.error("Empty distance matrix")
                matrix = None

        self.matrix = matrix
        if matrix is not None:
            self._set_items(matrix.row_items, matrix.axis)
        else:
            self._set_items(None)
        self._invalidate_clustering()

        self.unconditional_commit()

    def _set_items(self, items, axis=1):
        self.closeContext()
        self.items = items
        model = self.label_cb.model()
        if len(model) == 3:
            self.annotation_if_names = self.annotation
        elif len(model) == 2:
            self.annotation_if_enumerate = self.annotation
        if isinstance(items, Orange.data.Table) and axis:
            model[:] = chain(
                self.basic_annotations,
                [model.Separator],
                items.domain.class_vars,
                items.domain.metas,
                [model.Separator] if (items.domain.class_vars or items.domain.metas) and
                                     next(filter_visible(items.domain.attributes), False) else [],
                filter_visible(items.domain.attributes)
            )
            if items.domain.class_vars:
                self.annotation = items.domain.class_vars[0]
            else:
                self.annotation = "Enumeration"
            self.openContext(items.domain)
        else:
            name_option = bool(
                items is not None and (
                not axis or
                isinstance(items, list) and
                all(isinstance(var, Orange.data.Variable) for var in items)))
            model[:] = self.basic_annotations + ["Name"] * name_option
            self.annotation = self.annotation_if_names if name_option \
                else self.annotation_if_enumerate

    def _clear_plot(self):
        self.labels.set_labels([])
        self.dendrogram.set_root(None)

    def _set_displayed_root(self, root):
        self._clear_plot()
        self._displayed_root = root
        self.dendrogram.set_root(root)

        self._update_labels()

        self._main_graphics.resize(
            self._main_graphics.size().width(),
            self._main_graphics.sizeHint(Qt.PreferredSize).height()
        )
        self._main_graphics.layout().activate()

    def _update(self):
        self._clear_plot()

        distances = self.matrix

        if distances is not None:
            method = LINKAGE[self.linkage].lower()
            Z = dist_matrix_linkage(distances, linkage=method)

            tree = tree_from_linkage(Z)
            self.linkmatrix = Z
            self.root = tree

            self.top_axis.setRange(tree.value.height, 0.0)
            self.bottom_axis.setRange(tree.value.height, 0.0)

            if self.pruning:
                self._set_displayed_root(prune(tree, level=self.max_depth))
            else:
                self._set_displayed_root(tree)
        else:
            self.linkmatrix = None
            self.root = None
            self._set_displayed_root(None)

        self._apply_selection()

    def _update_labels(self):
        labels = []
        if self.root and self._displayed_root:
            indices = [leaf.value.index for leaf in leaves(self.root)]

            if self.annotation == "None":
                labels = []
            elif self.annotation == "Enumeration":
                labels = [str(i+1) for i in indices]
            elif self.annotation == "Name":
                attr = self.matrix.row_items.domain.attributes
                labels = [str(attr[i]) for i in indices]
            elif isinstance(self.annotation, Orange.data.Variable):
                col_data, _ = self.items.get_column_view(self.annotation)
                labels = [self.annotation.str_val(val) for val in col_data]
                labels = [labels[idx] for idx in indices]
            else:
                labels = []

            if labels and self._displayed_root is not self.root:
                joined = leaves(self._displayed_root)
                labels = [", ".join(labels[leaf.value.first: leaf.value.last])
                          for leaf in joined]

        self.labels.set_labels(labels)
        self.labels.setMinimumWidth(1 if labels else -1)

    def _invalidate_clustering(self):
        self._update()
        self._update_labels()
        self._invalidate_output()

    def _invalidate_output(self):
        self.commit()

    def _invalidate_pruning(self):
        if self.root:
            selection = self.dendrogram.selected_nodes()
            ranges = [node.value.range for node in selection]
            if self.pruning:
                self._set_displayed_root(
                    prune(self.root, level=self.max_depth))
            else:
                self._set_displayed_root(self.root)
            selected = [node for node in preorder(self._displayed_root)
                        if node.value.range in ranges]

            self.dendrogram.set_selected_clusters(selected)

        self._apply_selection()

    def commit(self):
        items = getattr(self.matrix, "items", self.items)
        if not items:
            self.send("Selected Data", None)
            self.send(ANNOTATED_DATA_SIGNAL_NAME, None)
            return

        selection = self.dendrogram.selected_nodes()
        selection = sorted(selection, key=lambda c: c.value.first)

        indices = [leaf.value.index for leaf in leaves(self.root)]

        maps = [indices[node.value.first:node.value.last]
                for node in selection]

        selected_indices = list(chain(*maps))
        unselected_indices = sorted(set(range(self.root.value.last)) -
                                    set(selected_indices))

        if not selected_indices:
            self.send("Selected Data", None)
            annotated_data = create_annotated_table(items, []) \
                if self.selection_method == 0 and self.matrix.axis else None
            self.send(ANNOTATED_DATA_SIGNAL_NAME, annotated_data)
            return

        selected_data = None

        if isinstance(items, Orange.data.Table) and self.matrix.axis == 1:
            # Select rows
            c = numpy.zeros(self.matrix.shape[0])

            for i, indices in enumerate(maps):
                c[indices] = i
            c[unselected_indices] = len(maps)

            mask = c != len(maps)

            if self.append_clusters:
                clust_var = Orange.data.DiscreteVariable(
                    str(self.cluster_name),
                    values=["C{}".format(i + 1)
                            for i in range(len(maps))] +
                           ["Other"]
                )
                data, domain = items, items.domain

                attrs = domain.attributes
                class_ = domain.class_vars
                metas = domain.metas

                if self.cluster_role == self.AttributeRole:
                    attrs = attrs + (clust_var,)
                elif self.cluster_role == self.ClassRole:
                    class_ = class_ + (clust_var,)
                elif self.cluster_role == self.MetaRole:
                    metas = metas + (clust_var,)

                domain = Orange.data.Domain(attrs, class_, metas)
                data = Orange.data.Table.from_table(domain, items)
                data.get_column_view(clust_var)[0][:] = c
            else:
                data = items

            if selected_indices:
                selected_data = data[mask]
                if self.append_clusters:
                    def remove_other_value(vars_):
                        vars_ = list(vars_)
                        clust_var = vars_[-1].copy()
                        clust_var.values.pop()
                        vars_[-1] = clust_var
                        return vars_
                    if self.cluster_role == self.AttributeRole:
                        attrs = remove_other_value(attrs)
                    elif self.cluster_role == self.ClassRole:
                        class_ = remove_other_value(class_)
                    elif self.cluster_role == self.MetaRole:
                        metas = remove_other_value(metas)
                    selected_data.domain = Domain(attrs, class_, metas)

        elif isinstance(items, Orange.data.Table) and self.matrix.axis == 0:
            # Select columns
            domain = Orange.data.Domain(
                [items.domain[i] for i in selected_indices],
                items.domain.class_vars, items.domain.metas)
            selected_data = items.from_table(domain, items)
            data = None

        self.send("Selected Data", selected_data)
        annotated_data = create_annotated_table(data, selected_indices) if \
            self.selection_method == 0 else None
        self.send(ANNOTATED_DATA_SIGNAL_NAME, annotated_data)

    def sizeHint(self):
        return QSize(800, 500)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.Resize:
            width = self.view.viewport().width() - 2
            self._main_graphics.setMaximumWidth(width)
            self._main_graphics.setMinimumWidth(width)
            self._main_graphics.layout().activate()
        elif event.type() == QEvent.MouseButtonPress and \
                (obj is self.top_axis_view.viewport() or
                 obj is self.bottom_axis_view.viewport()):
            self.selection_method = 1
            # Map click point to cut line local coordinates
            pos = self.top_axis_view.mapToScene(event.pos())
            cut = self.top_axis.line.mapFromScene(pos)
            self.top_axis.line.setValue(cut.x())
            # update the line visibility, output, ...
            self._selection_method_changed()
        elif obj is self._main_graphics and \
                event.type() == QEvent.LayoutRequest:
            self.__update_size_constraints()

        return super().eventFilter(obj, event)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._clear_plot()
        self.dendrogram.clear()
        self.dendrogram.deleteLater()

    def _dendrogram_geom_changed(self):
        pos = self.dendrogram.pos_at_height(self.cutoff_height)
        geom = self.dendrogram.geometry()
        crect = self.dendrogram.contentsRect()

        self._set_slider_value(pos.x(), geom.width())
        self.cut_line.setLength(geom.height())

        self.top_axis.resize(crect.width(), self.top_axis.height())
        self.top_axis.setPos(geom.left() + crect.left(), 0)
        self.top_axis.line.setPos(self.cut_line.scenePos().x(), 0)

        self.bottom_axis.resize(crect.width(), self.bottom_axis.height())
        self.bottom_axis.setPos(geom.left() + crect.left(), 0)
        self.bottom_axis.line.setPos(self.cut_line.scenePos().x(), 0)

        geom = self._main_graphics.geometry()
        assert geom.topLeft() == QPointF(0, 0)
        self.scene.setSceneRect(geom)

        geom.setHeight(self.top_axis.size().height())

        self.top_axis.scene().setSceneRect(geom)
        self.bottom_axis.scene().setSceneRect(geom)

    def _axis_slider_changed(self, value):
        self.cut_line.setValue(value)

    def _dendrogram_slider_changed(self, value):
        p = QPointF(value, 0)
        cl_height = self.dendrogram.height_at(p)

        self.set_cutoff_height(cl_height)

        # Sync the cut positions between the dendrogram and the axis.
        self._set_slider_value(value, self.dendrogram.size().width())

    def _set_slider_value(self, value, span):
        with blocked(self.cut_line):
            self.cut_line.setValue(value)
            self.cut_line.setRange(0, span)

        with blocked(self.top_axis.line):
            self.top_axis.line.setValue(value)
            self.top_axis.line.setRange(0, span)

        with blocked(self.bottom_axis.line):
            self.bottom_axis.line.setValue(value)
            self.bottom_axis.line.setRange(0, span)

    def set_cutoff_height(self, height):
        self.cutoff_height = height
        if self.root:
            self.cut_ratio = 100 * height / self.root.value.height
        self.select_max_height(height)

    def _set_cut_line_visible(self, visible):
        self.cut_line.setVisible(visible)
        self.top_axis.line.setVisible(visible)
        self.bottom_axis.line.setVisible(visible)

    def select_top_n(self, n):
        root = self._displayed_root
        if root:
            clusters = top_clusters(root, n)
            self.dendrogram.set_selected_clusters(clusters)

    def select_max_height(self, height):
        root = self._displayed_root
        if root:
            clusters = clusters_at_height(root, height)
            self.dendrogram.set_selected_clusters(clusters)

    def _selection_method_changed(self):
        self._set_cut_line_visible(self.selection_method == 1)
        if self.root:
            self._apply_selection()

    def _apply_selection(self):
        if not self.root:
            return

        if self.selection_method == 0:
            pass
        elif self.selection_method == 1:
            height = self.cut_ratio * self.root.value.height / 100
            self.set_cutoff_height(height)
            pos = self.dendrogram.pos_at_height(height)
            self._set_slider_value(pos.x(), self.dendrogram.size().width())
        elif self.selection_method == 2:
            self.select_top_n(self.top_n)

    def _selection_edited(self):
        # Selection was edited by clicking on a cluster in the
        # dendrogram view.
        self.selection_method = 0
        self._selection_method_changed()
        self._invalidate_output()

    def __zoom_in(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor + 1)
        self.__zoom_factor_changed()

    def __zoom_out(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor - 1)
        self.__zoom_factor_changed()

    def __zoom_reset(self):
        self.zoom_factor = 0
        self.__zoom_factor_changed()

    def __update_size_constraints(self):
        size = self._main_graphics.size()
        preferred = self._main_graphics.sizeHint(
            Qt.PreferredSize, constraint=QSizeF(size.width(), -1))
        self._main_graphics.resize(QSizeF(size.width(), preferred.height()))
        self._main_graphics.layout().activate()

    def __zoom_factor_changed(self):
        font = self.scene.font()
        factor = (1.25 ** self.zoom_factor)
        font = qfont_scaled(font, factor)
        self.labels.setFont(font)
        self.dendrogram.setFont(font)
        self.__update_size_constraints()

    def send_report(self):
        annot = self.label_cb.currentText()
        if isinstance(self.annotation, str):
            annot = annot.lower()
        if self.selection_method == 0:
            sel = "manual"
        elif self.selection_method == 1:
            sel = "at {:.1f} of height".format(self.cut_ratio)
        else:
            sel = "top {} clusters".format(self.top_n)
        self.report_items((
            ("Linkage", LINKAGE[self.linkage].lower()),
            ("Annotation", annot),
            ("Prunning",
             self.pruning != 0 and "{} levels".format(self.max_depth)),
            ("Selection", sel),
            ("Cluster ID in output",
             self.append_clusters and
             "{} (as {})".format(
                 self.cluster_name,
                 self.cluster_roles[self.cluster_role].lower()))
        ))
        self.report_plot()


def qfont_scaled(font, factor):
    scaled = QFont(font)
    if font.pointSizeF() != -1:
        scaled.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() != -1:
        scaled.setPixelSize(int(font.pixelSize() * factor))
    return scaled


class GraphicsSimpleTextList(QGraphicsWidget):
    """A simple text list widget."""

    def __init__(self, labels=[], orientation=Qt.Vertical,
                 alignment=Qt.AlignCenter, parent=None):
        QGraphicsWidget.__init__(self, parent)
        layout = QGraphicsLinearLayout(orientation)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.orientation = orientation
        self.alignment = alignment
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_items = []
        self.set_labels(labels)

    def clear(self):
        """Remove all text items."""
        layout = self.layout()
        for i in reversed(range(layout.count())):
            witem = layout.itemAt(i)
            witem.item.setParentItem(None)
            if self.scene():
                self.scene().removeItem(witem.item)
            layout.removeAt(i)

        self.label_items = []
        self.updateGeometry()

    def set_labels(self, labels):
        """Set the text labels."""
        self.clear()
        orientation = Qt.Horizontal if self.orientation == Qt.Vertical else Qt.Vertical
        for text in labels:
            item = QGraphicsSimpleTextItem(text, self)
            item.setFont(self.font())
            item.setToolTip(text)
            witem = WrapperLayoutItem(item, orientation, parent=self)
            self.layout().addItem(witem)
            self.layout().setAlignment(witem, self.alignment)
            self.label_items.append(item)

        self.layout().activate()
        self.updateGeometry()

    def setAlignment(self, alignment):
        """Set alignment of text items in the widget
        """
        self.alignment = alignment
        layout = self.layout()
        for i in range(layout.count()):
            layout.setAlignment(layout.itemAt(i), alignment)

    def setVisible(self, visible):
        QGraphicsWidget.setVisible(self, visible)
        self.updateGeometry()

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.__update_font()
        return super().changeEvent(event)

    def __iter__(self):
        return iter(self.label_items)

    def __update_font(self):
        for item in self.label_items:
            item.setFont(self.font())

        layout = self.layout()
        for i in range(layout.count()):
            layout.itemAt(i).updateGeometry()

        self.layout().invalidate()
        self.updateGeometry()


class WrapperLayoutItem(QGraphicsLayoutItem):
    """A Graphics layout item wrapping a QGraphicsItem allowing it
    to be managed by a layout.
    """
    def __init__(self, item, orientation=Qt.Horizontal, parent=None):
        QGraphicsLayoutItem.__init__(self, parent)
        self.orientation = orientation
        self.item = item
        if orientation == Qt.Vertical:
            self.item.setRotation(-90)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def setGeometry(self, rect):
        QGraphicsLayoutItem.setGeometry(self, rect)
        if self.orientation == Qt.Horizontal:
            self.item.setPos(rect.topLeft())
        else:
            self.item.setPos(rect.bottomLeft())

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            size = self.item.boundingRect().size()
            if self.orientation == Qt.Horizontal:
                return size
            else:
                return QSizeF(size.height(), size.width())
        else:
            return QSizeF()

    def setFont(self, font):
        self.item.setFont(font)
        self.updateGeometry()

    def setText(self, text):
        self.item.setText(text)
        self.updateGeometry()

    def setToolTip(self, tip):
        self.item.setToolTip(tip)


class SliderLine(QGraphicsObject):
    """A movable slider line."""
    valueChanged = Signal(float)

    linePressed = Signal()
    lineMoved = Signal()
    lineReleased = Signal()
    rangeChanged = Signal(float, float)

    def __init__(self, parent=None, orientation=Qt.Vertical, value=0.0,
                 length=10.0, **kwargs):
        self._orientation = orientation
        self._value = value
        self._length = length
        self._min = 0.0
        self._max = 1.0
        self._line = QLineF()
        self._pen = QPen()
        super().__init__(parent, **kwargs)

        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setPen(make_pen(brush=QColor(50, 50, 50), width=1, cosmetic=True))

        if self._orientation == Qt.Vertical:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeHorCursor)

    def setPen(self, pen):
        pen = QPen(pen)
        if self._pen != pen:
            self.prepareGeometryChange()
            self._pen = pen
            self._line = None
            self.update()

    def pen(self):
        return QPen(self._pen)

    def setValue(self, value):
        value = min(max(value, self._min), self._max)

        if self._value != value:
            self.prepareGeometryChange()
            self._value = value
            self._line = None
            self.valueChanged.emit(value)

    def value(self):
        return self._value

    def setRange(self, minval, maxval):
        maxval = max(minval, maxval)
        if minval != self._min or maxval != self._max:
            self._min = minval
            self._max = maxval
            self.rangeChanged.emit(minval, maxval)
            self.setValue(self._value)

    def setLength(self, length):
        if self._length != length:
            self.prepareGeometryChange()
            self._length = length
            self._line = None

    def length(self):
        return self._length

    def setOrientation(self, orientation):
        if self._orientation != orientation:
            self.prepareGeometryChange()
            self._orientation = orientation
            self._line = None
            if self._orientation == Qt.Vertical:
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, event):
        event.accept()
        self.linePressed.emit()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._orientation == Qt.Vertical:
            self.setValue(pos.y())
        else:
            self.setValue(pos.x())
        self.lineMoved.emit()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._orientation == Qt.Vertical:
            self.setValue(event.pos().y())
        else:
            self.setValue(event.pos().x())
        self.lineReleased.emit()
        event.accept()

    def boundingRect(self):
        if self._line is None:
            if self._orientation == Qt.Vertical:
                self._line = QLineF(0, self._value, self._length, self._value)
            else:
                self._line = QLineF(self._value, 0, self._value, self._length)
        r = QRectF(self._line.p1(), self._line.p2())
        penw = self.pen().width()
        return r.adjusted(-penw, -penw, penw, penw)

    def paint(self, painter, *args):
        if self._line is None:
            self.boundingRect()

        painter.save()
        painter.setPen(self.pen())
        painter.drawLine(self._line)
        painter.restore()


def clusters_at_height(root, height):
    """Return a list of clusters by cutting the clustering at `height`.
    """
    lower = set()
    cluster_list = []
    for cl in preorder(root):
        if cl in lower:
            continue
        if cl.value.height < height:
            cluster_list.append(cl)
            lower.update(preorder(cl))
    return cluster_list


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    import sip
    import Orange.distance as distance

    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris.tab"

    w = OWHierarchicalClustering()

    data = Orange.data.Table(filename)
    matrix = distance.Euclidean(distance._preprocess(data))

    w.set_distances(matrix)
    w.handleNewSignals()
    w.show()
    w.raise_()
    rval = app.exec_()
    w.set_distances(None)
    w.handleNewSignals()

    w.onDeleteWidget()
    sip.delete(w)
    del w
    app.processEvents()
    return rval

if __name__ == "__main__":
    sys.exit(main())
