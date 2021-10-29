from itertools import chain
from contextlib import contextmanager

import typing
from typing import Any, List, Tuple, Dict, Optional, Set, Union

import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsObject, QGraphicsScene, QGridLayout, QSizePolicy,
    QAction, QComboBox, QGraphicsGridLayout, QGraphicsSceneMouseEvent
)
from AnyQt.QtGui import QColor, QPen, QFont, QKeySequence
from AnyQt.QtCore import Qt, QSize, QSizeF, QPointF, QRectF, QLineF, QEvent
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

import Orange.data
from Orange.data.domain import filter_visible
from Orange.data import Domain
import Orange.misc
from Orange.clustering.hierarchical import \
    postorder, preorder, Tree, tree_from_linkage, dist_matrix_linkage, \
    leaves, prune, top_clusters
from Orange.data.util import get_unique_names

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, combobox
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, Msg

from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.graphicstextlist import TextListWidget
from Orange.widgets.utils.dendrogram import DendrogramWidget

__all__ = ["OWHierarchicalClustering"]


LINKAGE = ["Single", "Average", "Weighted", "Complete", "Ward"]


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


@contextmanager
def blocked(obj):
    old = obj.signalsBlocked()
    obj.blockSignals(True)
    try:
        yield obj
    finally:
        obj.blockSignals(old)


class SaveStateSettingsHandler(settings.SettingsHandler):
    """
    A settings handler that delegates session data store/restore to the
    OWWidget instance.

    The OWWidget subclass must implement `save_state() -> Dict[str, Any]` and
    `set_restore_state(state: Dict[str, Any])` methods.
    """
    def initialize(self, instance, data=None):
        super().initialize(instance, data)
        if data is not None and "__session_state_data" in data:
            session_data = data["__session_state_data"]
            instance.set_restore_state(session_data)

    def pack_data(self, widget):
        # type: (widget.OWWidget) -> dict
        res = super().pack_data(widget)
        state = widget.save_state()
        if state:
            assert "__session_state_data" not in res
            res["__session_state_data"] = state
        return res


class _DomainContextHandler(settings.DomainContextHandler,
                            SaveStateSettingsHandler):
    pass


if typing.TYPE_CHECKING:
    #: Encoded selection state for persistent storage.
    #: This is a list of tuples of leaf indices in the selection and
    #: a (N, 3) linkage matrix for validation (the 4-th column from scipy
    #: is omitted).
    SelectionState = Tuple[List[Tuple[int]], List[Tuple[int, int, float]]]


class OWHierarchicalClustering(widget.OWWidget):
    name = "Hierarchical Clustering"
    description = "Display a dendrogram of a hierarchical clustering " \
                  "constructed from the input distance matrix."
    icon = "icons/HierarchicalClustering.svg"
    priority = 2100
    keywords = []

    class Inputs:
        distances = Input("Distances", Orange.misc.DistMatrix)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    settingsHandler = _DomainContextHandler()

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

    autocommit = settings.Setting(True)

    graph_name = "scene"

    basic_annotations = ["None", "Enumeration"]

    class Error(widget.OWWidget.Error):
        not_finite_distances = Msg("Some distances are infinite")

    #: Stored (manual) selection state (from a saved workflow) to restore.
    __pending_selection_restore = None  # type: Optional[SelectionState]

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

        box = gui.widgetBox(self.controlArea, "Annotations")
        self.label_cb = cb = combobox.ComboBoxSearch(
            minimumContentsLength=14,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        cb.setModel(model)
        cb.setCurrentIndex(cb.findData(self.annotation, Qt.EditRole))

        def on_annotation_activated():
            self.annotation = cb.currentData(Qt.EditRole)
            self._update_labels()
        cb.activated.connect(on_annotation_activated)

        def on_annotation_changed(value):
            cb.setCurrentIndex(cb.findData(value, Qt.EditRole))
        self.connect_control("annotation", on_annotation_changed)

        box.layout().addWidget(self.label_cb)

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
            keyboardTracking=False, addToLayout=False
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
            spinType=float, callback=self._selection_method_changed,
            addToLayout=False
        )
        self.cut_ratio_spin.setSuffix("%")

        grid.addWidget(self.cut_ratio_spin, 1, 1)

        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Top N:", addToLayout=False),
            2, 0
        )
        self.top_n_spin = gui.spin(self.selection_box, self, "top_n", 1, 20,
                                   callback=self._selection_method_changed,
                                   addToLayout=False)
        grid.addWidget(self.top_n_spin, 2, 1)

        self.zoom_slider = gui.hSlider(
            self.controlArea, self, "zoom_factor", box="Zoom",
            minValue=-6, maxValue=3, step=1, ticks=True, createLabel=False,
            callback=self.__update_font_scale)

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

        gui.auto_send(self.buttonsArea, self, "autocommit")

        self.scene = QGraphicsScene(self)
        self.view = StickyGraphicsView(
            self.scene,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            alignment=Qt.AlignLeft | Qt.AlignVCenter
        )
        self.mainArea.layout().setSpacing(1)
        self.mainArea.layout().addWidget(self.view)

        def axis_view(orientation):
            ax = AxisItem(orientation=orientation, maxTickLength=7)
            ax.mousePressed.connect(self._activate_cut_line)
            ax.mouseMoved.connect(self._activate_cut_line)
            ax.mouseReleased.connect(self._activate_cut_line)
            ax.setRange(1.0, 0.0)
            return ax

        self.top_axis = axis_view("top")
        self.bottom_axis = axis_view("bottom")

        self._main_graphics = QGraphicsWidget()
        scenelayout = QGraphicsGridLayout()
        scenelayout.setHorizontalSpacing(10)
        scenelayout.setVerticalSpacing(10)

        self._main_graphics.setLayout(scenelayout)
        self.scene.addItem(self._main_graphics)

        self.dendrogram = DendrogramWidget()
        self.dendrogram.setSizePolicy(QSizePolicy.MinimumExpanding,
                                      QSizePolicy.MinimumExpanding)
        self.dendrogram.selectionChanged.connect(self._invalidate_output)
        self.dendrogram.selectionEdited.connect(self._selection_edited)

        self.labels = TextListWidget()
        self.labels.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.labels.setAlignment(Qt.AlignLeft)
        self.labels.setMaximumWidth(200)

        scenelayout.addItem(self.top_axis, 0, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.dendrogram, 1, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.labels, 1, 1,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        scenelayout.addItem(self.bottom_axis, 2, 0,
                            alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.view.viewport().installEventFilter(self)
        self._main_graphics.installEventFilter(self)

        self.top_axis.setZValue(self.dendrogram.zValue() + 10)
        self.bottom_axis.setZValue(self.dendrogram.zValue() + 10)
        self.cut_line = SliderLine(self.top_axis,
                                   orientation=Qt.Horizontal)
        self.cut_line.valueChanged.connect(self._dendrogram_slider_changed)
        self.dendrogram.geometryChanged.connect(self._dendrogram_geom_changed)
        self._set_cut_line_visible(self.selection_method == 1)
        self.__update_font_scale()

    @Inputs.distances
    def set_distances(self, matrix):
        if self.__pending_selection_restore is not None:
            selection_state = self.__pending_selection_restore
        else:
            # save the current selection to (possibly) restore later
            selection_state = self._save_selection()

        self.error()
        self.Error.clear()
        if matrix is not None:
            N, _ = matrix.shape
            if N < 2:
                self.error("Empty distance matrix")
                matrix = None
        if matrix is not None:
            if not np.all(np.isfinite(matrix)):
                self.Error.not_finite_distances()
                matrix = None

        self.matrix = matrix
        if matrix is not None:
            self._set_items(matrix.row_items, matrix.axis)
        else:
            self._set_items(None)
        self._invalidate_clustering()

        # Can now attempt to restore session state from a saved workflow.
        if self.root and selection_state is not None:
            self._restore_selection(selection_state)
            self.__pending_selection_restore = None

        self.commit.now()

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
        self.dendrogram.set_root(None)
        self.labels.setItems([])

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

        self.labels.setItems(labels)
        self.labels.setMinimumWidth(1 if labels else -1)

    def _restore_selection(self, state):
        # type: (SelectionState) -> bool
        """
        Restore the (manual) node selection state.

        Return True if successful; False otherwise.
        """
        linkmatrix = self.linkmatrix
        if self.selection_method == 0 and self.root:
            selected, linksaved = state
            linkstruct = np.array(linksaved, dtype=float)
            selected = set(selected)  # type: Set[Tuple[int]]
            if not selected:
                return False
            if linkmatrix.shape[0] != linkstruct.shape[0]:
                return False
            # check that the linkage matrix structure matches. Use isclose for
            # the height column to account for inexact floating point math
            # (e.g. summation order in different ?gemm implementations for
            # euclidean distances, ...)
            if np.any(linkstruct[:, :2] != linkmatrix[:, :2]) or \
                    not np.all(np.isclose(linkstruct[:, 2], linkstruct[:, 2])):
                return False
            selection = []
            indices = np.array([n.value.index for n in leaves(self.root)],
                               dtype=int)
            # mapping from ranges to display (pruned) nodes
            mapping = {node.value.range: node
                       for node in postorder(self._displayed_root)}
            for node in postorder(self.root):  # type: Tree
                r = tuple(indices[node.value.first: node.value.last])
                if r in selected:
                    if node.value.range not in mapping:
                        # the node was pruned from display and cannot be
                        # selected
                        break
                    selection.append(mapping[node.value.range])
                    selected.remove(r)
                if not selected:
                    break  # found all, nothing more to do
            if selection and selected:
                # Could not restore all selected nodes (only partial match)
                return False

            self._set_selected_nodes(selection)
            return True
        return False

    def _set_selected_nodes(self, selection):
        # type: (List[Tree]) -> None
        """
        Set the nodes in `selection` to be the current selected nodes.

        The selection nodes must be subtrees of the current `_displayed_root`.
        """
        self.dendrogram.selectionChanged.disconnect(self._invalidate_output)
        try:
            self.dendrogram.set_selected_clusters(selection)
        finally:
            self.dendrogram.selectionChanged.connect(self._invalidate_output)

    def _invalidate_clustering(self):
        self._update()
        self._update_labels()
        self._invalidate_output()

    def _invalidate_output(self):
        self.commit.deferred()

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

    @gui.deferred
    def commit(self):
        items = getattr(self.matrix, "items", self.items)
        if not items:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
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
            self.Outputs.selected_data.send(None)
            annotated_data = create_annotated_table(items, []) \
                if self.selection_method == 0 and self.matrix.axis else None
            self.Outputs.annotated_data.send(annotated_data)
            return

        selected_data = None

        if isinstance(items, Orange.data.Table) and self.matrix.axis == 1:
            # Select rows
            c = np.zeros(self.matrix.shape[0])

            for i, indices in enumerate(maps):
                c[indices] = i
            c[unselected_indices] = len(maps)

            mask = c != len(maps)

            data, domain = items, items.domain
            attrs = domain.attributes
            classes = domain.class_vars
            metas = domain.metas

            var_name = get_unique_names(domain, "Cluster")
            values = [f"C{i + 1}" for i in range(len(maps))]

            clust_var = Orange.data.DiscreteVariable(
                var_name, values=values + ["Other"])
            domain = Orange.data.Domain(attrs, classes, metas + (clust_var,))
            data = items.transform(domain)
            with data.unlocked(data.metas):
                data.get_column_view(clust_var)[0][:] = c

            if selected_indices:
                selected_data = data[mask]
                clust_var = Orange.data.DiscreteVariable(
                    var_name, values=values)
                selected_data.domain = Domain(
                    attrs, classes, metas + (clust_var, ))

            annotated_data = create_annotated_table(data, selected_indices)

        elif isinstance(items, Orange.data.Table) and self.matrix.axis == 0:
            # Select columns
            attrs = []
            for clust, indices in chain(enumerate(maps, start=1),
                                        [(0, unselected_indices)]):
                for i in indices:
                    attr = items.domain[i].copy()
                    attr.attributes["cluster"] = clust
                    attrs.append(attr)
            domain = Orange.data.Domain(
                # len(unselected_indices) can be 0
                attrs[:len(attrs) - len(unselected_indices)],
                items.domain.class_vars, items.domain.metas)
            selected_data = items.from_table(domain, items)

            domain = Orange.data.Domain(
                attrs,
                items.domain.class_vars, items.domain.metas)
            annotated_data = items.from_table(domain, items)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated_data)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.Resize:
            # NOTE: not using viewport.width(), due to 'transient' scroll bars
            # (macOS). Viewport covers the whole view, but QGraphicsView still
            # scrolls left, right with scroll bar extent (other
            # QAbstractScrollArea widgets behave as expected).
            w_frame = self.view.frameWidth()
            margin = self.view.viewportMargins()
            w_scroll = self.view.verticalScrollBar().width()
            width = (self.view.width() - w_frame * 2 -
                     margin.left() - margin.right() - w_scroll)
            # layout with new width constraint
            self.__layout_main_graphics(width=width)
        elif obj is self._main_graphics and \
                event.type() == QEvent.LayoutRequest:
            # layout preserving the width (vertical re layout)
            self.__layout_main_graphics()
        return super().eventFilter(obj, event)

    @Slot(QPointF)
    def _activate_cut_line(self, pos: QPointF):
        """Activate cut line selection an set cut value to `pos.x()`."""
        self.selection_method = 1
        self.cut_line.setValue(pos.x())
        self._selection_method_changed()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._clear_plot()
        self.dendrogram.clear()
        self.dendrogram.deleteLater()

    def _dendrogram_geom_changed(self):
        pos = self.dendrogram.pos_at_height(self.cutoff_height)
        geom = self.dendrogram.geometry()
        self._set_slider_value(pos.x(), geom.width())

        self.cut_line.setLength(
            self.bottom_axis.geometry().bottom()
            - self.top_axis.geometry().top()
        )

        geom = self._main_graphics.geometry()
        assert geom.topLeft() == QPointF(0, 0)

        def adjustLeft(rect):
            rect = QRectF(rect)
            rect.setLeft(geom.left())
            return rect
        margin = 3
        self.scene.setSceneRect(geom)
        self.view.setSceneRect(geom)
        self.view.setHeaderSceneRect(
            adjustLeft(self.top_axis.geometry()).adjusted(0, 0, 0, margin)
        )
        self.view.setFooterSceneRect(
            adjustLeft(self.bottom_axis.geometry()).adjusted(0, -margin, 0, 0)
        )

    def _dendrogram_slider_changed(self, value):
        p = QPointF(value, 0)
        cl_height = self.dendrogram.height_at(p)

        self.set_cutoff_height(cl_height)

    def _set_slider_value(self, value, span):
        with blocked(self.cut_line):
            self.cut_line.setRange(0, span)
            self.cut_line.setValue(value)

    def set_cutoff_height(self, height):
        self.cutoff_height = height
        if self.root:
            self.cut_ratio = 100 * height / self.root.value.height
        self.select_max_height(height)

    def _set_cut_line_visible(self, visible):
        self.cut_line.setVisible(visible)

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

    def _save_selection(self):
        # Save the current manual node selection state
        selection_state = None
        if self.selection_method == 0 and self.root:
            assert self.linkmatrix is not None
            linkmat = [(int(_0), int(_1), _2)
                       for _0, _1, _2 in self.linkmatrix[:, :3].tolist()]
            nodes_ = self.dendrogram.selected_nodes()
            # match the display (pruned) nodes back (by ranges)
            mapping = {node.value.range: node for node in postorder(self.root)}
            nodes = [mapping[node.value.range] for node in nodes_]
            indices = [tuple(node.value.index for node in leaves(node))
                       for node in nodes]
            if nodes:
                selection_state = (indices, linkmat)
        return selection_state

    def save_state(self):
        # type: () -> Dict[str, Any]
        """
        Save state for `set_restore_state`
        """
        selection = self._save_selection()
        res = {"version": (0, 0, 0)}
        if selection is not None:
            res["selection_state"] = selection
        return res

    def set_restore_state(self, state):
        # type: (Dict[str, Any]) -> bool
        """
        Restore session data from a saved state.

        Parameters
        ----------
        state : Dict[str, Any]

        NOTE
        ----
        This is method called while the instance (self) is being constructed,
        even before its `__init__` is called. Consider `self` to be only a
        `QObject` at this stage.
        """
        if "selection_state" in state:
            selection = state["selection_state"]
            self.__pending_selection_restore = selection
        return True

    def __zoom_in(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor + 1)
        self.__update_font_scale()

    def __zoom_out(self):
        def clip(minval, maxval, val):
            return min(max(val, minval), maxval)
        self.zoom_factor = clip(self.zoom_slider.minimum(),
                                self.zoom_slider.maximum(),
                                self.zoom_factor - 1)
        self.__update_font_scale()

    def __zoom_reset(self):
        self.zoom_factor = 0
        self.__update_font_scale()

    def __layout_main_graphics(self, width=-1):
        if width < 0:
            # Preserve current width.
            width = self._main_graphics.size().width()
        preferred = self._main_graphics.effectiveSizeHint(
            Qt.PreferredSize, constraint=QSizeF(width, -1))
        self._main_graphics.resize(QSizeF(width, preferred.height()))
        mw = self._main_graphics.minimumWidth() + 4
        self.view.setMinimumWidth(mw + self.view.verticalScrollBar().width())

    def __update_font_scale(self):
        font = self.scene.font()
        factor = (1.25 ** self.zoom_factor)
        font = qfont_scaled(font, factor)
        self._main_graphics.setFont(font)

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
        ))
        self.report_plot()


def qfont_scaled(font, factor):
    scaled = QFont(font)
    if font.pointSizeF() != -1:
        scaled.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() != -1:
        scaled.setPixelSize(int(font.pixelSize() * factor))
    return scaled


class AxisItem(pg.AxisItem):
    mousePressed = Signal(QPointF, Qt.MouseButton)
    mouseMoved = Signal(QPointF, Qt.MouseButtons)
    mouseReleased = Signal(QPointF, Qt.MouseButton)

    #: \reimp
    def wheelEvent(self, event):
        event.ignore()  # ignore event to propagate to the view -> scroll

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mousePressed.emit(event.pos(), event.button())
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.pos(), event.buttons())
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.mouseReleased.emit(event.pos(), event.button())
        super().mouseReleaseEvent(event)
        event.accept()


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
        self._line = QLineF()  # type: Optional[QLineF]
        self._pen = QPen()
        super().__init__(parent, **kwargs)

        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setPen(make_pen(brush=QColor(50, 50, 50), width=1, cosmetic=False,
                             style=Qt.DashLine))

        if self._orientation == Qt.Vertical:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeHorCursor)

    def setPen(self, pen: Union[QPen, Qt.GlobalColor, Qt.PenStyle]) -> None:
        pen = QPen(pen)
        if self._pen != pen:
            self.prepareGeometryChange()
            self._pen = pen
            self._line = None
            self.update()

    def pen(self) -> QPen:
        return QPen(self._pen)

    def setValue(self, value: float):
        value = min(max(value, self._min), self._max)

        if self._value != value:
            self.prepareGeometryChange()
            self._value = value
            self._line = None
            self.valueChanged.emit(value)

    def value(self) -> float:
        return self._value

    def setRange(self, minval: float, maxval: float) -> None:
        maxval = max(minval, maxval)
        if minval != self._min or maxval != self._max:
            self._min = minval
            self._max = maxval
            self.rangeChanged.emit(minval, maxval)
            self.setValue(self._value)

    def setLength(self, length: float):
        if self._length != length:
            self.prepareGeometryChange()
            self._length = length
            self._line = None

    def length(self) -> float:
        return self._length

    def setOrientation(self, orientation: Qt.Orientation):
        if self._orientation != orientation:
            self.prepareGeometryChange()
            self._orientation = orientation
            self._line = None
            if self._orientation == Qt.Vertical:
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        event.accept()
        self.linePressed.emit()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._orientation == Qt.Vertical:
            self.setValue(pos.y())
        else:
            self.setValue(pos.x())
        self.lineMoved.emit()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._orientation == Qt.Vertical:
            self.setValue(event.pos().y())
        else:
            self.setValue(event.pos().x())
        self.lineReleased.emit()
        event.accept()

    def boundingRect(self) -> QRectF:
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


if __name__ == "__main__":  # pragma: no cover
    from Orange import distance
    data = Orange.data.Table("iris")
    matrix = distance.Euclidean(distance._preprocess(data))
    WidgetPreview(OWHierarchicalClustering).run(matrix)
