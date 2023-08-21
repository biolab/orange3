from itertools import chain
from contextlib import contextmanager

import typing
from typing import Any, List, Tuple, Dict, Optional, Set, Union

import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsScene, QGridLayout, QSizePolicy,
    QAction, QComboBox, QGraphicsGridLayout, QGraphicsSceneMouseEvent, QLabel
)
from AnyQt.QtGui import (QPen, QFont, QKeySequence, QPainterPath, QColor,
    QFontMetrics)
from AnyQt.QtCore import (
    Qt, QObject, QSize, QPointF, QRectF, QLineF, QEvent, QModelIndex
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from Orange.widgets.utils.localization import pl
from orangewidget.utils.itemmodels import PyListModel
from orangewidget.utils.signals import LazyValue

import Orange.data
from Orange.data.domain import filter_visible
from Orange.data import Domain, DiscreteVariable, ContinuousVariable, \
    StringVariable, Table
import Orange.misc
from Orange.clustering.hierarchical import \
    postorder, preorder, Tree, tree_from_linkage, dist_matrix_linkage, \
    leaves, prune, top_clusters
from Orange.data.util import get_unique_names

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, combobox
from Orange.widgets.utils.annotated_data import (lazy_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME,
                                                 domain_with_annotation_column,
                                                 add_columns,
                                                 create_annotated_table)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import AxisItem
from Orange.widgets.widget import Input, Output, Msg

from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.graphicsview import GraphicsWidgetView
from Orange.widgets.utils.graphicstextlist import TextListView
from Orange.widgets.utils.dendrogram import DendrogramWidget

__all__ = ["OWHierarchicalClustering"]


LINKAGE = ["Single", "Average", "Weighted", "Complete", "Ward"]
LINKAGE_ARGS = ["single", "average", "weighted", "complete", "ward"]
DEFAULT_LINKAGE = "Ward"


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


class SelectedLabelsModel(PyListModel):
    def __init__(self):
        super().__init__([])
        self.subset = set()
        self.__font = QFont()
        self.__colors = None

    def rowCount(self, parent=QModelIndex()):
        count = super().rowCount()
        if self.__colors is not None:
            count = max(count, len(self.__colors))
        return count

    def _emit_data_changed(self):
        self.dataChanged.emit(self.index(0, 0), self.index(len(self) - 1, 0))

    def set_subset(self, subset):
        self.subset = set(subset)
        self._emit_data_changed()

    def set_colors(self, colors):
        self.__colors = colors
        self._emit_data_changed()

    def setFont(self, font):
        self.__font = font
        self._emit_data_changed()

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.FontRole:
            font = QFont(self.__font)
            font.setBold(index.row() in self.subset)
            return font
        if role == Qt.BackgroundRole:
            if self.__colors is not None:
                return self.__colors[index.row()]
            elif not any(self) and self.subset:  # no labels, no color, but subset
                return QColor(0, 0, 0)
        if role == Qt.UserRole and self.subset:
            return index.row() in self.subset

        return super().data(index, role)


class GraphicsView(GraphicsWidgetView, StickyGraphicsView):
    def minimumSizeHint(self) -> QSize:
        msh = super().minimumSizeHint()
        w = self.centralWidget()
        if w is not None:
            width = w.minimumWidth() + 4 + self.verticalScrollBar().width()
            msh.setWidth(max(int(width), msh.width()))
        return msh

    def eventFilter(self, recv: QObject, event: QEvent) -> bool:
        ret = super().eventFilter(recv, event)
        if event.type() == QEvent.LayoutRequest and recv is self.centralWidget():
            self.updateGeometry()
        return ret


class OWHierarchicalClustering(widget.OWWidget):
    name = "Hierarchical Clustering"
    description = "Display a dendrogram of a hierarchical clustering " \
                  "constructed from the input distance matrix."
    icon = "icons/HierarchicalClustering.svg"
    priority = 2100
    keywords = "hierarchical clustering"

    class Inputs:
        distances = Input("Distances", Orange.misc.DistMatrix)
        subset = Input("Data Subset", Orange.data.Table, explicit=True)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    settings_version = 2
    settingsHandler = _DomainContextHandler()

    #: Selected linkage
    linkage = settings.Setting(LINKAGE.index(DEFAULT_LINKAGE))
    #: Index of the selected annotation item (variable, ...)
    annotation = settings.ContextSetting("Enumeration")
    #: Out-of-context setting for the case when the "Name" option is available
    annotation_if_names = settings.Setting("Name")
    #: Out-of-context setting for the case with just "Enumeration" and "None"
    annotation_if_enumerate = settings.Setting("Enumeration")
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
    #: Show labels only for subset (if present)
    label_only_subset = settings.Setting(False)
    #: Color for label decoration
    color_by: Union[DiscreteVariable, ContinuousVariable, None] = \
        settings.ContextSetting(None)

    autocommit = settings.Setting(True)

    graph_name = "scene"  # QGraphicsScene

    basic_annotations = [None, "Enumeration"]

    class Error(widget.OWWidget.Error):
        empty_matrix = Msg("Distance matrix is empty.")
        not_finite_distances = Msg("Some distances are infinite")
        not_symmetric = widget.Msg("Distance matrix is not symmetric.")

    class Warning(widget.OWWidget.Warning):
        subset_on_no_table = \
            Msg("Unused data subset: distances do not refer to data instances")
        subset_not_subset = \
            Msg("Some data from the subset does not appear in distance matrix")
        subset_wrong = \
            Msg("Subset data refers to a different table")
        pruning_disables_colors = \
            Msg("Pruned cluster doesn't show colors and indicate subset")

    #: Stored (manual) selection state (from a saved workflow) to restore.
    __pending_selection_restore = None  # type: Optional[SelectionState]

    def __init__(self):
        super().__init__()

        self.matrix = None
        self.items = None
        self.subset = None
        self.subset_rows = set()
        self.linkmatrix = None
        self.root = None
        self._displayed_root = None
        self.cutoff_height = 0.0

        spin_width = QFontMetrics(self.font()).horizontalAdvance("M" * 7)
        gui.comboBox(
            self.controlArea, self, "linkage", items=LINKAGE, box="Linkage",
            callback=self._invalidate_clustering)

        model = itemmodels.VariableListModel(placeholder="None")
        model[:] = self.basic_annotations

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, "Annotations", orientation=grid)
        self.label_cb = cb = combobox.ComboBoxSearch(
            minimumContentsLength=14,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        cb.setModel(model)
        cb.setCurrentIndex(cb.findData(self.annotation, Qt.EditRole))

        def on_annotation_activated():
            self.annotation = self.label_cb.currentData(Qt.EditRole)
            self._update_labels()
        cb.activated.connect(on_annotation_activated)

        def on_annotation_changed(value):
            self.label_cb.setCurrentIndex(
                self.label_cb.findData(value, Qt.EditRole))
        self.connect_control("annotation", on_annotation_changed)

        grid.addWidget(self.label_cb, 0, 0, 1, 2)

        cb = gui.checkBox(
            None, self, "label_only_subset", "Show labels only for subset",
            disabled=True,
            callback=self._update_labels, stateWhenDisabled=False)
        grid.addWidget(cb, 1, 0, 1, 2)

        model = itemmodels.DomainModel(
            valid_types=(DiscreteVariable, ContinuousVariable),
            placeholder="None")
        cb = gui.comboBox(
            None, self, "color_by", orientation=Qt.Horizontal,
            model=model, callback=self._update_labels,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed))
        self.color_by_label = QLabel("Color by:")
        grid.addWidget(self.color_by_label, 2, 0)
        grid.addWidget(cb, 2, 1)

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
            controlWidth=spin_width, alignment=Qt.AlignRight,
            callback=self._max_depth_changed,
            keyboardTracking=False, addToLayout=False
        )
        self.max_depth_spin.lineEdit().returnPressed.connect(
            self._max_depth_return)

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
            controlWidth=spin_width, alignment = Qt.AlignRight,
            spinType=float, callback=self._cut_ratio_changed,
            addToLayout=False
        )
        self.cut_ratio_spin.setSuffix(" %")
        self.cut_ratio_spin.lineEdit().returnPressed.connect(
            self._cut_ratio_return)

        grid.addWidget(self.cut_ratio_spin, 1, 1)

        grid.addWidget(
            gui.appendRadioButton(
                self.selection_box, "Top N:", addToLayout=False),
            2, 0
        )
        self.top_n_spin = gui.spin(
            self.selection_box, self, "top_n", 1, 20,
            controlWidth=spin_width, alignment=Qt.AlignRight,
            callback=self._top_n_changed, addToLayout=False)
        self.top_n_spin.lineEdit().returnPressed.connect(self._top_n_return)
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
        self.view = GraphicsView(
            self.scene,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            alignment=Qt.AlignLeft | Qt.AlignVCenter,
            widgetResizable=True,
        )
        # Disable conflicting action shortcuts. We define our own.
        for a in self.view.viewActions():
            a.setEnabled(False)
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

        self._main_graphics = QGraphicsWidget(
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        )
        scenelayout = QGraphicsGridLayout()
        scenelayout.setHorizontalSpacing(10)
        scenelayout.setVerticalSpacing(10)

        self._main_graphics.setLayout(scenelayout)
        self.scene.addItem(self._main_graphics)
        self.view.setCentralWidget(self._main_graphics)
        self.scene.addItem(self._main_graphics)

        self.dendrogram = DendrogramWidget(pen_width=2)
        self.dendrogram.setSizePolicy(QSizePolicy.MinimumExpanding,
                                      QSizePolicy.MinimumExpanding)
        self.dendrogram.selectionChanged.connect(self._invalidate_output)
        self.dendrogram.selectionEdited.connect(self._selection_edited)

        self.labels = TextListView()
        self.label_model = SelectedLabelsModel()
        self.labels.setModel(self.label_model)
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
        self.error()
        self.Error.clear()

        self.matrix = None
        self.Error.clear()
        if matrix is not None:
            if len(matrix) < 2:
                self.Error.empty_matrix()
            elif not matrix.is_symmetric():
                self.Error.not_symmetric()
            elif not np.all(np.isfinite(matrix)):
                self.Error.not_finite_distances()
            else:
                self.matrix = matrix

    @Inputs.subset
    def set_subset(self, subset):
        self.subset = subset
        self.controls.label_only_subset.setDisabled(subset is None)

    def handleNewSignals(self):
        if self.__pending_selection_restore is not None:
            selection_state = self.__pending_selection_restore
        else:
            # save the current selection to (possibly) restore later
            selection_state = self._save_selection()

        matrix = self.matrix
        if matrix is not None:
            self._set_items(matrix.row_items, matrix.axis)
        else:
            self._set_items(None)
        self._update()

        # Can now attempt to restore session state from a saved workflow.
        if self.root and selection_state is not None:
            self._restore_selection(selection_state)
            self.__pending_selection_restore = None

        self.Warning.clear()
        rows = set()
        if self.subset:
            subsetids = set(self.subset.ids)
            if not isinstance(self.items, Orange.data.Table) \
                    or not self.matrix.axis:
                self.Warning.subset_on_no_table()
            elif (dataids := set(self.items.ids)) and not subsetids & dataids:
                self.Warning.subset_wrong()
            elif not subsetids <= dataids:
                self.Warning.subset_not_subset()
            else:
                indices = [leaf.value.index for leaf in leaves(self.root)]
                rows = {
                    row for row, rowid in enumerate(self.items.ids[indices])
                    if rowid in subsetids
                }

        self.subset_rows = rows
        self._update_labels()
        self.commit.now()

    def _set_items(self, items, axis=1):
        self.closeContext()
        self.items = items
        model = self.label_cb.model()
        color_model = self.controls.color_by.model()
        color_model.set_domain(None)
        self.color_by = None
        if len(model) == 2 and model[0] is None:
            if model[1] == "Name":
                self.annotation_if_names = self.annotation
            if model[1] == self.basic_annotations[1]:
                self.annotation_if_enumerate = self.annotation
        if isinstance(items, Orange.data.Table) and axis:
            metas_class = tuple(
                filter_visible(chain(items.domain.metas,
                                     items.domain.class_vars)))
            visible_attrs = tuple(filter_visible(items.domain.attributes))
            if not (metas_class or visible_attrs):
                model[:] = self.basic_annotations
            else:
                model[:] = (
                    (None, )
                    + metas_class
                    + (model.Separator, ) * bool(metas_class and visible_attrs)
                    + visible_attrs)
            for meta in items.domain.metas:
                if isinstance(meta, StringVariable):
                    self.annotation = meta
                    break
            else:
                if items.domain.class_vars:
                    # No string metas: show class
                    self.annotation = items.domain.class_vars[0]
                else:
                    # No string metas and no class: show the first option
                    # which is not None (in the worst case, Enumeration)
                    self.annotation = model[1]

            color_model.set_domain(items.domain)
            if items.domain.class_vars:
                self.color_by = items.domain.class_vars[0]
            self.openContext(items.domain)
        elif isinstance(items, Orange.data.Table) and not axis \
                or (isinstance(items, list) and \
                    all(isinstance(var, Orange.data.Variable)
                        for var in items)):
            model[:] = (None, "Name")
            self.annotation = self.annotation_if_names
        else:
            model[:] = self.basic_annotations
            self.annotation = self.annotation_if_enumerate

        no_colors = len(color_model) == 1
        self.controls.color_by.setDisabled(no_colors)
        self.color_by_label.setDisabled(no_colors)

    def _clear_plot(self):
        self.dendrogram.set_root(None)
        self.label_model.clear()

    def _set_displayed_root(self, root):
        self._clear_plot()
        self._displayed_root = root
        self.dendrogram.set_root(root)
        self._update_labels()

    def _update(self):
        self._clear_plot()

        distances = self.matrix

        if distances is not None:
            method = LINKAGE_ARGS[self.linkage]
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
        if not hasattr(self, "label_model"):
            # This method can be called during widget initialization when
            # creating check box for label_only_subset, if it's value is
            # initially True.
            # See https://github.com/biolab/orange-widget-base/pull/213;
            # if it's merged, this check can be removed.
            return

        self.Warning.pruning_disables_colors(
            shown=self.pruning
                  and (self.subset_rows or self.color_by is not None))
        labels = []
        if self.root and self._displayed_root:
            indices = [leaf.value.index for leaf in leaves(self.root)]

            if self.annotation is None:
                if not self.pruning \
                        and self.subset_rows and self.color_by is None:
                    # Model fails if number of labels and of colors mismatch
                    labels = [""] * len(indices)
                else:
                    labels = []
            elif self.annotation == "Enumeration":
                labels = [str(i+1) for i in indices]
            elif self.annotation == "Name":
                attr = self.matrix.row_items.domain.attributes
                labels = [str(attr[i]) for i in indices]
            elif isinstance(self.annotation, Orange.data.Variable):
                col_data = self.items.get_column(self.annotation)
                labels = [self.annotation.str_val(val) for val in col_data]
                labels = [labels[idx] for idx in indices]
            else:
                labels = []
            if not self.pruning and \
                    labels and self.label_only_subset and self.subset_rows:
                labels = [label if row in self.subset_rows else ""
                          for row, label in enumerate(labels)]

            if labels and self._displayed_root is not self.root:
                joined = leaves(self._displayed_root)
                labels = [", ".join(labels[leaf.value.first: leaf.value.last])
                          for leaf in joined]

        self.label_model[:] = labels
        self.label_model.set_subset(set() if self.pruning else self.subset_rows)
        self.labels.setMinimumWidth(1 if labels else -1)

        if not self.pruning and self.color_by is not None:
            col = self.items.get_column(self.color_by)
            self.label_model.set_colors(
                self.color_by.palette.values_to_qcolors(col[indices]))
        else:
            self.label_model.set_colors(None)

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

    def _max_depth_return(self):
        if self.pruning != 1:
            self.pruning = 1
            self._invalidate_pruning()

    def _max_depth_changed(self):
        self.pruning = 1
        self._invalidate_pruning()

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

        if not selected_indices:
            self.Outputs.selected_data.send(None)
            annotated_data = lazy_annotated_table(items, []) \
                if self.selection_method == 0 and self.matrix.axis else None
            self.Outputs.annotated_data.send(annotated_data)
            return

        selected_data = annotated_data = None

        if isinstance(items, Orange.data.Table) and self.matrix.axis == 1:
            # Select rows
            data, domain = items, items.domain

            c = np.full(self.matrix.shape[0], len(maps))
            for i, indices in enumerate(maps):
                c[indices] = i

            clust_name = get_unique_names(domain, "Cluster")
            values = [f"C{i + 1}" for i in range(len(maps))]

            sel_clust_var = Orange.data.DiscreteVariable(
                name=clust_name, values=values)
            sel_domain = add_columns(domain, metas=(sel_clust_var,))
            selected_data = LazyValue[Table](
                lambda: items.add_column(
                    sel_clust_var, c, to_metas=True)[c != len(maps)],
                domain=sel_domain, length=len(selected_indices))

            ann_clust_var = Orange.data.DiscreteVariable(
                name=clust_name, values=values + ["Other"]
            )
            ann_domain = add_columns(
                domain_with_annotation_column(data)[0], metas=(ann_clust_var, ))
            annotated_data = LazyValue[Table](
                lambda: create_annotated_table(
                    data=items.add_column(ann_clust_var, c, to_metas=True),
                    selected_indices=selected_indices),
                domain=ann_domain, length=len(items)
            )

        elif isinstance(items, Orange.data.Table) and self.matrix.axis == 0:
            # Select columns
            attrs = []
            unselected_indices = sorted(set(range(self.root.value.last)) -
                                        set(selected_indices))
            for clust, indices in chain(enumerate(maps, start=1),
                                        [(0, unselected_indices)]):
                for i in indices:
                    attr = items.domain[i].copy()
                    attr.attributes["cluster"] = clust
                    attrs.append(attr)
            all_domain = Orange.data.Domain(
                # len(unselected_indices) can be 0
                attrs[:len(attrs) - len(unselected_indices)],
                items.domain.class_vars, items.domain.metas)

            selected_data = LazyValue[Table](
                lambda: items.from_table(all_domain, items),
                domain=all_domain, length=len(items))

            sel_domain = Orange.data.Domain(
                attrs,
                items.domain.class_vars, items.domain.metas)
            annotated_data = LazyValue[Table](
                lambda: items.from_table(sel_domain, items),
                domain=sel_domain, length=len(items))

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated_data)

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

    def _cut_ratio_changed(self):
        self.selection_method = 1
        self._selection_method_changed()

    def _cut_ratio_return(self):
        if self.selection_method != 1:
            self.selection_method = 1
            self._selection_method_changed()

    def _top_n_changed(self):
        self.selection_method = 2
        self._selection_method_changed()

    def _top_n_return(self):
        if self.selection_method != 2:
            self.selection_method = 2
            self._selection_method_changed()

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

    def __update_font_scale(self):
        font = self.scene.font()
        factor = (1.25 ** self.zoom_factor)
        font = qfont_scaled(font, factor)
        self._main_graphics.setFont(font)
        self.label_model.setFont(font)

    def send_report(self):
        annot = self.label_cb.currentText()
        if isinstance(self.annotation, str):
            annot = annot.lower()
        if self.selection_method == 0:
            sel = "manual"
        elif self.selection_method == 1:
            sel = "at {:.1f} of height".format(self.cut_ratio)
        else:
            sel = f"top {self.top_n} {pl(self.top_n, 'cluster')}"
        self.report_items((
            ("Linkage", LINKAGE[self.linkage]),
            ("Annotation", annot),
            ("Pruning",
             self.pruning != 0 and "{} levels".format(self.max_depth)),
            ("Selection", sel),
        ))
        self.report_plot()

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            if context.values["annotation"] == "None":
                context.values["annotation"] = None


def qfont_scaled(font, factor):
    scaled = QFont(font)
    if font.pointSizeF() != -1:
        scaled.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() != -1:
        scaled.setPixelSize(int(font.pixelSize() * factor))
    return scaled


class AxisItem(AxisItem):
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


class SliderLine(QGraphicsWidget):
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
        self._line: Optional[QLineF] = QLineF()
        self._pen: Optional[QPen] = None
        super().__init__(parent, **kwargs)
        self.setAcceptedMouseButtons(Qt.LeftButton)
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
        if self._pen is None:
            return QPen(self.palette().text(), 1.0, Qt.DashLine)
        else:
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

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

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


def main():
    # pragma: no cover
    from Orange import distance  # pylint: disable=import-outside-toplevel
    data = Orange.data.Table("iris")
    matrix = distance.Euclidean(distance._preprocess(data))
    subset = data[10:30]
    WidgetPreview(OWHierarchicalClustering).run(matrix, set_subset=subset)


if __name__ == "__main__":  # pragma: no cover
    main()
