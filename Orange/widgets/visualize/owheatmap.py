import sys
import math
from collections import defaultdict
from types import SimpleNamespace as namespace

import numpy as np

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QSizePolicy, QGraphicsScene, QGraphicsView, QFontMetrics,
    QPen, QPixmap, QColor
)
from PyQt4.QtCore import Qt, QSize, QPointF, QSizeF, QRectF, QObject, QEvent
from PyQt4.QtCore import pyqtSignal as Signal
import pyqtgraph as pg

import Orange.data
import Orange.distance

from Orange.clustering import hierarchical
from Orange.widgets.utils import colorbrewer
from Orange.widgets import widget, gui, settings
from Orange.widgets.io import FileFormats

from Orange.widgets.unsupervised.owhierarchicalclustering import \
    DendrogramWidget


def split_domain(domain, split_label):
    """Split the domain based on values of `split_label` value.
    """
    groups = defaultdict(list)
    for attr in domain.attributes:
        groups[attr.attributes.get(split_label)].append(attr)

    attr_values = [attr.attributes.get(split_label)
                   for attr in domain.attributes]

    domains = []
    for value, attrs in groups.items():
        group_domain = Orange.data.Domain(
            attrs, domain.class_vars, domain.metas)

        domains.append((value, group_domain))

    if domains:
        assert(all(len(dom) == len(domains[0][1]) for _, dom in domains))

    return sorted(domains, key=lambda t: attr_values.index(t[0]))


def vstack_by_subdomain(data, sub_domains):
    domain = sub_domains[0]
    newtable = Orange.data.Table(domain)

    for sub_dom in sub_domains:
        sub_data = data.from_table(sub_dom, data)
        # TODO: improve O(N ** 2)
        newtable.extend(sub_data)

    return newtable


def select_by_class(data, class_):
    indices = select_by_class_indices(data, class_)
    return data[indices]


def select_by_class_indices(data, class_):
    col, _ = data.get_column_view(data.domain.class_var)
    return col == class_


def group_by_unordered(iterable, key):
    groups = defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)
    return groups.items()


def candidate_split_labels(data):
    """
    Return candidate labels on which we can split the data.
    """
    groups = defaultdict(list)
    for attr in data.domain.attributes:
        for item in attr.attributes.items():
            groups[item].append(attr)

    by_keys = defaultdict(list)
    for (key, _), attrs in groups.items():
        by_keys[key].append(attrs)

    # Find the keys for which all values have the same number
    # of attributes.
    candidates = []
    for key, groups in by_keys.items():
        count = len(groups[0])
        if all(len(attrs) == count for attrs in groups) and \
                len(groups) > 1 and count > 1:
            candidates.append(key)

    return candidates


def leaf_indices(tree):
    return [leaf.value.index for leaf in hierarchical.leaves(tree)]


def palette_gradient(colors, discrete=False):
    n = len(colors)
    stops = np.linspace(0.0, 1.0, n, endpoint=True)
    gradstops = [(float(stop), color) for stop, color in zip(stops, colors)]
    grad = QtGui.QLinearGradient(QPointF(0, 0), QPointF(1, 0))
    grad.setStops(gradstops)
    return grad


def palette_pixmap(colors, size):
    img = QPixmap(size)
    img.fill(Qt.transparent)

    grad = palette_gradient(colors)
    grad.setCoordinateMode(QtGui.QLinearGradient.ObjectBoundingMode)

    painter = QtGui.QPainter(img)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QtGui.QBrush(grad))
    painter.drawRect(0, 0, size.width(), size.height())
    painter.end()
    return img


def color_palette_model(palettes, iconsize=QSize(64, 16)):
    model = QtGui.QStandardItemModel()
    for name, palette in palettes:
        _, colors = max(palette.items())
        colors = [QColor(*c) for c in colors]
        item = QtGui.QStandardItem(name)
        item.setIcon(QtGui.QIcon(palette_pixmap(colors, iconsize)))
        item.setData(palette, Qt.UserRole)
        model.appendRow([item])
    return model


def color_palette_table(colors, samples=255,
                        threshold_low=0.0, threshold_high=1.0):
    N = len(colors)
    colors = np.array(colors, dtype=np.ubyte)
    low, high = threshold_low * 255, threshold_high * 255
    points = np.linspace(low, high, N)
    space = np.linspace(0, 255, 255)

    r = np.interp(space, points, colors[:, 0], left=255, right=0)
    g = np.interp(space, points, colors[:, 1], left=255, right=0)
    b = np.interp(space, points, colors[:, 2], left=255, right=0)
    return np.c_[r, g, b]

# TODO:
#     * Richer Tool Tips
#     * Color map edit/manage
#     * 'Gamma' color transform (nonlinear exponential interpolation)
#     * Restore saved row selection (?)
#     * 'namespace' use cleanup


class OWHeatMap(widget.OWWidget):
    name = "Heat Map"
    description = "Heatmap visualization."
    icon = "icons/Heatmap.svg"
    priority = 1040

    inputs = [("Data", Orange.data.Table, "set_dataset")]
    outputs = [("Selected Data", Orange.data.Table, widget.Default)]

    settingsHandler = settings.DomainContextHandler()

    NoSorting, Clustering, OrderedClustering = 0, 1, 2
    NoPosition, PositionTop, PositionBottom = 0, 1, 2

    gamma = settings.Setting(0)
    threshold_low = settings.Setting(0.0)
    threshold_high = settings.Setting(1.0)
    # Type of sorting to apply on rows
    sort_rows = settings.Setting(NoSorting)
    # Type of sorting to apply on columns
    sort_columns = settings.Setting(NoSorting)
    # Display stripe with averages
    averages = settings.Setting(True)
    # Display legend
    legend = settings.Setting(True)
    # Annotations
    annotation_index = settings.ContextSetting(0)
    # Stored color palette settings
    color_settings = settings.Setting(None)
    user_palettes = settings.Setting([])
    palette_index = settings.Setting(0)
    column_label_pos = settings.Setting(PositionTop)

    auto_commit = settings.Setting(True)

    want_graph = True

    def __init__(self, parent=None):
        super().__init__(self, parent)

        # set default settings
        self.SpaceX = 10
        self.ShowAnnotation = 0

        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.palette = None
        self.keep_aspect = False
        #: The data striped of discrete features
        self.data = None
        #: The original data with all features (retained to
        #: preserve the domain on the output)
        self.input_data = None

        self.annotation_vars = ['(None)']
        self.__rows_cache = {}
        self.__columns_cache = {}

        # GUI definition
        colorbox = gui.widgetBox(self.controlArea, "Color")
        self.color_cb = gui.comboBox(colorbox, self, "palette_index")
        self.color_cb.setIconSize(QSize(64, 16))
        palettes = sorted(colorbrewer.colorSchemes["sequential"].items())
        palettes += [("Green-Black-Red",
                      {3: [(0, 255, 0), (0, 0, 0), (255, 0, 0)]})]
        palettes += self.user_palettes
        model = color_palette_model(palettes, self.color_cb.iconSize())
        model.setParent(self)
        self.color_cb.setModel(model)
        self.color_cb.activated.connect(self.update_color_schema)
        self.color_cb.setCurrentIndex(self.palette_index)
        # TODO: Add 'Manage/Add/Remove' action.

        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow
        )

        lowslider = gui.hSlider(
            colorbox, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)
        highslider = gui.hSlider(
            colorbox, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)

        form.addRow("Low:", lowslider)
        form.addRow("High:", highslider)
        colorbox.layout().addLayout(form)

        sortbox = gui.widgetBox(self.controlArea, "Sorting")
        # For attributes
        gui.comboBox(sortbox, self, "sort_columns",
                     items=["No sorting",
                            "Clustering",
                            "Clustering with leaf ordering"],
                     label='Columns',
                     callback=self.update_sorting_attributes)

        # For examples
        gui.comboBox(sortbox, self, "sort_rows",
                     items=["No sorting",
                            "Clustering",
                            "Clustering with leaf ordering"],
                     label='Rows',
                     callback=self.update_sorting_examples)

        box = gui.widgetBox(self.controlArea, 'Annotation && Legends')

        gui.checkBox(box, self, 'legend', 'Show legend',
                     callback=self.update_legend)

        gui.checkBox(box, self, 'averages', 'Stripes with averages',
                     callback=self.update_averages_stripe)

        annotbox = gui.widgetBox(box, "Row Annotations")
        annotbox.setFlat(True)
        self.annotations_cb = gui.comboBox(annotbox, self, "annotation_index",
                                           items=self.annotation_vars,
                                           callback=self.update_annotations)

        posbox = gui.widgetBox(box, "Column Labels Position")
        posbox.setFlat(True)

        gui.comboBox(
            posbox, self, "column_label_pos",
            items=["None", "Top", "Bottom", "Top and Bottom"],
            callback=self.update_column_annotations)

        gui.checkBox(self.controlArea, self, "keep_aspect",
                     "Keep aspect ratio", box="Resize",
                     callback=self.__aspect_mode_changed)

        splitbox = gui.widgetBox(self.controlArea, "Split By")
        self.split_lb = QtGui.QListWidget()
        self.split_lb.itemSelectionChanged.connect(self.update_heatmaps)
        splitbox.layout().addWidget(self.split_lb)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_commit", "Commit")

        # Scene with heatmap
        self.heatmap_scene = self.scene = HeatmapScene(parent=self)
        self.selection_manager = HeatmapSelectionManager(self)
        self.selection_manager.selection_changed.connect(
            self.__update_selection_geometry)
        self.selection_manager.selection_finished.connect(
            self.on_selection_finished)
        self.heatmap_scene.set_selection_manager(self.selection_manager)

        item = QtGui.QGraphicsRectItem(0, 0, 10, 10, None, self.heatmap_scene)
        self.heatmap_scene.itemsBoundingRect()
        self.heatmap_scene.removeItem(item)

        policy = (Qt.ScrollBarAlwaysOn if self.keep_aspect
                  else Qt.ScrollBarAlwaysOff)
        self.sceneView = QGraphicsView(
            self.scene,
            verticalScrollBarPolicy=policy,
            horizontalScrollBarPolicy=policy)

        self.sceneView.viewport().installEventFilter(self)

        self.mainArea.layout().addWidget(self.sceneView)
        self.heatmap_scene.widget = None

        self.heatmap_widget_grid = [[]]
        self.attr_annotation_widgets = []
        self.attr_dendrogram_widgets = []
        self.gene_annotation_widgets = []
        self.gene_dendrogram_widgets = []

        self.selection_rects = []
        self.selected_rows = []
        self.graphButton.clicked.connect(self.save_graph)

    def sizeHint(self):
        return QSize(800, 400)

    def color_palette(self):
        data = self.color_cb.itemData(self.palette_index, role=Qt.UserRole)
        if data is None:
            return []
        else:
            _, colors = max(data.items())
            return color_palette_table(
                colors, threshold_low=self.threshold_low,
                threshold_high=self.threshold_high)

    def selected_split_label(self):
        """Return the current selected split label."""
        item = self.split_lb.currentItem()
        return str(item.text()) if item else None

    def clear(self):
        self.data = None
        self.input_data = None
        self.annotations_cb.clear()
        self.annotations_cb.addItem('(None)')
        self.split_lb.clear()
        self.annotation_vars = ['(None)']
        self.clear_scene()
        self.selected_rows = []
        self.__columns_cache.clear()
        self.__rows_cache.clear()

    def clear_scene(self):
        self.selection_manager.set_heatmap_widgets([[]])
        self.heatmap_scene.clear()
        self.heatmap_scene.widget = None
        self.heatmap_widget_grid = [[]]
        self.col_annotation_widgets = []
        self.col_annotation_widgets_bottom = []
        self.col_annotation_widgets_top = []
        self.row_annotation_widgets = []
        self.col_dendrograms = []
        self.row_dendrograms = []
        self.selection_rects = []

    def set_dataset(self, data=None):
        """Set the input dataset to display."""
        self.closeContext()
        self.clear()

        self.error(0)
        self.warning(0)
        input_data = data
        if data is not None and \
                any(var.is_discrete for var in data.domain.attributes):
            ndisc = sum(var.is_discrete for var in data.domain.attributes)
            data = data.from_table(
                Orange.data.Domain([var for var in data.domain.attributes
                                    if var.is_continuous],
                                   data.domain.class_vars,
                                   data.domain.metas),
                data)
            if not data.domain.attributes:
                self.error(0, "No continuous feature columns")
                input_data = data = None
            else:
                self.warning(0, "{} discrete column{} removed"
                                .format(ndisc, "s" if ndisc > 1 else ""))

        self.data = data
        self.input_data = input_data
        if data is not None:
            variables = self.data.domain.class_vars + self.data.domain.metas
            variables = [var for var in variables
                         if isinstance(var, (Orange.data.DiscreteVariable,
                                             Orange.data.StringVariable))]
            self.annotation_vars.extend(variables)

            for var in variables:
                self.annotations_cb.addItem(*gui.attributeItem(var))

            self.split_lb.addItems(candidate_split_labels(data))

            self.openContext(self.data)
            if self.annotation_index >= len(self.annotation_vars):
                self.annotation_index = 0

            self.update_heatmaps()

        self.commit()

    def update_heatmaps(self):
        if self.data is not None:
            self.clear_scene()
            self.construct_heatmaps(self.data, self.selected_split_label())
            self.construct_heatmaps_scene(self.heatmapparts)
        else:
            self.clear()

    def _make(self, data, group_var=None, group_key=None):
        if group_var is not None:
            assert group_var.is_discrete
            _col_data, _ = data.get_column_view(group_var)
            row_groups = [np.flatnonzero(_col_data == i)
                          for i in range(len(group_var.values))]
            row_indices = [np.flatnonzero(_col_data == i)
                           for i in range(len(group_var.values))]
            row_groups = [namespace(title=name, indices=ind, cluster=None,
                                    cluster_ord=None)
                          for name, ind in zip(group_var.values, row_indices)]
        else:
            row_groups = [namespace(title=None, indices=slice(0, -1),
                                    cluster=None, cluster_ord=None)]

        if group_key is not None:
            col_groups = split_domain(data.domain, group_key)
            assert len(col_groups) > 0
            col_indices = [np.array([data.domain.index(var) for var in group])
                           for _, group in col_groups]
            col_groups = [namespace(title=name, domain=d, indices=ind,
                                    cluster=None, cluster_ord=None)
                          for (name, d), ind in zip(col_groups, col_indices)]
        else:
            col_groups = [
                namespace(
                    title=None, indices=slice(0, len(data.domain.attributes)),
                    domain=data.domain, cluster=None, cluster_ord=None)
            ]

        minv, maxv = np.nanmin(data.X), np.nanmax(data.X)

        parts = namespace(
            rows=row_groups, columns=col_groups,
            levels=(minv, maxv),
        )
        return parts

    def cluster_rows(self, data, parts, ordered=False):
        row_groups = []
        for row in parts.rows:
            if row.cluster is not None:
                cluster = row.cluster
            else:
                cluster = None
            if row.cluster_ord is not None:
                cluster_ord = row.cluster_ord
            else:
                cluster_ord = None

            need_dist = cluster is None or (ordered and cluster_ord is None)
            if need_dist:
                subset = data[row.indices]
                subset = Orange.distance._preprocess(subset)
                matrix = Orange.distance.Euclidean(subset)

            if cluster is None:
                cluster = hierarchical.dist_matrix_clustering(matrix)

            if ordered and cluster_ord is None:
                cluster_ord = hierarchical.optimal_leaf_ordering(cluster, matrix)

            row_groups.append(namespace(title=row.title, indices=row.indices,
                                        cluster=cluster, cluster_ord=cluster_ord))

        return namespace(columns=parts.columns, rows=row_groups,
                         levels=parts.levels)

    def cluster_columns(self, data, parts, ordered=False):
        if len(parts.columns) > 1:
            data = vstack_by_subdomain(data, [col.domain for col in parts.columns])
        assert all(var.is_continuous for var in data.domain.attributes)

        col0 = parts.columns[0]
        if col0.cluster is not None:
            cluster = col0.cluster
        else:
            cluster = None
        if col0.cluster_ord is not None:
            cluster_ord = col0.cluster_ord
        else:
            cluster_ord = None
        need_dist = cluster is None or (ordered and cluster_ord is None)

        if need_dist:
            data = Orange.distance._preprocess(data)
            matrix = Orange.distance.PearsonR(data, axis=0)

        if cluster is None:
            cluster = hierarchical.dist_matrix_clustering(matrix)
        if ordered and cluster_ord is None:
            cluster_ord = hierarchical.optimal_leaf_ordering(cluster, matrix)

        col_groups = [namespace(title=col.title, indices=col.indices,
                                cluster=cluster, cluster_ord=cluster_ord,
                                domain=col.domain)
                      for col in parts.columns]
        return namespace(columns=col_groups,  rows=parts.rows,
                         levels=parts.levels)

    def construct_heatmaps(self, data, split_label=None):

        if split_label is not None:
            groups = split_domain(data.domain, split_label)
            assert len(groups) > 0
        else:
            groups = [("", data.domain)]

        if data.domain.has_discrete_class:
            group_var = data.domain.class_var
        else:
            group_var = None

        self.progressBarInit()

        group_label = split_label

        parts = self._make(data, group_var, group_label)
        # Restore/update the row/columns items descriptions from cache if
        # available
        if group_var in self.__rows_cache:
            parts.rows = self.__rows_cache[group_var].rows
        if group_label in self.__columns_cache:
            parts.columns = self.__columns_cache[group_label].columns

        if self.sort_rows != OWHeatMap.NoSorting:
            parts = self.cluster_rows(
                self.data, parts,
                ordered=self.sort_rows == OWHeatMap.OrderedClustering)

        if self.sort_columns != OWHeatMap.NoSorting:
            parts = self.cluster_columns(
                self.data, parts,
                ordered=self.sort_columns == OWHeatMap.OrderedClustering)

        # Cache the updated parts
        self.__rows_cache[group_var] = parts
        self.__columns_cache[group_label] = parts

        self.heatmapparts = parts
        self.progressBarFinished()

    def construct_heatmaps_scene(self, parts):
        def select_row(item):
            if self.sort_rows == OWHeatMap.NoSorting:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=None)
            elif self.sort_rows == OWHeatMap.Clustering:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=item.cluster)
            elif self.sort_rows == OWHeatMap.OrderedClustering:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=item.cluster_ord)

        def select_col(item):
            if self.sort_columns == OWHeatMap.NoSorting:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=None, domain=item.domain)
            elif self.sort_columns == OWHeatMap.Clustering:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=item.cluster, domain=item.domain)
            elif self.sort_columns == OWHeatMap.OrderedClustering:
                return namespace(title=item.title, indices=item.indices,
                                 cluster=item.cluster_ord, domain=item.domain)

        rows = [select_row(rowitem) for rowitem in parts.rows]
        cols = [select_col(colitem) for colitem in parts.columns]
        parts = namespace(columns=cols, rows=rows, levels=parts.levels)

        self.setup_scene(parts)

    def setup_scene(self, parts):
        # parts = * a list of row descriptors (title, indices, cluster,)
        #         * a list of col descriptors (title, indices, cluster, domain)

        self.heatmap_scene.clear()
        # The top level container widget
        widget = GraphicsWidget()
        widget.layoutDidActivate.connect(self.__update_selection_geometry)

        grid = QtGui.QGraphicsGridLayout()
        grid.setSpacing(self.SpaceX)
        self.heatmap_scene.addItem(widget)

        N, M = len(parts.rows), len(parts.columns)

        # Start row/column where the heatmap items are inserted
        # (after the titles/legends/dendrograms)
        Row0 = 3
        Col0 = 3
        LegendRow = 0
        # The column for the vertical dendrogram
        DendrogramColumn = 0
        # The row for the horizontal dendrograms
        DendrogramRow = 1
        RightLabelColumn = Col0 + M
        TopLabelsRow = 2
        BottomLabelsRow = Row0 + 2 * N

        widget.setLayout(grid)

        palette = self.color_palette()

        sort_i = []
        sort_j = []

        column_dendrograms = [None] * M
        row_dendrograms = [None] * N

        for i, rowitem in enumerate(parts.rows):
            if rowitem.title:
                title = QtGui.QGraphicsSimpleTextItem(rowitem.title, widget)
                item = GraphicsSimpleTextLayoutItem(title, parent=grid)
                grid.addItem(item, Row0 + i * 2, Col0)

            if rowitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=widget,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=True)
                dendrogram.set_root(rowitem.cluster)
                dendrogram.setMaximumWidth(100)
                dendrogram.setMinimumWidth(100)
                # Ignore dendrogram vertical size hint (heatmap's size
                # should define the  row's vertical size).
                dendrogram.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Ignored)
                dendrogram.itemClicked.connect(
                    lambda item, partindex=i:
                        self.__select_by_cluster(item, partindex)
                )

                grid.addItem(dendrogram, Row0 + i * 2 + 1, DendrogramColumn)
                sort_i.append(np.array(leaf_indices(rowitem.cluster)))
                row_dendrograms[i] = dendrogram
            else:
                sort_i.append(None)

        for j, colitem in enumerate(parts.columns):
            if colitem.title:
                title = QtGui.QGraphicsSimpleTextItem(colitem.title, widget)
                item = GraphicsSimpleTextLayoutItem(title, parent=grid)
                grid.addItem(item, 1, Col0 + j)

            if colitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=widget,
                    orientation=DendrogramWidget.Top,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=False)

                dendrogram.set_root(colitem.cluster)
                dendrogram.setMaximumHeight(100)
                dendrogram.setMinimumHeight(100)
                # Ignore dendrogram horizontal size hint (heatmap's width
                # should define the column width).
                dendrogram.setSizePolicy(
                    QSizePolicy.Ignored, QSizePolicy.Expanding)
                grid.addItem(dendrogram, DendrogramRow, Col0 + j)
                sort_j.append(np.array(leaf_indices(colitem.cluster)))
                column_dendrograms[j] = dendrogram
            else:
                sort_j.append(None)

        heatmap_widgets = []
        for i in range(N):
            heatmap_row = []
            for j in range(M):
                row_ix = parts.rows[i].indices
                col_ix = parts.columns[j].indices
                hw = GraphicsHeatmapWidget(parent=widget)
                data = self.data[row_ix, col_ix].X

                if sort_i[i] is not None:
                    data = data[sort_i[i]]
                if sort_j[j] is not None:
                    data = data[:, sort_j[j]]

                hw.set_heatmap_data(data)
                hw.set_levels(parts.levels)
                hw.set_color_table(palette)
                hw.set_show_averages(self.averages)

                grid.addItem(hw, Row0 + i * 2 + 1, Col0 + j)
                grid.setRowStretchFactor(Row0 + i * 2 + 1, data.shape[0] * 100)
                heatmap_row.append(hw)
            heatmap_widgets.append(heatmap_row)

        row_annotation_widgets = []
        col_annotation_widgets = []
        col_annotation_widgets_top = []
        col_annotation_widgets_bottom = []

        for i, rowitem in enumerate(parts.rows):
            if isinstance(rowitem.indices, slice):
                indices = np.array(
                    range(*rowitem.indices.indices(self.data.X.shape[0])))
            else:
                indices = rowitem.indices
            if sort_i[i] is not None:
                indices = indices[sort_i[i]]

            labels = [str(i) for i in indices]

            labelslist = GraphicsSimpleTextList(
                labels, parent=widget, orientation=Qt.Vertical)

            labelslist._indices = indices
            labelslist.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            labelslist.setContentsMargins(0.0, 0.0, 0.0, 0.0)
            labelslist.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

            grid.addItem(labelslist, Row0 + i * 2 + 1, RightLabelColumn)
            grid.setAlignment(labelslist, Qt.AlignLeft)
            row_annotation_widgets.append(labelslist)

        for j, colitem in enumerate(parts.columns):
            # Top attr annotations
            if isinstance(colitem.indices, slice):
                indices = np.array(
                    range(*colitem.indices.indices(self.data.X.shape[1])))
            else:
                indices = colitem.indices
            if sort_j[j] is not None:
                indices = indices[sort_j[j]]

            labels = [self.data.domain[i].name for i in indices]

            labelslist = GraphicsSimpleTextList(
                labels, parent=widget, orientation=Qt.Horizontal)
            labelslist.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
            labelslist._indices = indices

            labelslist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            grid.addItem(labelslist, TopLabelsRow, Col0 + j,
                         Qt.AlignBottom | Qt.AlignLeft)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_top.append(labelslist)

            # Bottom attr annotations
            labelslist = GraphicsSimpleTextList(
                labels, parent=widget, orientation=Qt.Horizontal)
            labelslist.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            labelslist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            grid.addItem(labelslist, BottomLabelsRow, Col0 + j)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_bottom.append(labelslist)

        legend = GradientLegendWidget(
            parts.levels[0], parts.levels[1],
            parent=widget)

        legend.set_color_table(palette)
        legend.setMinimumSize(QSizeF(100, 20))
        legend.setVisible(self.legend)

        grid.addItem(legend, LegendRow, Col0)

        self.heatmap_scene.widget = widget
        self.heatmap_widget_grid = heatmap_widgets
        self.selection_manager.set_heatmap_widgets(heatmap_widgets)

        self.row_annotation_widgets = row_annotation_widgets
        self.col_annotation_widgets = col_annotation_widgets
        self.col_annotation_widgets_top = col_annotation_widgets_top
        self.col_annotation_widgets_bottom = col_annotation_widgets_bottom
        self.col_dendrograms = column_dendrograms
        self.row_dendrograms = row_dendrograms

        self.update_annotations()
        self.update_column_annotations()

        self.__update_size_constraints()

    def __update_size_constraints(self):
        if self.heatmap_scene.widget is not None:
            mode = Qt.KeepAspectRatio if self.keep_aspect \
                   else Qt.IgnoreAspectRatio
            size = QSizeF(self.sceneView.viewport().size())
            widget = self.heatmap_scene.widget
            layout = widget.layout()
            if mode == Qt.IgnoreAspectRatio:
                # Reset the row height constraints ...
                for i, hm_row in enumerate(self.heatmap_widget_grid):
                    layout.setRowMaximumHeight(3 + i * 2 + 1, np.finfo(np.float32).max)
                    layout.setRowPreferredHeight(3 + i * 2 + 1, 0)
                # ... and resize to match the viewport, taking the minimum size
                # into account
                minsize = widget.minimumSize()
                size = size.expandedTo(minsize)
                widget.resize(size)
            else:
                # First set/update the widget's width (the layout will
                # distribute the available width to heatmap widgets in
                # the grid)
                minsize = widget.minimumSize()
                widget.resize(size.expandedTo(minsize).width(),
                              widget.size().height())
                # calculate and set the heatmap row's heights based on
                # the width
                for i, hm_row in enumerate(self.heatmap_widget_grid):
                    heights = []
                    for hm in hm_row:
                        hm_size = QSizeF(hm.heatmap_item.pixmap().size())
                        hm_size = scaled(
                            hm_size, QSizeF(hm.size().width(), -1),
                            Qt.KeepAspectRatioByExpanding)

                        heights.append(hm_size.height())
                    layout.setRowMaximumHeight(3 + i * 2 + 1, max(heights))
                    layout.setRowPreferredHeight(3 + i * 2 + 1, max(heights))

                # set/update the widget's height
                constraint = QSizeF(size.width(), -1)
                sh = widget.effectiveSizeHint(Qt.PreferredSize, constraint)
                minsize = widget.effectiveSizeHint(Qt.MinimumSize, constraint)
                sh = sh.expandedTo(minsize).expandedTo(widget.minimumSize())

#                 print("Resize 2", sh)
#                 print("  old:", widget.size().width(), widget.size().height())
#                 print("  new:", widget.size().width(), sh.height())

                widget.resize(sh)
#                 print("Did resize")
            self.__fixup_grid_layout()

    def __fixup_grid_layout(self):
        self.__update_margins()
        rect = self.scene.widget.geometry()
        self.heatmap_scene.setSceneRect(rect)
        self.__update_selection_geometry()

    def __aspect_mode_changed(self):
        if self.keep_aspect:
            policy = Qt.ScrollBarAlwaysOn
        else:
            policy = Qt.ScrollBarAlwaysOff

        viewport = self.sceneView.viewport()
        # Temp. remove the event filter so we won't process the resize twice
        viewport.removeEventFilter(self)
        self.sceneView.setVerticalScrollBarPolicy(policy)
        self.sceneView.setHorizontalScrollBarPolicy(policy)
        viewport.installEventFilter(self)
        self.__update_size_constraints()

    def eventFilter(self, reciever, event):
        if reciever is self.sceneView.viewport() and \
                event.type() == QEvent.Resize:
            self.__update_size_constraints()

        return super().eventFilter(reciever, event)

    def __update_margins(self):
        """
        Update dendrogram and text list widgets margins to include the
        space for average stripe.
        """
        def offset(hm):
            if hm.show_averages:
                return hm.averages_item.size().width()
            else:
                return 0

        hm_row = self.heatmap_widget_grid[0]
        hm_col = next(zip(*self.heatmap_widget_grid))
        dendrogram_col = self.col_dendrograms
        dendrogram_row = self.row_dendrograms

        col_annot = zip(self.col_annotation_widgets_top,
                        self.col_annotation_widgets_bottom)
        row_annot = self.row_annotation_widgets

        for hm, annot, dendrogram in zip(hm_row, col_annot, dendrogram_col):
            width = hm.size().width()
            left_offset = offset(hm)
            col_count = hm.heatmap_data().shape[1]
            half_col = (width - left_offset) / col_count / 2
            if dendrogram is not None:
                _, top, _, bottom = dendrogram.getContentsMargins()
                dendrogram.setContentsMargins(
                    left_offset + half_col, top, half_col, bottom)

            _, top, right, bottom = annot[0].getContentsMargins()
            annot[0].setContentsMargins(left_offset, top, right, bottom)
            _, top, right, bottom = annot[1].getContentsMargins()
            annot[1].setContentsMargins(left_offset, top, right, bottom)

        for hm, annot, dendrogram in zip(hm_col, row_annot, dendrogram_row):
            height = hm.size().height()
            row_count = hm.heatmap_data().shape[0]
            half_row = height / row_count / 2
            if dendrogram is not None:
                left, _, right, _ = dendrogram.getContentsMargins()
                dendrogram.setContentsMargins(left, half_row, right, half_row)

    def heatmap_widgets(self):
        """Iterate over heatmap widgets.
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, GraphicsHeatmapWidget):
                yield item

    def label_widgets(self):
        """Iterate over GraphicsSimpleTextList widgets.
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, GraphicsSimpleTextList):
                yield item

    def dendrogram_widgets(self):
        """Iterate over dendrogram widgets
        """
        for item in self.heatmap_scene.items():
            if isinstance(item, DendrogramWidget):
                yield item

    def legend_widgets(self):
        for item in self.heatmap_scene.items():
            if isinstance(item, GradientLegendWidget):
                yield item

    def update_averages_stripe(self):
        """Update the visibility of the averages stripe.
        """
        if self.data is not None:
            for widget in self.heatmap_widgets():
                widget.set_show_averages(self.averages)
                widget.layout().activate()

            self.scene.widget.layout().activate()
            self.__fixup_grid_layout()

    def update_grid_spacing(self):
        """Update layout spacing.
        """
        if self.scene.widget:
            layout = self.scene.widget.layout()
            layout.setSpacing(self.SpaceX)
            self.__fixup_grid_layout()

    def update_color_schema(self):
        palette = self.color_palette()
        for heatmap in self.heatmap_widgets():
            heatmap.set_color_table(palette)

        for legend in self.legend_widgets():
            legend.set_color_table(palette)

    def update_sorting_examples(self):
        if self.data:
            self.update_heatmaps()

    def update_sorting_attributes(self):
        if self.data:
            self.update_heatmaps()

    def update_legend(self):
        for item in self.heatmap_scene.items():
            if isinstance(item, GradientLegendWidget):
                item.setVisible(self.legend)

    def update_annotations(self):
        if self.data is not None:
            if self.annotation_vars:
                var = self.annotation_vars[self.annotation_index]
                if var == '(None)':
                    var = None
            else:
                var = None

            show = var is not None
            if show:
                annot_col, _ = self.data.get_column_view(var)
            else:
                annot_col = None

            for labelslist in self.row_annotation_widgets:
                labelslist.setVisible(bool(show))
                if show:
                    indices = labelslist._indices
                    data = annot_col[indices]
                    labels = [var.str_val(val) for val in data]
                    labelslist.set_labels(labels)

    def update_column_annotations(self):
        if self.data is not None:
            show_top = self.column_label_pos & OWHeatMap.PositionTop
            show_bottom = self.column_label_pos & OWHeatMap.PositionBottom

            for labelslist in self.col_annotation_widgets_top:
                labelslist.setVisible(show_top)

            TopLabelsRow = 2
            Row0 = 3
            BottomLabelsRow = Row0 + 2 * len(self.heatmapparts.rows)

            layout = self.heatmap_scene.widget.layout()
            layout.setRowMaximumHeight(TopLabelsRow, -1 if show_top else 0)
            layout.setRowSpacing(TopLabelsRow, -1 if show_top else 0)

            for labelslist in self.col_annotation_widgets_bottom:
                labelslist.setVisible(show_bottom)

            layout.setRowMaximumHeight(BottomLabelsRow, -1 if show_top else 0)

            self.__fixup_grid_layout()

    def __select_by_cluster(self, item, dendrogramindex):
        # User clicked on a dendrogram node.
        # Select all rows corresponding to the cluster item.
        node = item.node
        try:
            hm = self.heatmap_widget_grid[dendrogramindex][0]
        except IndexError:
            pass
        else:
            key = QtGui.QApplication.keyboardModifiers()
            clear = not (key & ((Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier)))
            remove = (key & (Qt.ControlModifier | Qt.AltModifier))
            append = (key & Qt.ControlModifier)
            self.selection_manager.selection_add(
                node.value.first, node.value.last - 1, hm,
                clear=clear, remove=remove, append=append)

    def __update_selection_geometry(self):
        for item in self.selection_rects:
            item.setParentItem(None)
            self.heatmap_scene.removeItem(item)

        self.selection_rects = []
        self.selection_manager.update_selection_rects()
        rects = self.selection_manager.selection_rects
        for rect in rects:
            item = QtGui.QGraphicsRectItem(rect, None)
            pen = QPen(Qt.black, 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.heatmap_scene.addItem(item)
            self.selection_rects.append(item)

    def on_selection_finished(self):
        self.selected_rows = self.selection_manager.selections
        self.commit()

    def commit(self):
        data = None
        if self.input_data is not None and self.selected_rows:
            sortind = np.hstack([labels._indices for labels in self.row_annotation_widgets])
            indices = sortind[self.selected_rows]
            data = self.input_data[indices]

        self.send("Selected Data", data)

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.scene,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


class GraphicsWidget(QtGui.QGraphicsWidget):
    """A graphics widget which can notify on relayout events.
    """
    #: The widget's layout has activated (i.e. did a relayout
    #: of the widget's contents)
    layoutDidActivate = Signal()

    def event(self, event):
        rval = super().event(event)
        if event.type() == QEvent.LayoutRequest and self.layout() is not None:
            self.layoutDidActivate.emit()
        return rval

QWIDGETSIZE_MAX = 16777215


def scaled(size, constraint, mode=Qt.KeepAspectRatio):
    if constraint.width() < 0 and constraint.height() < 0:
        return size

    size, constraint = QSizeF(size), QSizeF(constraint)
    if mode == Qt.IgnoreAspectRatio:
        if constraint.width() >= 0:
            size.setWidth(constraint.width())
        if constraint.height() >= 0:
            size.setHeight(constraint.height())
    elif mode == Qt.KeepAspectRatio:
        if constraint.width() < 0:
            constraint.setWidth(QWIDGETSIZE_MAX)
        if constraint.height() < 0:
            constraint.setHeight(QWIDGETSIZE_MAX)
        size.scale(constraint, mode)
    elif mode == Qt.KeepAspectRatioByExpanding:
        if constraint.width() < 0:
            constraint.setWidth(0)
        if constraint.height() < 0:
            constraint.setHeight(0)
        size.scale(constraint, mode)
    return size


class GraphicsPixmapWidget(QtGui.QGraphicsWidget):
    def __init__(self, parent=None, pixmap=None, scaleContents=False,
                 aspectMode=Qt.KeepAspectRatio, **kwargs):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.__scaleContents = scaleContents
        self.__aspectMode = aspectMode

        self.__pixmap = pixmap or QPixmap()
        self.__item = QtGui.QGraphicsPixmapItem(self.__pixmap, self)
        self.__updateScale()

    def setPixmap(self, pixmap):
        self.prepareGeometryChange()
        self.__pixmap = pixmap or QPixmap()
        self.__item.setPixmap(self.__pixmap)
        self.updateGeometry()

    def pixmap(self):
        return self.__pixmap

    def setAspectRatioMode(self, mode):
        if self.__aspectMode != mode:
            self.__aspectMode = mode

    def aspectRatioMode(self):
        return self.__aspectMode

    def setScaleContents(self, scale):
        if self.__scaleContents != scale:
            self.__scaleContents = bool(scale)
            self.updateGeometry()
            self.__updateScale()

    def scaleContents(self):
        return self.__scaleContents

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            sh = QSizeF(self.__pixmap.size())
            if self.__scaleContents:
                sh = scaled(sh, constraint, self.__aspectMode)
            return sh
        elif which == Qt.MinimumSize:
            if self.__scaleContents:
                return QSizeF(0, 0)
            else:
                return QSizeF(self.__pixmap.size())
        elif which == Qt.MaximumSize:
            if self.__scaleContents:
                return QSizeF()
            else:
                return QSizeF(self.__pixmap.size())
        else:
            # Qt.MinimumDescent
            return QSizeF()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        crect = self.contentsRect()
        self.__item.setPos(crect.topLeft())
        self.__updateScale()

    def __updateScale(self):
        if self.__pixmap.isNull():
            return
        pxsize = QSizeF(self.__pixmap.size())
        crect = self.contentsRect()
        self.__item.setPos(crect.topLeft())

        if self.__scaleContents:
            csize = scaled(pxsize, crect.size(), self.__aspectMode)
        else:
            csize = pxsize

        xscale = csize.width() / pxsize.width()
        yscale = csize.height() / pxsize.height()

        t = QtGui.QTransform().scale(xscale, yscale)
        self.__item.setTransform(t)

    def pixmapTransform(self):
        return QtGui.QTransform(self.__item.transform())


class GraphicsHeatmapWidget(QtGui.QGraphicsWidget):
    def __init__(self, parent=None, data=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setAcceptHoverEvents(True)

        self.__levels = None
        self.__colortable = None
        self.__data = data

        self.__pixmap = QPixmap()
        self.__avgpixmap = QPixmap()

        layout = QtGui.QGraphicsLinearLayout(Qt.Horizontal)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_item = GraphicsPixmapWidget(
            self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio)

        self.averages_item = GraphicsPixmapWidget(
            self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio)

        layout.addItem(self.averages_item)
        layout.addItem(self.heatmap_item)
        layout.setItemSpacing(0, 2)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.show_averages = True

        self.set_heatmap_data(data)

    def clear(self):
        """Clear/reset the widget."""
        self.__data = None
        self.__pixmap = None
        self.__avgpixmap = None

        self.heatmap_item.setPixmap(QtGui.QPixmap())
        self.averages_item.setPixmap(QtGui.QPixmap())
        self.show_averages = True
        self.updateGeometry()
        self.layout().invalidate()

    def set_heatmap(self, heatmap):
        """Set the heatmap data for display.
        """
        self.clear()

        self.set_heatmap_data(heatmap)
        self.update()

    def set_heatmap_data(self, data):
        """Set the heatmap data for display."""
        if self.__data is not data:
            self.clear()
            self.__data = data
            self._update_pixmap()
            self.update()

    def heatmap_data(self):
        if self.__data is not None:
            v = self.__data.view()
            v.flags.writeable = False
            return v
        else:
            return None

    def set_levels(self, levels):
        if levels != self.__levels:
            self.__levels = levels
            self._update_pixmap()
            self.update()

    def set_show_averages(self, show):
        if self.show_averages != show:
            self.show_averages = show
            self.averages_item.setVisible(show)
            self.averages_item.setMaximumWidth(-1 if show else 0)
            self.layout().invalidate()
            self.update()

    def set_color_table(self, table):
        self.__colortable = table
        self._update_pixmap()
        self.update()

    def _update_pixmap(self):
        """
        Update the pixmap if its construction arguments changed.
        """
        if self.__data is not None:
            if self.__colortable is not None:
                lut = self.__colortable
            else:
                lut = None
            argb, _ = pg.makeARGB(
                self.__data, lut=lut, levels=self.__levels, scale=250)
            argb[np.isnan(self.__data)] = (100, 100, 100, 255)

            qimage = pg.makeQImage(argb, transpose=False)
            self.__pixmap = QPixmap.fromImage(qimage)
            avg = np.nanmean(self.__data, axis=1, keepdims=True)
            argb, _ = pg.makeARGB(
                avg, lut=lut, levels=self.__levels, scale=250)
            qimage = pg.makeQImage(argb, transpose=False)
            self.__avgpixmap = QPixmap.fromImage(qimage)
        else:
            self.__pixmap = QPixmap()
            self.__avgpixmap = QPixmap()

        self.heatmap_item.setPixmap(self.__pixmap)
        self.averages_item.setPixmap(self.__avgpixmap)
        self.layout().invalidate()

    def cell_at(self, pos):
        """Return the cell row, column from `pos` in local coordinates.
        """
        if self.__pixmap.isNull() or not (
                    self.heatmap_item.geometry().contains(pos) or
                    self.averages_item.geometry().contains(pos)):
            return (-1, -1)

        if self.heatmap_item.geometry().contains(pos):
            item_clicked = self.heatmap_item
        elif self.averages_item.geometry().contains(pos):
            item_clicked = self.averages_item
        pos = self.mapToItem(item_clicked, pos)
        size = self.heatmap_item.size()

        x, y = pos.x(), pos.y()

        N, M = self.__data.shape
        fx = x / size.width()
        fy = y / size.height()
        i = min(int(math.floor(fy * N)), N - 1)
        j = min(int(math.floor(fx * M)), M - 1)
        return i, j

    def cell_rect(self, row, column):
        """Return a rectangle in local coordinates containing the cell
        at `row` and `column`.
        """
        size = self.__pixmap.size()
        if not (0 <= column < size.width() or 0 <= row < size.height()):
            return QRectF()

        topleft = QPointF(column, row)
        bottomright = QPointF(column + 1, row + 1)
        t = self.heatmap_item.pixmapTransform()
        rect = t.mapRect(QRectF(topleft, bottomright))
        rect.translated(self.heatmap_item.pos())
        return rect

    def row_rect(self, row):
        """
        Return a QRectF in local coordinates containing the entire row.
        """
        rect = self.cell_rect(row, 0)
        rect.setLeft(0)
        rect.setRight(self.size().width())
        return rect

    def cell_tool_tip(self, row, column):
        return "{}, {}: {:g}".format(row, column, self.__data[row, column])

    def hoverMoveEvent(self, event):
        pos = event.pos()
        row, column = self.cell_at(pos)
        tooltip = self.cell_tool_tip(row, column)
        # TODO: Move/delegate to (Scene) helpEvent
        self.setToolTip(tooltip)
        return super().hoverMoveEvent(event)


class HeatmapScene(QGraphicsScene):
    """A Graphics Scene with heatmap widgets."""
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.selection_manager = HeatmapSelectionManager()

    def set_selection_manager(self, manager):
        self.selection_manager = manager

    def _items(self, pos=None, cls=object):
        if pos is not None:
            items = self.items(QRectF(pos, QSizeF(3, 3)).translated(-1.5, -1.5))
        else:
            items = self.items()

        for item in items:
            if isinstance(item, cls):
                yield item

    def heatmap_at_pos(self, pos):
        items = list(self._items(pos, GraphicsHeatmapWidget))
        if items:
            return items[0]
        else:
            return None

    def dendrogram_at_pos(self, pos):
        return None

        items = list(self._items(pos, DendrogramItem))
        if items:
            return items[0]
        else:
            return None

    def heatmap_widgets(self):
        return self._items(None, GraphicsHeatmapWidget)

    def select_from_dendrogram(self, dendrogram, key):
        """Select all heatmap rows which belong to the dendrogram.
        """
        dendrogram_widget = dendrogram.parentWidget()
        anchors = list(dendrogram_widget.leaf_anchors())
        cluster = dendrogram.cluster
        start, end = anchors[cluster.first], anchors[cluster.last - 1]
        start, end = dendrogram_widget.mapToScene(start), dendrogram_widget.mapToScene(end)
        # Find a heatmap widget containing start and end y coordinates.

        heatmap = None
        for hm in self.heatmap_widgets():
            b_rect = hm.sceneBoundingRect()
            if b_rect.contains(QPointF(b_rect.center().x(), start.y())):
                heatmap = hm
                break

        if dendrogram:
            b_rect = heatmap.boundingRect()
            start, end = heatmap.mapFromScene(start), heatmap.mapFromScene(end)
            start, _ = heatmap.cell_at(QPointF(b_rect.center().x(), start.y()))
            end, _ = heatmap.cell_at(QPointF(b_rect.center().x(), end.y()))
            clear = not (key & ((Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier)))
            remove = (key & (Qt.ControlModifier | Qt.AltModifier))
            append = (key & Qt.ControlModifier)
            self.selection_manager.selection_add(
                start, end, heatmap, clear=clear, remove=remove, append=append)
        return

    def mousePressEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap and event.button() & Qt.LeftButton:
            row, _ = heatmap.cell_at(heatmap.mapFromScene(pos))
            self.selection_manager.selection_start(heatmap, event)

        dendrogram = self.dendrogram_at_pos(pos)
        if dendrogram and event.button() & Qt.LeftButton:
            if dendrogram.orientation == Qt.Vertical:
                self.select_from_dendrogram(dendrogram, event.modifiers())
            return

        return QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap and event.buttons() & Qt.LeftButton:
            row, _ = heatmap.cell_at(heatmap.mapFromScene(pos))
            self.selection_manager.selection_update(heatmap, event)

        dendrogram = self.dendrogram_at_pos(pos)
        if dendrogram and dendrogram.orientation == Qt.Horizontal:  # Filter mouse move events
            return

        return QGraphicsScene.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        pos = event.scenePos()
        heatmap = self.heatmap_at_pos(pos)
        if heatmap:
            row, _ = heatmap.cell_at(heatmap.mapFromScene(pos))
            self.selection_manager.selection_finish(heatmap, event)

        dendrogram = self.dendrogram_at_pos(pos)
        if dendrogram and dendrogram.orientation == Qt.Horizontal:  # Filter mouse events
            return

        return QGraphicsScene.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        pos = event.scenePos()
        dendrogram = self.dendrogram_at_pos(pos)
        if dendrogram:  # Filter mouse events
            return
        return QGraphicsScene.mouseDoubleClickEvent(self, event)


class GraphicsSimpleTextLayoutItem(QtGui.QGraphicsLayoutItem):
    """ A Graphics layout item wrapping a QGraphicsSimpleTextItem alowing it
    to be managed by a layout.

    """
    def __init__(self, text_item, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.text_item = text_item
        if orientation == Qt.Vertical:
            self.text_item.rotate(-90)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        if self.orientation == Qt.Horizontal:
            self.text_item.setPos(rect.topLeft())
        else:
            self.text_item.setPos(rect.bottomLeft())

    def sizeHint(self, which, constraint=QSizeF()):
        if which in [Qt.PreferredSize]:
            size = self.text_item.boundingRect().size()
            if self.orientation == Qt.Horizontal:
                return size
            else:
                return QSizeF(size.height(), size.width())
        else:
            return QSizeF()

    def updateGeometry(self):
        super().updateGeometry()
        parent = self.parentLayoutItem()
        if parent.isLayout():
            parent.updateGeometry()

    def setFont(self, font):
        self.text_item.setFont(font)
        self.updateGeometry()

    def setText(self, text):
        self.text_item.setText(text)
        self.updateGeometry()


class GraphicsSimpleTextList(QtGui.QGraphicsWidget):
    """A simple text list widget."""
    def __init__(self, labels=[], orientation=Qt.Vertical, parent=None):
        super().__init__(parent)
        self.label_items = []
        self.orientation = orientation
        self.alignment = Qt.AlignCenter
        self.__resize_in_progress = False

        layout = QtGui.QGraphicsLinearLayout(orientation)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.set_labels(labels)

    def clear(self):
        """Remove all text items."""
        layout = self.layout()
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            item.text_item.setParentItem(None)
            if self.scene():
                self.scene().removeItem(item.text_item)
            layout.removeAt(i)

        self.label_items = []
#         self.updateGeometry()

    def set_labels(self, labels):
        """Set the text labels to show in the widget.
        """
        self.clear()
        orientation = Qt.Horizontal if self.orientation == Qt.Vertical else Qt.Vertical
        for text in labels:
            item = QtGui.QGraphicsSimpleTextItem(text, self)
            item.setFont(self.font())
            item.setToolTip(text)
            item = GraphicsSimpleTextLayoutItem(item, orientation, parent=self)
            self.layout().addItem(item)
            self.layout().setAlignment(item, self.alignment)
            self.label_items.append(item)

    def setAlignment(self, alignment):
        """Set alignment of text items in the widget
        """
        self.alignment = alignment
        layout = self.layout()
        for i in range(layout.count()):
            layout.setAlignment(layout.itemAt(i), alignment)

    def sizeHint(self, which, constraint=QRectF()):
        if not self.isVisible():
            return QSizeF(0, 0)
        else:
            return super().sizeHint(which, constraint)

    def setVisible(self, visible):
        super().setVisible(visible)
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.__resize_in_progress = True
        self._updateFontSize()
        self.__resize_in_progress = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            font = self.font()
            for item in self.label_items:
                item.setFont(font)

            if not self.__resize_in_progress:
                self.updateGeometry()
                self.layout().invalidate()
                self.layout().activate()

    def _updateFontSize(self):
        crect = self.contentsRect()
        if self.orientation == Qt.Vertical:
            h = crect.height()
        else:
            h = crect.width()
        n = len(self.label_items)
        if n == 0:
            return

        if self.scene() is not None:
            maxfontsize = self.scene().font().pointSize()
        else:
            maxfontsize = QtGui.QApplication.instance().font().pointSize()

        lineheight = max(1, h / n)
        fontsize = min(self._pointSize(lineheight), maxfontsize)

        font = self.font()
        font.setPointSize(fontsize)
        self.setFont(font)

    def _pointSize(self, height):
        font = self.font()
        font.setPointSize(height)
        fix = 0
        while QFontMetrics(font).lineSpacing() > height and height - fix > 1:
            fix += 1
            font.setPointSize(height - fix)
        return height - fix


class GradientLegendWidget(QtGui.QGraphicsWidget):
    def __init__(self, low, high, parent=None):
        super().__init__(parent)
        self.low = low
        self.high = high
        self.color_table = None

        layout = QtGui.QGraphicsLinearLayout(Qt.Vertical)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        layout_labels = QtGui.QGraphicsLinearLayout(Qt.Horizontal)
        layout.addItem(layout_labels)
        layout_labels.setContentsMargins(0, 0, 0, 0)
        label_lo = QtGui.QGraphicsSimpleTextItem("%.2f" % low, self)
        label_hi = QtGui.QGraphicsSimpleTextItem("%.2f" % high, self)
        self.item_low = GraphicsSimpleTextLayoutItem(label_lo, parent=self)
        self.item_high = GraphicsSimpleTextLayoutItem(label_hi, parent=self)

        layout_labels.addItem(self.item_low)
        layout_labels.addStretch(10)
        layout_labels.addItem(self.item_high)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.__pixitem = GraphicsPixmapWidget(parent=self, scaleContents=True,
                                              aspectMode=Qt.IgnoreAspectRatio)
        self.__pixitem.setMinimumHeight(12)
        layout.addItem(self.__pixitem)
        self.__update()

    def set_color_table(self, color_table):
        self.color_table = color_table
        self.__update()

    def __update(self):
        data = np.linspace(self.low, self.high, num=50, endpoint=True)
        data = data.reshape((1, -1))
        argb, _ = pg.makeARGB(data, lut=self.color_table,
                              levels=(self.low, self.high))
        qimg = pg.makeQImage(argb, transpose=False)
        self.__pixitem.setPixmap(QPixmap.fromImage(qimg))

        self.item_low.setText("%.2f" % self.low)
        self.item_high.setText("%.2f" % self.high)
        self.layout().invalidate()


class HeatmapSelectionManager(QObject):
    """Selection manager for heatmap rows
    """
    selection_changed = Signal()
    selection_finished = Signal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.selections = []
        self.selection_ranges = []
        self.selection_ranges_temp = []
        self.heatmap_widgets = []
        self.selection_rects = []
        self.heatmaps = []
        self._heatmap_ranges = {}
        self._start_row = 0

    def clear(self):
        self.remove_rows(self.selection)

    def set_heatmap_widgets(self, widgets):
        self.remove_rows(self.selections)
        self.heatmaps = list(zip(*widgets))

        # Compute row ranges for all heatmaps
        self._heatmap_ranges = {}
        start = end = 0

        for group in zip(*widgets):
            start = end = 0
            for heatmap in group:
                end += heatmap.heatmap_data().shape[0]
                self._heatmap_ranges[heatmap] = (start, end)
                start = end

    def select_rows(self, rows, heatmap=None, clear=True):
        """Add `rows` to selection. If `heatmap` is provided the rows
        are mapped from the local indices to global heatmap indics. If `clear`
        then remove previous rows.
        """
        if heatmap is not None:
            start, _ = self._heatmap_ranges[heatmap]
            rows = [start + r for r in rows]

        old_selection = list(self.selections)
        if clear:
            self.selections = rows
        else:
            self.selections = sorted(set(self.selections + rows))

        if self.selections != old_selection:
            self.update_selection_rects()
            self.selection_changed.emit()

    def remove_rows(self, rows):
        """Remove `rows` from the selection.
        """
        old_selection = list(self.selections)
        self.selections = sorted(set(self.selections) - set(rows))
        if old_selection != self.selections:
            self.update_selection_rects()
            self.selection_changed.emit()

    def combined_ranges(self, ranges):
        combined_ranges = set()
        for start, end in ranges:
            if start <= end:
                rng = range(start, end + 1)
            else:
                rng = range(start, end - 1, -1)
            combined_ranges.update(rng)
        return sorted(combined_ranges)

    def selection_start(self, heatmap_widget, event):
        """ Selection  started by `heatmap_widget` due to `event`.
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)

        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        self._start_row = row
        range = (row, row)
        self.selection_ranges_temp = []
        if event.modifiers() & Qt.ControlModifier:
            self.selection_ranges_temp = self.selection_ranges
            self.selection_ranges = self.remove_range(
                self.selection_ranges, row, row, append=True)
        elif event.modifiers() & Qt.ShiftModifier:
            self.selection_ranges.append(range)
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, row, row, append=False)
        else:
            self.selection_ranges = [range]
        self.select_rows(self.combined_ranges(self.selection_ranges))

    def selection_update(self, heatmap_widget, event):
        """ Selection updated by `heatmap_widget due to `event` (mouse drag).
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)
        if row < 0:
            return

        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        if event.modifiers() & Qt.ControlModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges_temp, self._start_row, row, append=True)
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, self._start_row, row, append=False)
        else:
            if self.selection_ranges:
                self.selection_ranges[-1] = (self._start_row, row)
            else:
                self.selection_ranges = [(row, row)]

        self.select_rows(self.combined_ranges(self.selection_ranges))

    def selection_finish(self, heatmap_widget, event):
        """ Selection finished by `heatmap_widget due to `event`.
        """
        pos = heatmap_widget.mapFromScene(event.scenePos())
        row, _ = heatmap_widget.cell_at(pos)
        start, _ = self._heatmap_ranges[heatmap_widget]
        row = start + row
        if event.modifiers() & Qt.ControlModifier:
            pass
        elif event.modifiers() & Qt.AltModifier:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, self._start_row, row, append=False)
        else:
            self.selection_ranges[-1] = (self._start_row, row)
        self.select_rows(self.combined_ranges(self.selection_ranges))
        self.selection_finished.emit()

    def selection_add(self, start, end, heatmap=None, clear=True,
                      remove=False, append=False):
        """ Add/remove a selection range from `start` to `end`.
        """
        if heatmap is not None:
            _start, _ = self._heatmap_ranges[heatmap]
            start = _start + start
            end = _start + end

        if clear:
            self.selection_ranges = []
        if remove:
            self.selection_ranges = self.remove_range(
                self.selection_ranges, start, end, append=append)
        else:
            self.selection_ranges.append((start, end))
        self.select_rows(self.combined_ranges(self.selection_ranges))
        self.selection_finished.emit()

    def remove_range(self, ranges, start, end, append=False):
        if start > end:
            start, end = end, start
        comb_ranges = [i for i in self.combined_ranges(ranges)
                       if i > end or i < start]
        if append:
            comb_ranges += [i for i in range(start, end + 1)
                            if i not in self.combined_ranges(ranges)]
            comb_ranges = sorted(comb_ranges)
        return self.combined_to_ranges(comb_ranges)

    def combined_to_ranges(self, comb_ranges):
        ranges = []
        if len(comb_ranges) > 0:
            i, start, end = 0, comb_ranges[0], comb_ranges[0]
            for val in comb_ranges[1:]:
                i += 1
                if start + i < val:
                    ranges.append((start, end))
                    i, start = 0, val
                end = val
            ranges.append((start, end))
        return ranges

    def update_selection_rects(self):
        """ Update the selection rects.
        """
        def continuous_ranges(selections):
            """ Group continuous ranges
            """
            selections = iter(selections)
            start = end = next(selections)
            try:
                while True:
                    new_end = next(selections)
                    if new_end > end + 1:
                        yield start, end
                        start = end = new_end
                    else:
                        end = new_end
            except StopIteration:
                yield start, end

        def group_selections(selections):
            """Group selections along with heatmaps.
            """
            rows2hm = self.rows_to_heatmaps()
            selections = iter(selections)
            start = end = next(selections)
            end_heatmaps = rows2hm[end]
            try:
                while True:
                    new_end = next(selections)
                    new_end_heatmaps = rows2hm[new_end]
                    if new_end > end + 1 or new_end_heatmaps != end_heatmaps:
                        yield start, end, end_heatmaps
                        start = end = new_end
                        end_heatmaps = new_end_heatmaps
                    else:
                        end = new_end

            except StopIteration:
                yield start, end, end_heatmaps

        def selection_rect(start, end, heatmaps):
            rect = QRectF()
            for heatmap in heatmaps:
                h_start, _ = self._heatmap_ranges[heatmap]
                rect |= heatmap.mapToScene(heatmap.row_rect(start - h_start)).boundingRect()
                rect |= heatmap.mapToScene(heatmap.row_rect(end - h_start)).boundingRect()
            return rect

        self.selection_rects = []
        for start, end, heatmaps in group_selections(self.selections):
            rect = selection_rect(start, end, heatmaps)
            self.selection_rects.append(rect)

    def rows_to_heatmaps(self):
        heatmap_groups = zip(*self.heatmaps)
        rows2hm = {}
        for heatmaps in heatmap_groups:
            hm = heatmaps[0]
            start, end = self._heatmap_ranges[hm]
            rows2hm.update(dict.fromkeys(range(start, end), heatmaps))
        return rows2hm


def test_main(argv=sys.argv):
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    app = QtGui.QApplication(argv)
    ow = OWHeatMap()

    ow.set_dataset(Orange.data.Table(filename))
    ow.handleNewSignals()
    ow.show()
    ow.raise_()
    app.exec_()
    ow.set_dataset(None)
    ow.handleNewSignals()
    ow.saveSettings()
    return 0

if __name__ == "__main__":
    sys.exit(test_main())
