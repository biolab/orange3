import enum
from collections import defaultdict
from itertools import islice
from typing import (
    Iterable, Mapping, Any, TypeVar, NamedTuple, Sequence, Optional,
    Union, Tuple, List, Callable
)

import numpy as np
import scipy.sparse as sp

from AnyQt.QtWidgets import (
    QGraphicsView, QFormLayout, QComboBox, QGroupBox, QMenu, QAction,
    QSizePolicy
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QFont, QKeySequence
from AnyQt.QtCore import Qt, QSize, QRectF, QObject

from orangewidget.utils.combobox import ComboBox, ComboBoxSearch
from Orange.data import Domain, Table, Variable, DiscreteVariable, \
    ContinuousVariable
from Orange.data.sql.table import SqlTable
import Orange.distance

from Orange.clustering import hierarchical, kmeans
from Orange.widgets.utils import colorpalettes, apply_all, enum_get, itemmodels
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.utils.graphicsview import GraphicsWidgetView
from Orange.widgets.utils.graphicsscene import GraphicsScene
from Orange.widgets.utils.colorpalettes import Palette

from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Msg, Input, Output

from Orange.widgets.data.oweditdomain import table_column_data
from Orange.widgets.visualize.utils.heatmap import HeatmapGridWidget, \
    ColorMap, CategoricalColorMap, GradientColorMap
from Orange.widgets.utils.colorgradientselection import ColorGradientSelection
from Orange.widgets.utils.widgetpreview import WidgetPreview


__all__ = []


def kmeans_compress(X, k=50):
    km = kmeans.KMeans(n_clusters=k, n_init=5, random_state=42)
    return km.get_model(X)


def split_domain(domain: Domain, split_label: str):
    """Split the domain based on values of `split_label` value.
    """
    groups = defaultdict(list)
    for var in domain.attributes:
        val = var.attributes.get(split_label)
        groups[val].append(var)
    if None in groups:
        na = groups.pop(None)
        return [*groups.items(), ("N/A", na)]
    else:
        return list(groups.items())


def cbselect(cb: QComboBox, value, role: Qt.ItemDataRole = Qt.EditRole) -> None:
    """
    Find and select the `value` in the `cb` QComboBox.

    Parameters
    ----------
    cb: QComboBox
    value: Any
    role: Qt.ItemDataRole
        The data role in the combo box model to match value against
    """
    cb.setCurrentIndex(cb.findData(value, role))


class Clustering(enum.IntEnum):
    #: No clustering
    None_ = 0
    #: Hierarchical clustering
    Clustering = 1
    #: Hierarchical clustering with optimal leaf ordering
    OrderedClustering = 2


ClusteringRole = Qt.UserRole + 13
#: Item data for clustering method selection models
ClusteringModelData = [
    {
        Qt.DisplayRole: "None",
        Qt.ToolTipRole: "No clustering",
        ClusteringRole: Clustering.None_,
    }, {
        Qt.DisplayRole: "Clustering",
        Qt.ToolTipRole: "Apply hierarchical clustering",
        ClusteringRole: Clustering.Clustering,
    }, {
        Qt.DisplayRole: "Clustering (opt. ordering)",
        Qt.ToolTipRole: "Apply hierarchical clustering with optimal leaf "
                        "ordering.",
        ClusteringRole: Clustering.OrderedClustering,
    }
]

ColumnLabelsPosData = [
    {Qt.DisplayRole: name, Qt.UserRole: value}
    for name, value in [
        ("None", HeatmapGridWidget.NoPosition),
        ("Top", HeatmapGridWidget.PositionTop),
        ("Bottom", HeatmapGridWidget.PositionBottom),
        ("Top and Bottom", (HeatmapGridWidget.PositionTop |
                            HeatmapGridWidget.PositionBottom)),
    ]
]


def create_list_model(
        items: Iterable[Mapping[Qt.ItemDataRole, Any]],
        parent: Optional[QObject] = None,
) -> QStandardItemModel:
    """Create list model from an item date iterable."""
    model = QStandardItemModel(parent)
    for item in items:
        sitem = QStandardItem()
        for role, value in item.items():
            sitem.setData(value, role)
        model.appendRow([sitem])
    return model


class OWHeatMap(widget.OWWidget):
    name = "Heat Map"
    description = "Plot a data matrix heatmap."
    icon = "icons/Heatmap.svg"
    priority = 260
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settings_version = 3

    settingsHandler = settings.DomainContextHandler()

    # Disable clustering for inputs bigger than this
    MaxClustering = 25000
    # Disable cluster leaf ordering for inputs bigger than this
    MaxOrderedClustering = 1000

    threshold_low = settings.Setting(0.0)
    threshold_high = settings.Setting(1.0)
    color_center = settings.Setting(0)

    merge_kmeans = settings.Setting(False)
    merge_kmeans_k = settings.Setting(50)

    # Display column with averages
    averages: bool = settings.Setting(True)
    # Display legend
    legend: bool = settings.Setting(True)
    # Annotations
    #: text row annotation (row names)
    annotation_var = settings.ContextSetting(None)
    #: color row annotation
    annotation_color_var = settings.ContextSetting(None)
    column_annotation_color_key: Optional[Tuple[str, str]] = settings.ContextSetting(None)

    # Discrete variable used to split that data/heatmaps (vertically)
    split_by_var = settings.ContextSetting(None)
    # Split heatmap columns by 'key' (horizontal)
    split_columns_key: Optional[Tuple[str, str]] = settings.ContextSetting(None)
    # Selected row/column clustering method (name)
    col_clustering_method: str = settings.Setting(Clustering.None_.name)
    row_clustering_method: str = settings.Setting(Clustering.None_.name)

    palette_name = settings.Setting(colorpalettes.DefaultContinuousPaletteName)
    column_label_pos: int = settings.Setting(1)
    selected_rows: List[int] = settings.Setting(None, schema_only=True)

    auto_commit = settings.Setting(True)

    graph_name = "scene"

    class Information(widget.OWWidget.Information):
        sampled = Msg("Data has been sampled")
        discrete_ignored = Msg("{} categorical feature{} ignored")
        row_clust = Msg("{}")
        col_clust = Msg("{}")
        sparse_densified = Msg("Showing this data may require a lot of memory")

    class Error(widget.OWWidget.Error):
        no_continuous = Msg("No numeric features")
        not_enough_features = Msg("Not enough features for column clustering")
        not_enough_instances = Msg("Not enough instances for clustering")
        not_enough_instances_k_means = Msg(
            "Not enough instances for k-means merging")
        not_enough_memory = Msg("Not enough memory to show this data")

    class Warning(widget.OWWidget.Warning):
        empty_clusters = Msg("Empty clusters were removed")

    UserAdviceMessages = [
        widget.Message(
            "For data with a meaningful mid-point, "
            "choose one of diverging palettes.",
            "diverging_palette")]

    def __init__(self):
        super().__init__()
        self.__pending_selection = self.selected_rows

        # A kingdom for a save_state/restore_state
        self.col_clustering = enum_get(
            Clustering, self.col_clustering_method, Clustering.None_)
        self.row_clustering = enum_get(
            Clustering, self.row_clustering_method, Clustering.None_)

        self.settingsAboutToBePacked.connect(self._save_state_for_serialization)
        self.keep_aspect = False

        #: The original data with all features (retained to
        #: preserve the domain on the output)
        self.input_data = None
        #: The effective data striped of discrete features, and often
        #: merged using k-means
        self.data = None
        self.effective_data = None
        #: Source of column annotations (derived from self.data)
        self.col_annot_data: Optional[Table] = None
        #: kmeans model used to merge rows of input_data
        self.kmeans_model = None
        #: merge indices derived from kmeans
        #: a list (len==k) of int ndarray where the i-th item contains
        #: the indices which merge the input_data into the heatmap row i
        self.merge_indices = None
        self.parts: Optional[Parts] = None
        self.__rows_cache = {}
        self.__columns_cache = {}

        # GUI definition
        colorbox = gui.vBox(self.controlArea, "Color")

        self.color_map_widget = cmw = ColorGradientSelection(
            thresholds=(self.threshold_low, self.threshold_high),
            center=self.color_center
        )
        model = itemmodels.ContinuousPalettesModel(parent=self)
        cmw.setModel(model)
        idx = cmw.findData(self.palette_name, model.KeyRole)
        if idx != -1:
            cmw.setCurrentIndex(idx)

        cmw.activated.connect(self.update_color_schema)

        def _set_thresholds(low, high):
            self.threshold_low, self.threshold_high = low, high
            self.update_color_schema()
        cmw.thresholdsChanged.connect(_set_thresholds)

        def _set_centering(center):
            self.color_center = center
            self.update_color_schema()
        cmw.centerChanged.connect(_set_centering)

        colorbox.layout().addWidget(self.color_map_widget)

        mergebox = gui.vBox(self.controlArea, "Merge",)
        gui.checkBox(mergebox, self, "merge_kmeans", "Merge by k-means",
                     callback=self.__update_row_clustering)
        ibox = gui.indentedBox(mergebox)
        gui.spin(ibox, self, "merge_kmeans_k", minv=5, maxv=500,
                 label="Clusters:", keyboardTracking=False,
                 callbackOnReturn=True, callback=self.update_merge)

        cluster_box = gui.vBox(self.controlArea, "Clustering")
        # Row clustering
        self.row_cluster_cb = cb = ComboBox()
        cb.setModel(create_list_model(ClusteringModelData, self))
        cbselect(cb, self.row_clustering, ClusteringRole)
        self.connect_control(
            "row_clustering",
            lambda value, cb=cb: cbselect(cb, value, ClusteringRole)
        )
        @cb.activated.connect
        def _(idx, cb=cb):
            self.set_row_clustering(cb.itemData(idx, ClusteringRole))

        # Column clustering
        self.col_cluster_cb = cb = ComboBox()
        cb.setModel(create_list_model(ClusteringModelData, self))
        cbselect(cb, self.col_clustering, ClusteringRole)
        self.connect_control(
            "col_clustering",
            lambda value, cb=cb: cbselect(cb, value, ClusteringRole)
        )
        @cb.activated.connect
        def _(idx, cb=cb):
            self.set_col_clustering(cb.itemData(idx, ClusteringRole))

        form = QFormLayout(
            labelAlignment=Qt.AlignLeft, formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        form.addRow("Rows:", self.row_cluster_cb)
        form.addRow("Columns:", self.col_cluster_cb)
        cluster_box.layout().addLayout(form)
        box = gui.vBox(self.controlArea, "Split By")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft, labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        box.layout().addLayout(form)

        self.row_split_model = DomainModel(
            placeholder="(None)",
            valid_types=(Orange.data.DiscreteVariable,),
            parent=self,
        )
        self.row_split_cb = cb = ComboBoxSearch(
            enabled=not self.merge_kmeans,
            sizeAdjustPolicy=ComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=14,
            toolTip="Split the heatmap vertically by a categorical column"
        )
        self.row_split_cb.setModel(self.row_split_model)
        self.connect_control(
            "split_by_var", lambda value, cb=cb: cbselect(cb, value)
        )
        self.connect_control(
            "merge_kmeans", self.row_split_cb.setDisabled
        )
        self.split_by_var = None

        self.row_split_cb.activated.connect(
            self.__on_split_rows_activated
        )
        self.col_split_model = DomainModel(
            placeholder="(None)",
            order=DomainModel.MIXED,
            valid_types=(Orange.data.DiscreteVariable,),
            parent=self,
        )
        self.col_split_cb = cb = ComboBoxSearch(
            sizeAdjustPolicy=ComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=14,
            toolTip="Split the heatmap horizontally by column annotation"
        )
        self.col_split_cb.setModel(self.col_split_model)
        self.connect_control(
            "split_columns_var", lambda value, cb=cb: cbselect(cb, value)
        )
        self.split_columns_var = None
        self.col_split_cb.activated.connect(self.__on_split_cols_activated)
        form.addRow("Rows:", self.row_split_cb)
        form.addRow("Columns:", self.col_split_cb)

        box = gui.vBox(self.controlArea, 'Annotation && Legends')

        gui.checkBox(box, self, 'legend', 'Show legend',
                     callback=self.update_legend)

        gui.checkBox(box, self, 'averages', 'Stripes with averages',
                     callback=self.update_averages_stripe)
        gui.separator(box)
        annotbox = QGroupBox("Row Annotations")
        form = QFormLayout(
            annotbox,
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        self.annotation_model = DomainModel(placeholder="(None)")
        self.annotation_text_cb = ComboBoxSearch(
            minimumContentsLength=12,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.annotation_text_cb.setModel(self.annotation_model)
        self.annotation_text_cb.activated.connect(self.set_annotation_var)
        self.connect_control("annotation_var", self.annotation_var_changed)

        self.row_side_color_model = DomainModel(
            order=(DomainModel.CLASSES, DomainModel.Separator,
                   DomainModel.METAS),
            placeholder="(None)", valid_types=DomainModel.PRIMITIVE,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled,
            parent=self,
        )
        self.row_side_color_cb = ComboBoxSearch(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=12
        )
        self.row_side_color_cb.setModel(self.row_side_color_model)
        self.row_side_color_cb.activated.connect(self.set_annotation_color_var)
        self.connect_control("annotation_color_var", self.annotation_color_var_changed)
        form.addRow("Text", self.annotation_text_cb)
        form.addRow("Color", self.row_side_color_cb)
        box.layout().addWidget(annotbox)
        annotbox = QGroupBox("Column annotations")
        form = QFormLayout(
            annotbox,
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        self.col_side_color_model = DomainModel(
            placeholder="(None)",
            valid_types=(DiscreteVariable, ContinuousVariable),
            parent=self
        )
        self.col_side_color_cb = cb = ComboBoxSearch(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=12
        )
        self.col_side_color_cb.setModel(self.col_side_color_model)
        self.connect_control(
            "column_annotation_color_var", self.column_annotation_color_var_changed,
        )
        self.column_annotation_color_var = None
        self.col_side_color_cb.activated.connect(
            self.__set_column_annotation_color_var_index)

        cb = gui.comboBox(
            None, self, "column_label_pos",
            callback=self.update_column_annotations)
        cb.setModel(create_list_model(ColumnLabelsPosData, parent=self))
        cb.setCurrentIndex(self.column_label_pos)
        form.addRow("Position", cb)
        form.addRow("Color", self.col_side_color_cb)
        box.layout().addWidget(annotbox)

        gui.checkBox(self.controlArea, self, "keep_aspect",
                     "Keep aspect ratio", box="Resize",
                     callback=self.__aspect_mode_changed)

        gui.rubber(self.controlArea)

        gui.auto_send(self.buttonsArea, self, "auto_commit")

        # Scene with heatmap
        class HeatmapScene(GraphicsScene):
            widget: Optional[HeatmapGridWidget] = None

        self.scene = self.scene = HeatmapScene(parent=self)
        self.view = GraphicsView(
            self.scene,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            viewportUpdateMode=QGraphicsView.FullViewportUpdate,
            widgetResizable=True,
        )
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            self._on_view_context_menu
        )
        self.mainArea.layout().addWidget(self.view)
        self.selected_rows = []
        self.__font_inc = QAction(
            "Increase Font", self, shortcut=QKeySequence("ctrl+>"))
        self.__font_dec = QAction(
            "Decrease Font", self, shortcut=QKeySequence("ctrl+<"))
        self.__font_inc.triggered.connect(lambda: self.__adjust_font_size(1))
        self.__font_dec.triggered.connect(lambda: self.__adjust_font_size(-1))
        if hasattr(QAction, "setShortcutVisibleInContextMenu"):
            apply_all(
                [self.__font_inc, self.__font_dec],
                lambda a: a.setShortcutVisibleInContextMenu(True)
            )
        self.addActions([self.__font_inc, self.__font_dec])

    def _save_state_for_serialization(self):
        def desc(var: Optional[Variable]) -> Optional[Tuple[str, str]]:
            if var is not None:
                return type(var).__name__, var.name
            else:
                return None

        self.col_clustering_method = self.col_clustering.name
        self.row_clustering_method = self.row_clustering.name

        self.column_annotation_color_key = desc(self.column_annotation_color_var)
        self.split_columns_key = desc(self.split_columns_var)

    @property
    def center_palette(self):
        palette = self.color_map_widget.currentData()
        return bool(palette.flags & palette.Diverging)

    @property
    def _column_label_pos(self) -> HeatmapGridWidget.Position:
        return ColumnLabelsPosData[self.column_label_pos][Qt.UserRole]

    def annotation_color_var_changed(self, value):
        cbselect(self.row_side_color_cb, value, Qt.EditRole)

    def annotation_var_changed(self, value):
        cbselect(self.annotation_text_cb, value, Qt.EditRole)

    def set_row_clustering(self, method: Clustering) -> None:
        assert isinstance(method, Clustering)
        if self.row_clustering != method:
            self.row_clustering = method
            cbselect(self.row_cluster_cb, method, ClusteringRole)
            self.__update_row_clustering()

    def set_col_clustering(self, method: Clustering) -> None:
        assert isinstance(method, Clustering)
        if self.col_clustering != method:
            self.col_clustering = method
            cbselect(self.col_cluster_cb, method, ClusteringRole)
            self.__update_column_clustering()

    def sizeHint(self) -> QSize:
        return super().sizeHint().expandedTo(QSize(900, 700))

    def color_palette(self):
        return self.color_map_widget.currentData().lookup_table()

    def color_map(self) -> GradientColorMap:
        return GradientColorMap(
            self.color_palette(), (self.threshold_low, self.threshold_high),
            self.color_map_widget.center() if self.center_palette else None
        )

    def clear(self):
        self.data = None
        self.input_data = None
        self.effective_data = None
        self.kmeans_model = None
        self.merge_indices = None
        self.annotation_model.set_domain(None)
        self.annotation_var = None
        self.row_side_color_model.set_domain(None)
        self.col_side_color_model.set_domain(None)
        self.annotation_color_var = None
        self.column_annotation_color_var = None
        self.row_split_model.set_domain(None)
        self.col_split_model.set_domain(None)
        self.split_by_var = None
        self.split_columns_var = None
        self.parts = None
        self.clear_scene()
        self.selected_rows = []
        self.__columns_cache.clear()
        self.__rows_cache.clear()
        self.__update_clustering_enable_state(None)

    def clear_scene(self):
        if self.scene.widget is not None:
            self.scene.widget.layoutDidActivate.disconnect(
                self.__on_layout_activate
            )
            self.scene.widget.selectionFinished.disconnect(
                self.on_selection_finished
            )
        self.scene.widget = None
        self.scene.clear()

        self.view.setSceneRect(QRectF())
        self.view.setHeaderSceneRect(QRectF())
        self.view.setFooterSceneRect(QRectF())

    @Inputs.data
    def set_dataset(self, data=None):
        """Set the input dataset to display."""
        self.closeContext()
        self.clear()
        self.clear_messages()

        if isinstance(data, SqlTable):
            if data.approx_len() < 4000:
                data = Table(data)
            else:
                self.Information.sampled()
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)

        if data is not None and not len(data):
            data = None

        if data is not None and sp.issparse(data.X):
            try:
                data = data.to_dense()
            except MemoryError:
                data = None
                self.Error.not_enough_memory()
            else:
                self.Information.sparse_densified()

        input_data = data

        # Data contains no attributes or meta attributes only
        if data is not None and len(data.domain.attributes) == 0:
            self.Error.no_continuous()
            input_data = data = None

        # Data contains some discrete attributes which must be filtered
        if data is not None and \
                any(var.is_discrete for var in data.domain.attributes):
            ndisc = sum(var.is_discrete for var in data.domain.attributes)
            data = data.transform(
                Domain([var for var in data.domain.attributes
                        if var.is_continuous],
                       data.domain.class_vars,
                       data.domain.metas))
            if not data.domain.attributes:
                self.Error.no_continuous()
                input_data = data = None
            else:
                self.Information.discrete_ignored(
                    ndisc, "s" if ndisc > 1 else "")

        self.data = data
        self.input_data = input_data

        if data is not None:
            self.annotation_model.set_domain(self.input_data.domain)
            self.row_side_color_model.set_domain(self.input_data.domain)
            self.annotation_var = None
            self.annotation_color_var = None
            self.row_split_model.set_domain(data.domain)
            self.col_annot_data = data.transpose(data[:0].transform(Domain(data.domain.attributes)))
            self.col_split_model.set_domain(self.col_annot_data.domain)
            self.col_side_color_model.set_domain(self.col_annot_data.domain)
            if data.domain.has_discrete_class:
                self.split_by_var = data.domain.class_var
            else:
                self.split_by_var = None
            self.split_columns_var = None
            self.column_annotation_color_var = None
            self.openContext(self.input_data)
            if self.split_by_var not in self.row_split_model:
                self.split_by_var = None

            def match(desc: Tuple[str, str], source: Iterable[Variable]):
                for v in source:
                    if desc == (type(v).__name__, v.name):
                        return v
                return None

            def is_variable(obj):
                return isinstance(obj, Variable)

            if self.split_columns_key is not None:
                self.split_columns_var = match(
                    self.split_columns_key,
                    filter(is_variable, self.col_split_model)
                )

            if self.column_annotation_color_key is not None:
                self.column_annotation_color_var = match(
                    self.column_annotation_color_key,
                    filter(is_variable, self.col_side_color_model)
                )

        self.update_heatmaps()
        if data is not None and self.__pending_selection is not None:
            if self.scene.widget is not None:
                self.scene.widget.selectRows(self.__pending_selection)
            self.selected_rows = self.__pending_selection
            self.__pending_selection = None

        self.commit.now()

    def __on_split_rows_activated(self):
        self.set_split_variable(self.row_split_cb.currentData(Qt.EditRole))

    def set_split_variable(self, var):
        if var is not self.split_by_var:
            self.split_by_var = var
            self.update_heatmaps()

    def __on_split_cols_activated(self):
        self.set_column_split_var(self.col_split_cb.currentData(Qt.EditRole))

    def set_column_split_var(self, var: Optional[Variable]):
        if var is not self.split_columns_var:
            self.split_columns_var = var
            self.update_heatmaps()

    def update_heatmaps(self):
        if self.data is not None:
            self.clear_scene()
            self.clear_messages()
            if self.col_clustering != Clustering.None_ and \
                    len(self.data.domain.attributes) < 2:
                self.Error.not_enough_features()
            elif (self.col_clustering != Clustering.None_ or
                  self.row_clustering != Clustering.None_) and \
                    len(self.data) < 2:
                self.Error.not_enough_instances()
            elif self.merge_kmeans and len(self.data) < 3:
                self.Error.not_enough_instances_k_means()
            else:
                parts = self.construct_heatmaps(self.data, self.split_by_var, self.split_columns_var)
                self.construct_heatmaps_scene(parts, self.effective_data)
                self.selected_rows = []
        else:
            self.clear()

    def update_merge(self):
        self.kmeans_model = None
        self.merge_indices = None
        if self.data is not None and self.merge_kmeans:
            self.update_heatmaps()
            self.commit.deferred()

    def _make_parts(self, data, group_var=None, column_split_key=None):
        """
        Make initial `Parts` for data, split by group_var, group_key
        """
        if group_var is not None:
            assert group_var.is_discrete
            _col_data = table_column_data(data, group_var)
            row_indices = [np.flatnonzero(_col_data == i)
                           for i in range(len(group_var.values))]

            row_groups = [RowPart(title=name, indices=ind,
                                  cluster=None, cluster_ordered=None)
                          for name, ind in zip(group_var.values, row_indices)]
            if np.any(_col_data.mask):
                row_groups.append(RowPart(
                    title="N/A", indices=np.flatnonzero(_col_data.mask),
                    cluster=None, cluster_ordered=None
                ))
        else:
            row_groups = [RowPart(title=None, indices=range(0, len(data)),
                                  cluster=None, cluster_ordered=None)]

        if column_split_key is not None:
            col_groups = split_domain(data.domain, column_split_key)
            assert len(col_groups) > 0
            col_indices = [np.array([data.domain.index(var) for var in group])
                           for _, group in col_groups]
            col_groups = [ColumnPart(title=str(name), domain=d, indices=ind,
                                     cluster=None, cluster_ordered=None)
                          for (name, d), ind in zip(col_groups, col_indices)]
        else:
            col_groups = [
                ColumnPart(
                    title=None, indices=range(0, len(data.domain.attributes)),
                    domain=data.domain.attributes, cluster=None, cluster_ordered=None)
            ]

        minv, maxv = np.nanmin(data.X), np.nanmax(data.X)
        return Parts(row_groups, col_groups, span=(minv, maxv))

    def cluster_rows(self, data: Table, parts: 'Parts', ordered=False) -> 'Parts':
        row_groups = []
        for row in parts.rows:
            if row.cluster is not None:
                cluster = row.cluster
            else:
                cluster = None
            if row.cluster_ordered is not None:
                cluster_ord = row.cluster_ordered
            else:
                cluster_ord = None

            if row.can_cluster:
                matrix = None
                need_dist = cluster is None or (ordered and cluster_ord is None)
                if need_dist:
                    subset = data[row.indices]
                    matrix = Orange.distance.Euclidean(subset)

                if cluster is None:
                    cluster = hierarchical.dist_matrix_clustering(
                        matrix, linkage=hierarchical.WARD
                    )
                if ordered and cluster_ord is None:
                    cluster_ord = hierarchical.optimal_leaf_ordering(
                        cluster, matrix,
                    )
            row_groups.append(row._replace(cluster=cluster, cluster_ordered=cluster_ord))

        return parts._replace(rows=row_groups)

    def cluster_columns(self, data, parts: 'Parts', ordered=False):
        assert all(var.is_continuous for var in data.domain.attributes)
        col_groups = []
        for col in parts.columns:
            if col.cluster is not None:
                cluster = col.cluster
            else:
                cluster = None
            if col.cluster_ordered is not None:
                cluster_ord = col.cluster_ordered
            else:
                cluster_ord = None
            if col.can_cluster:
                need_dist = cluster is None or (ordered and cluster_ord is None)
                matrix = None
                if need_dist:
                    subset = data.transform(Domain(col.domain))
                    subset = Orange.distance._preprocess(subset)
                    matrix = np.asarray(Orange.distance.PearsonR(subset, axis=0))
                    # nan values break clustering below
                    matrix = np.nan_to_num(matrix)

                if cluster is None:
                    assert matrix is not None
                    cluster = hierarchical.dist_matrix_clustering(
                        matrix, linkage=hierarchical.WARD
                    )
                if ordered and cluster_ord is None:
                    cluster_ord = hierarchical.optimal_leaf_ordering(cluster, matrix)

            col_groups.append(col._replace(cluster=cluster, cluster_ordered=cluster_ord))
        return parts._replace(columns=col_groups)

    def construct_heatmaps(self, data, group_var=None, column_split_key=None) -> 'Parts':
        if self.merge_kmeans:
            if self.kmeans_model is None:
                effective_data = self.input_data.transform(
                    Orange.data.Domain(
                        [var for var in self.input_data.domain.attributes
                         if var.is_continuous],
                        self.input_data.domain.class_vars,
                        self.input_data.domain.metas))
                nclust = min(self.merge_kmeans_k, len(effective_data) - 1)
                self.kmeans_model = kmeans_compress(effective_data, k=nclust)
                effective_data.domain = self.kmeans_model.domain
                merge_indices = [np.flatnonzero(self.kmeans_model.labels == ind)
                                 for ind in range(nclust)]
                not_empty_indices = [i for i, x in enumerate(merge_indices)
                                     if len(x) > 0]
                self.merge_indices = \
                    [merge_indices[i] for i in not_empty_indices]
                if len(merge_indices) != len(self.merge_indices):
                    self.Warning.empty_clusters()
                effective_data = Orange.data.Table(
                    Orange.data.Domain(effective_data.domain.attributes),
                    self.kmeans_model.centroids[not_empty_indices]
                )
            else:
                effective_data = self.effective_data

            group_var = None
        else:
            self.kmeans_model = None
            self.merge_indices = None
            effective_data = data

        self.effective_data = effective_data

        parts = self._make_parts(
            effective_data, group_var,
            column_split_key.name if column_split_key is not None else None)

        self.__update_clustering_enable_state(parts)
        # Restore/update the row/columns items descriptions from cache if
        # available
        rows_cache_key = (group_var,
                          self.merge_kmeans_k if self.merge_kmeans else None)
        if rows_cache_key in self.__rows_cache:
            parts = parts._replace(rows=self.__rows_cache[rows_cache_key].rows)

        if column_split_key in self.__columns_cache:
            parts = parts._replace(
                columns=self.__columns_cache[column_split_key].columns)

        if self.row_clustering != Clustering.None_:
            parts = self.cluster_rows(
                effective_data, parts,
                ordered=self.row_clustering == Clustering.OrderedClustering
            )
        if self.col_clustering != Clustering.None_:
            parts = self.cluster_columns(
                effective_data, parts,
                ordered=self.col_clustering == Clustering.OrderedClustering
            )

        # Cache the updated parts
        self.__rows_cache[rows_cache_key] = parts
        return parts

    def construct_heatmaps_scene(self, parts: 'Parts', data: Table) -> None:
        _T = TypeVar("_T", bound=Union[RowPart, ColumnPart])

        def select_cluster(clustering: Clustering, item: _T) -> _T:
            if clustering == Clustering.None_:
                return item._replace(cluster=None, cluster_ordered=None)
            elif clustering == Clustering.Clustering:
                return item._replace(cluster=item.cluster, cluster_ordered=None)
            elif clustering == Clustering.OrderedClustering:
                return item._replace(cluster=item.cluster_ordered, cluster_ordered=None)
            else:  # pragma: no cover
                raise TypeError()

        rows = [select_cluster(self.row_clustering, rowitem)
                for rowitem in parts.rows]
        cols = [select_cluster(self.col_clustering, colitem)
                for colitem in parts.columns]
        parts = Parts(columns=cols, rows=rows, span=parts.span)

        self.setup_scene(parts, data)

    def setup_scene(self, parts, data):
        # type: (Parts, Table) -> None
        widget = HeatmapGridWidget()
        widget.setColorMap(self.color_map())
        self.scene.addItem(widget)
        self.scene.widget = widget
        columns = [v.name for v in data.domain.attributes]
        parts = HeatmapGridWidget.Parts(
            rows=[
                HeatmapGridWidget.RowItem(r.title, r.indices, r.cluster)
                for r in parts.rows
            ],
            columns=[
                HeatmapGridWidget.ColumnItem(c.title, c.indices, c.cluster)
                for c in parts.columns
            ],
            data=data.X,
            span=parts.span,
            row_names=None,
            col_names=columns,
        )
        widget.setHeatmaps(parts)

        side = self.row_side_colors()
        if side is not None:
            widget.setRowSideColorAnnotations(side[0], side[1], name=side[2].name)

        side = self.column_side_colors()
        if side is not None:
            widget.setColumnSideColorAnnotations(side[0], side[1], name=side[2].name)

        widget.setColumnLabelsPosition(self._column_label_pos)
        widget.setAspectRatioMode(
            Qt.KeepAspectRatio if self.keep_aspect else Qt.IgnoreAspectRatio
        )
        widget.setShowAverages(self.averages)
        widget.setLegendVisible(self.legend)

        widget.layoutDidActivate.connect(self.__on_layout_activate)
        widget.selectionFinished.connect(self.on_selection_finished)

        self.update_annotations()
        self.view.setCentralWidget(widget)
        self.parts = parts

    def __update_scene_rects(self):
        widget = self.scene.widget
        if widget is None:
            return
        rect = widget.geometry()
        self.scene.setSceneRect(rect)
        self.view.setSceneRect(rect)
        self.view.setHeaderSceneRect(widget.headerGeometry())
        self.view.setFooterSceneRect(widget.footerGeometry())

    def __on_layout_activate(self):
        self.__update_scene_rects()

    def __aspect_mode_changed(self):
        widget = self.scene.widget
        if widget is None:
            return
        widget.setAspectRatioMode(
            Qt.KeepAspectRatio if self.keep_aspect else Qt.IgnoreAspectRatio
        )
        # when aspect fixed the vertical sh is fixex, when not, it can
        # shrink vertically
        sp = widget.sizePolicy()
        if self.keep_aspect:
            sp.setVerticalPolicy(QSizePolicy.Fixed)
        else:
            sp.setVerticalPolicy(QSizePolicy.Preferred)
        widget.setSizePolicy(sp)

    def __update_clustering_enable_state(self, parts: Optional['Parts']):
        def c_cost(sizes: Iterable[int]) -> int:
            """Estimated cost for clustering of `sizes`"""
            return sum(n ** 2 for n in sizes)

        def co_cost(sizes: Iterable[int]) -> int:
            """Estimated cost for cluster ordering of `sizes`"""
            # ~O(N ** 3) but O(N ** 4) worst case.
            return sum(n ** 4 for n in sizes)

        if parts is not None:
            Ns = [len(p.indices) for p in parts.rows]
            Ms = [len(p.indices) for p in parts.columns]
        else:
            Ns = Ms = [0]

        rc_enabled = c_cost(Ns) <= c_cost([self.MaxClustering])
        rco_enabled = co_cost(Ns) <= co_cost([self.MaxOrderedClustering])
        cc_enabled = c_cost(Ms) <= c_cost([self.MaxClustering])
        cco_enabled = co_cost(Ms) <= co_cost([self.MaxOrderedClustering])
        row_clust, col_clust = self.row_clustering, self.col_clustering

        row_clust_msg = ""
        col_clust_msg = ""

        if not rco_enabled and row_clust == Clustering.OrderedClustering:
            row_clust = Clustering.Clustering
            row_clust_msg = "Row cluster ordering was disabled due to the " \
                            "estimated runtime cost"
        if not rc_enabled and row_clust == Clustering.Clustering:
            row_clust = Clustering.None_
            row_clust_msg = "Row clustering was was disabled due to the " \
                            "estimated runtime cost"

        if not cco_enabled and col_clust == Clustering.OrderedClustering:
            col_clust = Clustering.Clustering
            col_clust_msg = "Column cluster ordering was disabled due to " \
                            "estimated runtime cost"
        if not cc_enabled and col_clust == Clustering.Clustering:
            col_clust = Clustering.None_
            col_clust_msg = "Column clustering was disabled due to the " \
                            "estimated runtime cost"

        self.col_clustering = col_clust
        self.row_clustering = row_clust

        self.Information.row_clust(row_clust_msg, shown=bool(row_clust_msg))
        self.Information.col_clust(col_clust_msg, shown=bool(col_clust_msg))

        # Disable/enable the combobox items for the clustering methods
        def setenabled(cb: QComboBox, clu: bool, clu_op: bool):
            model = cb.model()
            assert isinstance(model, QStandardItemModel)
            idx = cb.findData(Clustering.OrderedClustering, ClusteringRole)
            assert idx != -1
            model.item(idx).setEnabled(clu_op)
            idx = cb.findData(Clustering.Clustering, ClusteringRole)
            assert idx != -1
            model.item(idx).setEnabled(clu)

        setenabled(self.row_cluster_cb, rc_enabled, rco_enabled)
        setenabled(self.col_cluster_cb, cc_enabled, cco_enabled)

    def update_averages_stripe(self):
        """Update the visibility of the averages stripe.
        """
        widget = self.scene.widget
        if widget is not None:
            widget.setShowAverages(self.averages)

    def update_color_schema(self):
        self.palette_name = self.color_map_widget.currentData().name
        w = self.scene.widget
        if w is not None:
            w.setColorMap(self.color_map())

    def __update_column_clustering(self):
        self.update_heatmaps()
        self.commit.deferred()

    def __update_row_clustering(self):
        self.update_heatmaps()
        self.commit.deferred()

    def update_legend(self):
        widget = self.scene.widget
        if widget is not None:
            widget.setLegendVisible(self.legend)

    def row_annotation_var(self):
        return self.annotation_var

    def row_annotation_data(self):
        var = self.row_annotation_var()
        if var is None:
            return None
        return column_str_from_table(self.input_data, var)

    def _merge_row_indices(self):
        if self.merge_kmeans and self.kmeans_model is not None:
            return self.merge_indices
        else:
            return None

    def set_annotation_var(self, var: Union[None, Variable, int]):
        if isinstance(var, int):
            var = self.annotation_model[var]
        if self.annotation_var is not var:
            self.annotation_var = var
            self.update_annotations()

    def update_annotations(self):
        widget = self.scene.widget
        if widget is not None:
            annot_col = self.row_annotation_data()
            merge_indices = self._merge_row_indices()
            if merge_indices is not None and annot_col is not None:
                join = lambda _1: join_elided(", ", 42, _1, " ({} more)")
                annot_col = aggregate_apply(join, annot_col, merge_indices)
            if annot_col is not None:
                widget.setRowLabels(annot_col)
                widget.setRowLabelsVisible(True)
            else:
                widget.setRowLabelsVisible(False)
                widget.setRowLabels(None)

    def row_side_colors(self):
        var = self.annotation_color_var
        if var is None:
            return None
        column_data = column_data_from_table(self.input_data, var)
        merges = self._merge_row_indices()
        if merges is not None:
            column_data = aggregate(var, column_data, merges)
        data, colormap = colorize(var, column_data)
        if var.is_continuous:
            span = (np.nanmin(column_data), np.nanmax(column_data))
            if np.any(np.isnan(span)):
                span = 0., 1.
            colormap.span = span
        return data, colormap, var

    def set_annotation_color_var(self, var: Union[None, Variable, int]):
        """Set the current side color annotation variable."""
        if isinstance(var, int):
            var = self.row_side_color_model[var]
        if self.annotation_color_var is not var:
            self.annotation_color_var = var
            self.update_row_side_colors()

    def update_row_side_colors(self):
        widget = self.scene.widget
        if widget is None:
            return
        colors = self.row_side_colors()
        if colors is None:
            widget.setRowSideColorAnnotations(None)
        else:
            widget.setRowSideColorAnnotations(colors[0], colors[1], colors[2].name)

    def __set_column_annotation_color_var_index(self, index: int):
        key = self.col_side_color_cb.itemData(index, Qt.EditRole)
        self.set_column_annotation_color_var(key)

    def column_annotation_color_var_changed(self, value):
        cbselect(self.col_side_color_cb, value, Qt.EditRole)

    def set_column_annotation_color_var(self, var):
        if self.column_annotation_color_var is not var:
            self.column_annotation_color_var = var
            colors = self.column_side_colors()
            if colors is not None:
                self.scene.widget.setColumnSideColorAnnotations(
                    colors[0], colors[1], colors[2].name,
                )
            else:
                self.scene.widget.setColumnSideColorAnnotations(None)

    def column_side_colors(self):
        var = self.column_annotation_color_var
        if var is None:
            return None
        table = self.col_annot_data
        return color_annotation_data(table, var)

    def update_column_annotations(self):
        widget = self.scene.widget
        if self.data is not None and widget is not None:
            widget.setColumnLabelsPosition(self._column_label_pos)

    def __adjust_font_size(self, diff):
        widget = self.scene.widget
        if widget is None:
            return
        curr = widget.font().pointSizeF()
        new = curr + diff

        self.__font_dec.setEnabled(new > 1.0)
        self.__font_inc.setEnabled(new <= 32)
        if new > 1.0:
            font = QFont()
            font.setPointSizeF(new)
            widget.setFont(font)

    def _on_view_context_menu(self, pos):
        widget = self.scene.widget
        if widget is None:
            return
        assert isinstance(widget, HeatmapGridWidget)
        menu = QMenu(self.view.viewport())
        menu.setAttribute(Qt.WA_DeleteOnClose)
        menu.addActions(self.view.actions())
        menu.addSeparator()
        menu.addActions([self.__font_inc, self.__font_dec])
        menu.addSeparator()
        a = QAction("Keep aspect ratio", menu, checkable=True)
        a.setChecked(self.keep_aspect)

        def ontoggled(state):
            self.keep_aspect = state
            self.__aspect_mode_changed()
        a.toggled.connect(ontoggled)
        menu.addAction(a)
        menu.popup(self.view.viewport().mapToGlobal(pos))

    def on_selection_finished(self):
        if self.scene.widget is not None:
            self.selected_rows = list(self.scene.widget.selectedRows())
        else:
            self.selected_rows = []
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        data = None
        indices = None
        if self.merge_kmeans:
            merge_indices = self.merge_indices
        else:
            merge_indices = None

        if self.input_data is not None and self.selected_rows:
            indices = self.selected_rows
            if merge_indices is not None:
                # expand merged indices
                indices = np.hstack([merge_indices[i] for i in indices])

            data = self.input_data[indices]

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(create_annotated_table(self.input_data, indices))

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    def send_report(self):
        self.report_items((
            ("Columns:", "Clustering" if self.col_clustering else "No sorting"),
            ("Rows:", "Clustering" if self.row_clustering else "No sorting"),
            ("Split:",
             self.split_by_var is not None and self.split_by_var.name),
            ("Row annotation",
             self.annotation_var is not None and self.annotation_var.name),
        ))
        self.report_plot()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is not None and version < 3:
            def st2cl(state: bool) -> Clustering:
                return Clustering.OrderedClustering if state else \
                    Clustering.None_

            rc = settings.pop("row_clustering", False)
            cc = settings.pop("col_clustering", False)
            settings["row_clustering_method"] = st2cl(rc).name
            settings["col_clustering_method"] = st2cl(cc).name


# If StickyGraphicsView ever defines qt signals/slots/properties this will
# break
class GraphicsView(GraphicsWidgetView, StickyGraphicsView):
    pass


class RowPart(NamedTuple):
    """
    A row group

    Attributes
    ----------
    title: str
        Group title
    indices : (N, ) Sequence[int]
        Indices in the input data to retrieve the row subset for the group.
    cluster : hierarchical.Tree optional
    cluster_ordered : hierarchical.Tree optional
    """
    title: str
    indices: Sequence[int]
    cluster: Optional[hierarchical.Tree] = None
    cluster_ordered: Optional[hierarchical.Tree] = None

    @property
    def can_cluster(self) -> bool:
        if isinstance(self.indices, slice):
            return (self.indices.stop - self.indices.start) > 1
        else:
            return len(self.indices) > 1


class ColumnPart(NamedTuple):
    """
    A column group

    Attributes
    ----------
    title : str
        Column group title
    indices : (N, ) int ndarray
        Indexes the input data to retrieve the column subset for the group.
    domain : List[Variable]
        List of variables in the group.
    cluster : hierarchical.Tree optional
    cluster_ordered : hierarchical.Tree optional
    """
    title: str
    indices: Sequence[int]
    domain: Sequence[int]
    cluster: Optional[hierarchical.Tree] = None
    cluster_ordered: Optional[hierarchical.Tree] = None

    @property
    def can_cluster(self) -> bool:
        if isinstance(self.indices, slice):
            return (self.indices.stop - self.indices.start) > 1
        else:
            return len(self.indices) > 1


class Parts(NamedTuple):
    rows: Sequence[RowPart]
    columns: Sequence[ColumnPart]
    span: Tuple[float, float]


def join_elided(sep, maxlen, values, elidetemplate="..."):
    def generate(sep, ellidetemplate, values):
        count = len(values)
        length = 0
        parts = []
        for i, val in enumerate(values):
            elide = ellidetemplate.format(count - i) if count - i > 1 else ""
            parts.append(val)
            length += len(val) + (len(sep) if parts else 0)
            yield i, islice(parts, i + 1), length, elide

    best = None
    for _, parts, length, elide in generate(sep, elidetemplate, values):
        if length > maxlen:
            if best is None:
                best = sep.join(parts) + elide
            return best
        fulllen = length + len(elide)
        if fulllen < maxlen or best is None:
            best = sep.join(parts) + elide
    return best


def column_str_from_table(
        table: Orange.data.Table,
        column: Union[int, Orange.data.Variable],
) -> np.ndarray:
    var = table.domain[column]
    data, _ = table.get_column_view(column)
    return np.asarray([var.str_val(v) for v in data], dtype=object)


def column_data_from_table(
        table: Orange.data.Table,
        column: Union[int, Orange.data.Variable],
) -> np.ndarray:
    var = table.domain[column]
    data, _ = table.get_column_view(column)
    if var.is_primitive() and data.dtype.kind != "f":
        data = data.astype(float)
    return data


def color_annotation_data(
        table: Table, var: Union[int, str, Variable]
) -> Tuple[np.ndarray, ColorMap, Variable]:
    var = table.domain[var]
    column_data = column_data_from_table(table, var)
    data, colormap = colorize(var, column_data)
    return data, colormap, var


def colorize(var: Variable, data: np.ndarray) -> Tuple[np.ndarray, ColorMap]:
    palette = var.palette  # type: Palette
    colors = np.array(
        [[c.red(), c.green(), c.blue()] for c in palette.qcolors_w_nan],
        dtype=np.uint8,
    )
    if var.is_discrete:
        mask = np.isnan(data)
        data = data.astype(int)
        data[mask] = -1
        if mask.any():
            values = (*var.values, "N/A")
        else:
            values = var.values
            colors = colors[: -1]
        return data, CategoricalColorMap(colors, values)
    elif var.is_continuous:
        span = np.nanmin(data), np.nanmax(data)
        if np.any(np.isnan(span)):
            span = 0, 1.
        return data, GradientColorMap(colors[:-1], span=span)
    else:
        raise TypeError


def aggregate(
        var: Variable, data: np.ndarray, groupindices: Sequence[Sequence[int]],
) -> np.ndarray:
    if var.is_string:
        join = lambda values: (join_elided(", ", 42, values, " ({} more)"))
        # collect all original labels for every merged row
        values = [data[indices] for indices in groupindices]
        data = [join(list(map(var.str_val, vals))) for vals in values]
        return np.array(data, dtype=object)
    elif var.is_continuous:
        data = [np.nanmean(data[indices]) if len(indices) else np.nan
                for indices in groupindices]
        return np.array(data, dtype=float)
    elif var.is_discrete:
        from Orange.statistics.util import nanmode
        data = [nanmode(data[indices])[0] if len(indices) else np.nan
                for indices in groupindices]
        return np.asarray(data, dtype=float)
    else:
        raise TypeError(type(var))


def agg_join_str(var, data, groupindices, maxlen=50, elidetemplate=" ({} more)"):
    join_s = lambda values: (
        join_elided(", ", maxlen, values, elidetemplate=elidetemplate)
    )
    join = lambda values: join_s(map(var.str_val, values))
    return aggregate_apply(join, data, groupindices)


_T = TypeVar("_T")


def aggregate_apply(
        f: Callable[[Sequence], _T],
        data: np.ndarray,
        groupindices: Sequence[Sequence[int]]
) -> Sequence[_T]:
    return [f(data[indices]) for indices in groupindices]


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWHeatMap).run(Table("brown-selected.tab"))
