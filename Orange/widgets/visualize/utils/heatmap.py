import math
import enum
from itertools import chain, zip_longest

from typing import (
    Optional, List, NamedTuple, Sequence, Tuple, Dict, Union, Iterable
)

import numpy as np

from AnyQt.QtCore import (
    Signal, Property, Qt, QRectF, QSizeF, QEvent, QPointF, QObject
)
from AnyQt.QtGui import QPixmap, QPalette, QPen, QColor, QFontMetrics
from AnyQt.QtWidgets import (
    QGraphicsWidget, QSizePolicy, QGraphicsGridLayout, QGraphicsRectItem,
    QApplication, QGraphicsSceneMouseEvent, QGraphicsLinearLayout,
    QGraphicsItem, QGraphicsSimpleTextItem, QGraphicsLayout,
    QGraphicsLayoutItem
)

import pyqtgraph as pg

from Orange.clustering import hierarchical
from Orange.clustering.hierarchical import Tree
from Orange.widgets.utils import apply_all
from Orange.widgets.utils.colorpalettes import DefaultContinuousPalette
from Orange.widgets.utils.graphicslayoutitem import SimpleLayoutItem, scaled
from Orange.widgets.utils.graphicsflowlayout import GraphicsFlowLayout
from Orange.widgets.utils.graphicspixmapwidget import GraphicsPixmapWidget
from Orange.widgets.utils.image import qimage_from_array

from Orange.widgets.utils.graphicstextlist import TextListWidget
from Orange.widgets.utils.dendrogram import DendrogramWidget


def leaf_indices(tree: Tree) -> Sequence[int]:
    return [leaf.value.index for leaf in hierarchical.leaves(tree)]


class ColorMap:
    """Base color map class."""

    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def replace(self, **kwargs) -> 'ColorMap':
        raise NotImplementedError


class CategoricalColorMap(ColorMap):
    """A categorical color map."""
    #: A color table. A (N, 3) uint8 ndarray
    colortable: np.ndarray
    #: A N sequence of categorical names
    names: Sequence[str]

    def __init__(self, colortable, names):
        self.colortable = np.asarray(colortable)
        self.names = names
        assert len(colortable) == len(names)

    def apply(self, data) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        table = self.colortable[data]
        return table

    def replace(self, **kwargs) -> 'CategoricalColorMap':
        kwargs.setdefault("colortable", self.colortable)
        kwargs.setdefault("names", self.names)
        return CategoricalColorMap(**kwargs)


class GradientColorMap(ColorMap):
    """Color map for the heatmap."""
    #: A color table. A (N, 3) uint8 ndarray
    colortable: np.ndarray
    #: The data range (min, max)
    span: Optional[Tuple[float, float]] = None
    #: Lower and upper thresholding operator parameters. Expressed as relative
    #: to the data span (range) so (0, 1) applies no thresholding, while
    #: (0.05, 0.95) squeezes the effective range by 5% from both ends
    thresholds: Tuple[float, float] = (0., 1.)
    #: Should the color map be center and if so around which value.
    center: Optional[float] = None

    def __init__(self, colortable, thresholds=thresholds, center=None, span=None):
        self.colortable = np.asarray(colortable)
        self.thresholds = thresholds
        assert thresholds[0] <= thresholds[1]
        self.center = center
        self.span = span

    def adjust_levels(self, low: float, high: float) -> Tuple[float, float]:
        """
        Adjust the data low, high levels by applying the thresholding and
        centering.
        """
        if np.any(np.isnan([low, high])):
            return np.nan, np.nan
        elif low > high:
            raise ValueError(f"low > high ({low} > {high})")
        threshold_low, threshold_high = self.thresholds
        lt = low + (high - low) * threshold_low
        ht = low + (high - low) * threshold_high
        if self.center is not None:
            center = self.center
            maxoff = max(abs(center - lt), abs(center - ht))
            lt = center - maxoff
            ht = center + maxoff
        return lt, ht

    def apply(self, data) -> np.ndarray:
        if self.span is None:
            low, high = np.nanmin(data), np.nanmax(data)
        else:
            low, high = self.span
        low, high = self.adjust_levels(low, high)
        mask = np.isnan(data)
        normalized = data - low
        finfo = np.finfo(normalized.dtype)
        if high - low <= 1 / finfo.max:
            n_fact = finfo.max
        else:
            n_fact = 1. / (high - low)
        # over/underflow to inf are expected and cliped with the rest in the
        # next step
        with np.errstate(over="ignore", under="ignore"):
            normalized *= n_fact
        normalized = np.clip(normalized, 0, 1, out=normalized)
        table = np.empty_like(normalized, dtype=np.uint8)
        ncolors = len(self.colortable)
        assert ncolors - 1 <= np.iinfo(table.dtype).max
        table = np.multiply(
            normalized, ncolors - 1, out=table, where=~mask, casting="unsafe",
        )
        colors = self.colortable[table]
        colors[mask] = 0
        return colors

    def replace(self, **kwargs) -> 'GradientColorMap':
        kwargs.setdefault("colortable", self.colortable)
        kwargs.setdefault("thresholds", self.thresholds)
        kwargs.setdefault("center", self.center)
        kwargs.setdefault("span", self.span)
        return GradientColorMap(**kwargs)


def normalized_indices(item: Union['RowItem', 'ColumnItem']) -> np.ndarray:
    if item.cluster is None:
        return np.asarray(item.indices, dtype=int)
    else:
        reorder = np.array(leaf_indices(item.cluster), dtype=int)
        indices = np.asarray(item.indices, dtype=int)
        return indices[reorder]


class GridLayout(QGraphicsGridLayout):
    def setGeometry(self, rect: QRectF) -> None:
        super().setGeometry(rect)
        parent = self.parentLayoutItem()
        if isinstance(parent, HeatmapGridWidget):
            parent.layoutDidActivate.emit()


def grid_layout_row_geometry(layout: QGraphicsGridLayout, row: int) -> QRectF:
    """
    Return the geometry of the `row` in the grid layout.

    If the row is empty return an empty geometry
    """
    if not 0 <= row < layout.rowCount():
        return QRectF()

    columns = layout.columnCount()
    geometries: List[QRectF] = []
    for item in (layout.itemAt(row, column) for column in range(columns)):
        if item is not None:
            itemgeom = item.geometry()
            if itemgeom.isValid():
                geometries.append(itemgeom)
    if geometries:
        rect = layout.geometry()
        rect.setTop(min(g.top() for g in geometries))
        rect.setBottom(max(g.bottom() for g in geometries))
        return rect
    else:
        return QRectF()


# Positions
class Position(enum.IntFlag):
    NoPosition = 0
    Left, Top, Right, Bottom = 1, 2, 4, 8


Left, Right = Position.Left, Position.Right
Top, Bottom = Position.Top, Position.Bottom


FLT_MAX = np.finfo(np.float32).max


class HeatmapGridWidget(QGraphicsWidget):
    """
    A graphics widget with a annotated 2D grid of heatmaps.
    """
    class RowItem(NamedTuple):
        """
        A row group item

        Attributes
        ----------
        title: str
            Group title
        indices : (N, ) Sequence[int]
            Indices in the input data to retrieve the row subset for the group.
        cluster : Optional[Tree]

        """
        title: str
        indices: Sequence[int]
        cluster: Optional[Tree] = None

        @property
        def size(self):
            return len(self.indices)

        @property
        def normalized_indices(self):
            return normalized_indices(self)

    class ColumnItem(NamedTuple):
        """
        A column group

        Attributes
        ----------
        title: str
            Column group title
        indices: (N, ) Sequence[int]
            Indexes the input data to retrieve the column subset for the group.
        cluster: Optional[Tree]
        """
        title: str
        indices: Sequence[int]
        cluster: Optional[Tree] = None

        @property
        def size(self):
            return len(self.indices)

        @property
        def normalized_indices(self):
            return normalized_indices(self)

    class Parts(NamedTuple):
        #: define the splits of data over rows, and define dendrogram and/or row
        #: reordering
        rows: Sequence['RowItem']
        #: define the splits of data over columns, and define dendrogram and/or
        #: column reordering
        columns: Sequence['ColumnItem']
        #: span (min, max) of the values in `data`
        span: Tuple[float, float]
        #: the complete data array (shape (N, M))
        data: np.ndarray
        #: Row names (len N)
        row_names: Optional[Sequence[str]] = None
        #: Column names (len M)
        col_names: Optional[Sequence[str]] = None

    # Positions
    class Position(enum.IntFlag):
        NoPosition = 0
        Left, Top, Right, Bottom = 1, 2, 4, 8

    Left, Right = Position.Left, Position.Right
    Top, Bottom = Position.Top, Position.Bottom

    #: The widget's layout has activated (i.e. did a relayout
    #: of the widget's contents)
    layoutDidActivate = Signal()

    #: Signal emitted when the user finished a selection operation
    selectionFinished = Signal()
    #: Signal emitted on any change in selection
    selectionChanged = Signal()

    NoPosition, PositionTop, PositionBottom = 0, Top, Bottom

    # Start row/column where the heatmap items are inserted
    # (after the titles/legends/dendrograms)
    Row0 = 5
    Col0 = 3
    # The (color) legend row and column
    LegendRow, LegendCol = 0, 4
    # The column for the vertical dendrogram
    DendrogramColumn = 1
    # Horizontal split title column
    GroupTitleRow = 1
    # The row for the horizontal dendrograms
    DendrogramRow = 2
    # The row for top column annotation labels
    TopLabelsRow = 3
    # Top color annotation row
    TopAnnotationRow = 4
    # Vertical split title column
    GroupTitleColumn = 0

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__spacing = 3
        self.__colormap = GradientColorMap(
            DefaultContinuousPalette.lookup_table()
        )
        self.parts = None  # type: Optional[Parts]
        self.__averagesVisible = False
        self.__legendVisible = True
        self.__aspectRatioMode = Qt.IgnoreAspectRatio
        self.__columnLabelPosition = Top
        self.heatmap_widget_grid = []  # type: List[List[GraphicsHeatmapWidget]]
        self.row_annotation_widgets = []  # type: List[TextListWidget]
        self.col_annotation_widgets = []  # type: List[TextListWidget]
        self.col_annotation_widgets_top = []  # type: List[TextListWidget]
        self.col_annotation_widgets_bottom = []  # type: List[TextListWidget]
        self.col_dendrograms = []  # type: List[Optional[DendrogramWidget]]
        self.row_dendrograms = []  # type: List[Optional[DendrogramWidget]]
        self.right_side_colors = []  # type: List[Optional[GraphicsPixmapWidget]]
        self.top_side_colors = []  # type: List[Optional[GraphicsPixmapWidget]]
        self.heatmap_colormap_legend = None
        self.bottom_legend_container = None
        self.__layout = GridLayout()
        self.__layout.setSpacing(self.__spacing)
        self.setLayout(self.__layout)
        self.__selection_manager = SelectionManager(self)
        self.__selection_manager.selection_changed.connect(
            self.__update_selection_geometry
        )
        self.__selection_manager.selection_finished.connect(
            self.selectionFinished
        )
        self.__selection_manager.selection_changed.connect(
            self.selectionChanged
        )
        self.selection_rects = []

    def clear(self):
        """Clear the widget."""
        for i in reversed(range(self.__layout.count())):
            item = self.__layout.itemAt(i)
            self.__layout.removeAt(i)
            if item is not None and item.graphicsItem() is not None:
                remove_item(item.graphicsItem())

        self.heatmap_widget_grid = []
        self.row_annotation_widgets = []
        self.col_annotation_widgets = []
        self.col_dendrograms = []
        self.row_dendrograms = []
        self.right_side_colors = []
        self.top_side_colors = []
        self.heatmap_colormap_legend = None
        self.bottom_legend_container = None
        self.parts = None
        self.updateGeometry()

    def setHeatmaps(self, parts: 'Parts') -> None:
        """Set the heatmap parts for display"""
        self.clear()
        grid = self.__layout
        N, M = len(parts.rows), len(parts.columns)

        # Start row/column where the heatmap items are inserted
        # (after the titles/legends/dendrograms)
        Row0 = self.Row0
        Col0 = self.Col0
        # The column for the vertical dendrograms
        DendrogramColumn = self.DendrogramColumn
        # The row for the horizontal dendrograms
        DendrogramRow = self.DendrogramRow
        RightLabelColumn = Col0 + 2 * M + 1
        TopAnnotationRow = self.TopAnnotationRow
        TopLabelsRow = self.TopLabelsRow
        BottomLabelsRow = Row0 + N
        colormap = self.__colormap
        column_dendrograms: List[Optional[DendrogramWidget]] = [None] * M
        row_dendrograms: List[Optional[DendrogramWidget]] = [None] * N
        right_side_colors: List[Optional[GraphicsPixmapWidget]] = [None] * N
        top_side_colors: List[Optional[GraphicsPixmapWidget]] = [None] * M

        data = parts.data
        if parts.col_names is None:
            col_names = np.full(data.shape[1], "", dtype=object)
        else:
            col_names = np.asarray(parts.col_names, dtype=object)
        if parts.row_names is None:
            row_names = np.full(data.shape[0], "", dtype=object)
        else:
            row_names = np.asarray(parts.row_names, dtype=object)

        assert len(col_names) == data.shape[1]
        assert len(row_names) == data.shape[0]

        for i, rowitem in enumerate(parts.rows):
            if rowitem.title:
                item = QGraphicsSimpleTextItem(rowitem.title, parent=self)
                item.setTransform(item.transform().rotate(-90))
                item = SimpleLayoutItem(item, parent=grid, anchor=(0, 1),
                                        anchorItem=(0, 0))
                item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
                grid.addItem(item, Row0 + i, self.GroupTitleColumn,
                             alignment=Qt.AlignCenter)
            if rowitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=self,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=True,
                )
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
                grid.addItem(dendrogram, Row0 + i, DendrogramColumn)
                row_dendrograms[i] = dendrogram

        for j, colitem in enumerate(parts.columns):
            if colitem.title:
                item = SimpleLayoutItem(
                    QGraphicsSimpleTextItem(colitem.title, parent=self),
                    parent=grid, anchor=(0.5, 0.5), anchorItem=(0.5, 0.5)
                )
                item.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                grid.addItem(item, self.GroupTitleRow, Col0 + 2 * j + 1)

            if colitem.cluster:
                dendrogram = DendrogramWidget(
                    parent=self,
                    orientation=DendrogramWidget.Top,
                    selectionMode=DendrogramWidget.NoSelection,
                    hoverHighlightEnabled=False
                )
                dendrogram.set_root(colitem.cluster)
                dendrogram.setMaximumHeight(100)
                dendrogram.setMinimumHeight(100)
                # Ignore dendrogram horizontal size hint (heatmap's width
                # should define the column width).
                dendrogram.setSizePolicy(
                    QSizePolicy.Ignored, QSizePolicy.Expanding)
                grid.addItem(dendrogram, DendrogramRow, Col0 + 2 * j + 1)
                column_dendrograms[j] = dendrogram

        heatmap_widgets = []
        for i in range(N):
            heatmap_row = []
            for j in range(M):
                row_ix = parts.rows[i].normalized_indices
                col_ix = parts.columns[j].normalized_indices
                X_part = data[np.ix_(row_ix, col_ix)]
                hw = GraphicsHeatmapWidget(
                    aspectRatioMode=self.__aspectRatioMode,
                    data=X_part, span=parts.span, colormap=colormap,
                )
                sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                sp.setHeightForWidth(True)
                hw.setSizePolicy(sp)

                avgimg = GraphicsHeatmapWidget(
                    data=np.nanmean(X_part, axis=1, keepdims=True),
                    span=parts.span, colormap=colormap,
                    visible=self.__averagesVisible,
                    minimumSize=QSizeF(5, -1)
                )
                avgimg.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
                grid.addItem(avgimg, Row0 + i, Col0 + 2 * j)
                grid.addItem(hw, Row0 + i, Col0 + 2 * j + 1)

                heatmap_row.append(hw)
            heatmap_widgets.append(heatmap_row)

        for j in range(M):
            grid.setColumnStretchFactor(Col0 + 2 * j, 1)
            grid.setColumnStretchFactor(
                Col0 + 2 * j + 1, parts.columns[j].size)
        grid.setColumnStretchFactor(RightLabelColumn - 1, 1)

        for i in range(N):
            grid.setRowStretchFactor(Row0 + i, parts.rows[i].size)

        row_annotation_widgets = []
        col_annotation_widgets = []
        col_annotation_widgets_top = []
        col_annotation_widgets_bottom = []

        for i, rowitem in enumerate(parts.rows):
            # Right row annotations
            indices = np.asarray(rowitem.normalized_indices, dtype=np.intp)
            labels = row_names[indices]
            labelslist = TextListWidget(
                items=labels, parent=self, orientation=Qt.Vertical,
                alignment=Qt.AlignLeft | Qt.AlignVCenter,
                sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored),
                autoScale=True,
                objectName="row-labels-right"
            )
            labelslist.setMaximumWidth(300)
            rowauxsidecolor = GraphicsPixmapWidget(
                parent=self, visible=False,
                scaleContents=True, aspectMode=Qt.IgnoreAspectRatio,
                sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Ignored),
                minimumSize=QSizeF(10, -1)
            )
            grid.addItem(rowauxsidecolor, Row0 + i, RightLabelColumn - 1)
            grid.addItem(labelslist, Row0 + i, RightLabelColumn, Qt.AlignLeft)
            row_annotation_widgets.append(labelslist)
            right_side_colors[i] = rowauxsidecolor

        for j, colitem in enumerate(parts.columns):
            # Top attr annotations
            indices = np.asarray(colitem.normalized_indices, dtype=np.intp)
            labels = col_names[indices]
            sp = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            sp.setHeightForWidth(True)
            labelslist = TextListWidget(
                items=labels, parent=self,
                alignment=Qt.AlignLeft | Qt.AlignVCenter,
                orientation=Qt.Horizontal,
                autoScale=True,
                sizePolicy=sp,
                visible=self.__columnLabelPosition & Position.Top,
                objectName="column-labels-top",
            )
            colauxsidecolor = GraphicsPixmapWidget(
                parent=self, visible=False,
                scaleContents=True, aspectMode=Qt.IgnoreAspectRatio,
                sizePolicy=QSizePolicy(QSizePolicy.Ignored,
                                       QSizePolicy.Maximum),
                minimumSize=QSizeF(-1, 10)
            )

            grid.addItem(labelslist, TopLabelsRow, Col0 + 2 * j + 1,
                         Qt.AlignBottom | Qt.AlignLeft)
            grid.addItem(colauxsidecolor, TopAnnotationRow, Col0 + 2 * j + 1)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_top.append(labelslist)
            top_side_colors[j] = colauxsidecolor

            # Bottom attr annotations
            labelslist = TextListWidget(
                items=labels, parent=self,
                alignment=Qt.AlignRight | Qt.AlignVCenter,
                orientation=Qt.Horizontal,
                autoScale=True,
                sizePolicy=sp,
                visible=self.__columnLabelPosition & Position.Bottom,
                objectName="column-labels-bottom",
            )
            grid.addItem(labelslist, BottomLabelsRow, Col0 + 2 * j + 1)
            col_annotation_widgets.append(labelslist)
            col_annotation_widgets_bottom.append(labelslist)

        row_color_annotation_header = QGraphicsSimpleTextItem("", self)
        row_color_annotation_header.setTransform(
            row_color_annotation_header.transform().rotate(-90))

        grid.addItem(SimpleLayoutItem(
            row_color_annotation_header, anchor=(0, 1),
            aspectMode=Qt.KeepAspectRatio,
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred),
            ),
            0, RightLabelColumn - 1, self.TopLabelsRow + 1, 1,
            alignment=Qt.AlignBottom
        )

        col_color_annotation_header = QGraphicsSimpleTextItem("", self)
        grid.addItem(SimpleLayoutItem(
            col_color_annotation_header, anchor=(1, 1), anchorItem=(1, 1),
            aspectMode=Qt.KeepAspectRatio,
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed),
        ),
            TopAnnotationRow, 0, 1, Col0, alignment=Qt.AlignRight
        )

        legend = GradientLegendWidget(
            parts.span[0], parts.span[1],
            colormap,
            parent=self,
            minimumSize=QSizeF(100, 20),
            visible=self.__legendVisible,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        )
        legend.setMaximumWidth(300)
        grid.addItem(legend, self.LegendRow, self.LegendCol, 1, M * 2 - 1)

        def container(parent=None, orientation=Qt.Horizontal, margin=0, spacing=0, **kwargs):
            widget = QGraphicsWidget(**kwargs)
            layout = QGraphicsLinearLayout(orientation)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setSpacing(spacing)
            widget.setLayout(layout)
            if parent is not None:
                widget.setParentItem(parent)

            return widget
        # Container for color annotation legends
        legend_container = container(
            spacing=3,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed),
            visible=False, objectName="annotation-legend-container"
        )
        legend_container_rows = container(
            parent=legend_container,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed),
            visible=False, objectName="row-annotation-legend-container"
        )
        legend_container_cols = container(
            parent=legend_container,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed),
            visible=False, objectName="col-annotation-legend-container"
        )
        # ? keep refs to child containers; segfault in scene.clear() ?
        legend_container._refs = (legend_container_rows, legend_container_cols)
        legend_container.layout().addItem(legend_container_rows)
        legend_container.layout().addItem(legend_container_cols)

        grid.addItem(legend_container, BottomLabelsRow + 1, Col0 + 1, 1, M * 2 - 1,
                     alignment=Qt.AlignRight)

        self.heatmap_widget_grid = heatmap_widgets
        self.row_annotation_widgets = row_annotation_widgets
        self.col_annotation_widgets = col_annotation_widgets
        self.col_annotation_widgets_top = col_annotation_widgets_top
        self.col_annotation_widgets_bottom = col_annotation_widgets_bottom
        self.col_dendrograms = column_dendrograms
        self.row_dendrograms = row_dendrograms
        self.right_side_colors = right_side_colors
        self.top_side_colors = top_side_colors
        self.heatmap_colormap_legend = legend
        self.bottom_legend_container = legend_container
        self.parts = parts
        self.__selection_manager.set_heatmap_widgets(heatmap_widgets)

    def legendVisible(self) -> bool:
        """Is the colormap legend visible."""
        return self.__legendVisible

    def setLegendVisible(self, visible: bool) -> None:
        """Set colormap legend visible state."""
        self.__legendVisible = visible
        legends = [
            self.heatmap_colormap_legend,
            self.bottom_legend_container
        ]
        apply_all(filter(None, legends), lambda item: item.setVisible(visible))

    legendVisible_ = Property(bool, legendVisible, setLegendVisible)

    def setAspectRatioMode(self, mode: Qt.AspectRatioMode) -> None:
        """
        Set the scale aspect mode.

        The widget will try to keep (hint) the scale ratio via the sizeHint
        reimplementation.
        """
        if self.__aspectRatioMode != mode:
            self.__aspectRatioMode = mode
            for hm in chain.from_iterable(self.heatmap_widget_grid):
                hm.setAspectMode(mode)
            sp = self.sizePolicy()
            sp.setHeightForWidth(mode != Qt.IgnoreAspectRatio)
            self.setSizePolicy(sp)

    def aspectRatioMode(self) -> Qt.AspectRatioMode:
        return self.__aspectRatioMode

    aspectRatioMode_ = Property(
        Qt.AspectRatioMode, aspectRatioMode, setAspectRatioMode
    )

    def setColumnLabelsPosition(self, position: Position) -> None:
        self.__columnLabelPosition = position
        top = bool(position & HeatmapGridWidget.PositionTop)
        bottom = bool(position & HeatmapGridWidget.PositionBottom)
        for w in self.col_annotation_widgets_top:
            w.setVisible(top)
            w.setMaximumHeight(FLT_MAX if top else 0)
        for w in self.col_annotation_widgets_bottom:
            w.setVisible(bottom)
            w.setMaximumHeight(FLT_MAX if bottom else 0)

    def columnLabelPosition(self) -> Position:
        return self.__columnLabelPosition

    def setColumnLabels(self, data: Optional[Sequence[str]]) -> None:
        """Set the column labels to display. If None clear the row names."""
        if data is not None:
            data = np.asarray(data, dtype=object)
        for top, bottom, part in zip(self.col_annotation_widgets_top,
                                     self.col_annotation_widgets_bottom,
                                     self.parts.columns):
            if data is not None:
                top.setItems(data[part.normalized_indices])
                bottom.setItems(data[part.normalized_indices])
            else:
                top.clear()
                bottom.clear()

    def setRowLabels(self, data: Optional[Sequence[str]]):
        """
        Set the row labels to display. If None clear the row names.
        """
        if data is not None:
            data = np.asarray(data, dtype=object)
        for widget, part in zip(self.row_annotation_widgets, self.parts.rows):
            if data is not None:
                widget.setItems(data[part.normalized_indices])
            else:
                widget.clear()

    def setRowLabelsVisible(self, visible: bool):
        """Set row labels visibility"""
        for widget in self.row_annotation_widgets:
            widget.setVisible(visible)

    def setRowSideColorAnnotations(
            self, data: np.ndarray, colormap: ColorMap = None, name=""
    ):
        """
        Set an optional row side color annotations.

        Parameters
        ----------
        data: Optional[np.ndarray]
            A sequence such that it is accepted by `colormap.apply`. If None
            then the color annotations are cleared.
        colormap: ColorMap
        name: str
            Name/title for the annotation column.
        """
        col = self.Col0 + 2 * len(self.parts.columns)
        legend_layout = self.bottom_legend_container.layout()
        legend_container = legend_layout.itemAt(1)
        self.__setColorAnnotationsHelper(
            data, colormap, name, self.right_side_colors, col, Qt.Vertical,
            legend_container
        )
        legend_container.setVisible(True)

    def setColumnSideColorAnnotations(
            self, data: np.ndarray, colormap: ColorMap = None, name=""
    ):
        """
        Set an optional column color annotations.

        Parameters
        ----------
        data: Optional[np.ndarray]
            A sequence such that it is accepted by `colormap.apply`. If None
            then the color annotations are cleared.
        colormap: ColorMap
        name: str
            Name/title for the annotation column.
        """
        row = self.TopAnnotationRow
        legend_layout = self.bottom_legend_container.layout()
        legend_container = legend_layout.itemAt(0)
        self.__setColorAnnotationsHelper(
            data, colormap, name, self.top_side_colors, row, Qt.Horizontal,
            legend_container)
        legend_container.setVisible(True)

    def __setColorAnnotationsHelper(
            self, data: np.ndarray, colormap: ColorMap, name: str,
            items: List[GraphicsPixmapWidget], position: int,
            orientation: Qt.Orientation, legend_container: QGraphicsWidget):
        layout = self.__layout
        if orientation == Qt.Horizontal:
            nameitem = layout.itemAt(position, 0)
        else:
            nameitem = layout.itemAt(self.TopLabelsRow, position)
        size = QFontMetrics(self.font()).lineSpacing()
        layout_clear(legend_container.layout())

        def grid_set_maximum_size(position: int, size: float):
            if orientation == Qt.Horizontal:
                layout.setRowMaximumHeight(position, size)
            else:
                layout.setColumnMaximumWidth(position, size)

        def set_minimum_size(item: QGraphicsLayoutItem, size: float):
            if orientation == Qt.Horizontal:
                item.setMinimumHeight(size)
            else:
                item.setMinimumWidth(size)
            item.updateGeometry()

        def reset_minimum_size(item: QGraphicsLayoutItem):
            set_minimum_size(item, -1)

        def set_hidden(item: GraphicsPixmapWidget):
            item.setVisible(False)
            reset_minimum_size(item,)

        def set_visible(item: GraphicsPixmapWidget):
            item.setVisible(True)
            set_minimum_size(item, 10)

        def set_preferred_size(item, size):
            if orientation == Qt.Horizontal:
                item.setPreferredHeight(size)
            else:
                item.setPreferredWidth(size)
            item.updateGeometry()

        if data is None:
            apply_all(filter(None, items), set_hidden)
            grid_set_maximum_size(position, 0)

            nameitem.item.setVisible(False)
            nameitem.updateGeometry()
            legend_container.setVisible(False)
            return
        else:
            apply_all(filter(None, items), set_visible)
            grid_set_maximum_size(position, FLT_MAX)
            legend_container.setVisible(True)

        if orientation == Qt.Horizontal:
            parts = self.parts.columns
        else:
            parts = self.parts.rows
        for p, item in zip(parts, items):
            if item is not None:
                subset = data[p.normalized_indices]
                subset = colormap.apply(subset)
                rgbdata = subset.reshape((-1, 1, subset.shape[-1]))
                if orientation == Qt.Horizontal:
                    rgbdata = rgbdata.reshape((1, -1, rgbdata.shape[-1]))
                img = qimage_from_array(rgbdata)
                item.setPixmap(img)
                item.setVisible(True)
                set_preferred_size(item, size)

        nameitem.item.setText(name)
        nameitem.item.setVisible(True)
        set_preferred_size(nameitem, size)

        container = legend_container.layout()
        if isinstance(colormap, CategoricalColorMap):
            legend = CategoricalColorLegend(
                colormap, title=name,
                orientation=Qt.Horizontal,
                sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum),
                visible=self.__legendVisible,
            )
            container.addItem(legend)
        elif isinstance(colormap, GradientColorMap):
            legend = GradientLegendWidget(
                *colormap.span, colormap, title=name,
                sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum),
            )
            legend.setMinimumWidth(100)
            container.addItem(legend)

    def headerGeometry(self) -> QRectF:
        """Return the 'header' geometry.

        This is the top part of the widget spanning the top dendrogram,
        column labels... (can be empty).
        """
        layout = self.__layout
        geom1 = grid_layout_row_geometry(layout, self.DendrogramRow)
        geom2 = grid_layout_row_geometry(layout, self.TopLabelsRow)
        first = grid_layout_row_geometry(layout, self.TopLabelsRow + 1)
        geom = geom1.united(geom2)
        if geom.isValid():
            if first.isValid():
                geom.setBottom(geom.bottom() / 2.0 + first.top() / 2.0)
            return QRectF(self.geometry().topLeft(), geom.bottomRight())
        else:
            return QRectF()

    def footerGeometry(self) -> QRectF:
        """Return the 'footer' geometry.

        This is the bottom part of the widget spanning the bottom column labels
        when applicable (can be empty).
        """
        layout = self.__layout
        row = self.Row0 + len(self.heatmap_widget_grid)
        geom = grid_layout_row_geometry(layout, row)
        nextolast = grid_layout_row_geometry(layout, row - 1)
        if geom.isValid():
            if nextolast.isValid():
                geom.setTop(geom.top() / 2 + nextolast.bottom() / 2)
            return QRectF(geom.topLeft(), self.geometry().bottomRight())
        else:
            return QRectF()

    def setColorMap(self, colormap: GradientColorMap) -> None:
        self.__colormap = colormap
        for hm in chain.from_iterable(self.heatmap_widget_grid):
            hm.setColorMap(colormap)
        for item in self.__avgitems():
            item.setColorMap(colormap)
        for ch in self.childItems():
            if isinstance(ch, GradientLegendWidget):
                ch.setColorMap(colormap)

    def colorMap(self) -> ColorMap:
        return self.__colormap

    def __avgitems(self):
        if self.parts is None:
            return
        N = len(self.parts.rows)
        M = len(self.parts.columns)
        layout = self.__layout
        for i in range(N):
            for j in range(M):
                item = layout.itemAt(self.Row0 + i, self.Col0 + 2 * j)
                if isinstance(item, GraphicsHeatmapWidget):
                    yield item

    def setShowAverages(self, visible):
        self.__averagesVisible = visible
        for item in self.__avgitems():
            item.setVisible(visible)
            item.setPreferredWidth(0 if not visible else 10)

    def event(self, event):
        # type: (QEvent) -> bool
        rval = super().event(event)
        if event.type() == QEvent.LayoutRequest and self.layout() is not None:
            self.__update_selection_geometry()
        return rval

    def setGeometry(self, rect: QRectF) -> None:
        super().setGeometry(rect)
        self.__update_selection_geometry()

    def __update_selection_geometry(self):
        scene = self.scene()
        self.__selection_manager.update_selection_rects()
        rects = self.__selection_manager.selection_rects
        palette = self.palette()
        pen = QPen(palette.color(QPalette.Foreground), 2)
        pen.setCosmetic(True)
        brushcolor = QColor(palette.color(QPalette.Highlight))
        brushcolor.setAlpha(50)
        selection_rects = []
        for rect, item in zip_longest(rects, self.selection_rects):
            assert rect is not None or item is not None
            if item is None:
                item = QGraphicsRectItem(rect, None)
                item.setPen(pen)
                item.setBrush(brushcolor)
                scene.addItem(item)
                selection_rects.append(item)
            elif rect is not None:
                item.setRect(rect)
                item.setPen(pen)
                item.setBrush(brushcolor)
                selection_rects.append(item)
            else:
                scene.removeItem(item)
        self.selection_rects = selection_rects

    def __select_by_cluster(self, item, dendrogramindex):
        # User clicked on a dendrogram node.
        # Select all rows corresponding to the cluster item.
        node = item.node
        try:
            hm = self.heatmap_widget_grid[dendrogramindex][0]
        except IndexError:
            pass
        else:
            key = QApplication.keyboardModifiers()
            clear = not (key & ((Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier)))
            remove = (key & (Qt.ControlModifier | Qt.AltModifier))
            append = (key & Qt.ControlModifier)
            self.__selection_manager.selection_add(
                node.value.first, node.value.last - 1, hm,
                clear=clear, remove=remove, append=append)

    def heatmapAtPos(self, pos: QPointF) -> Optional['GraphicsHeatmapWidget']:
        for hw in chain.from_iterable(self.heatmap_widget_grid):
            if hw.contains(hw.mapFromItem(self, pos)):
                return hw
        return None

    __selecting = False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        heatmap = self.heatmapAtPos(pos)
        if heatmap and event.button() & Qt.LeftButton:
            row, _ = heatmap.heatmapCellAt(heatmap.mapFromScene(event.scenePos()))
            if row != -1:
                self.__selection_manager.selection_start(heatmap, event)
                self.__selecting = True
                event.setAccepted(True)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        heatmap = self.heatmapAtPos(pos)
        if heatmap and event.buttons() & Qt.LeftButton and self.__selecting:
            row, _ = heatmap.heatmapCellAt(heatmap.mapFromScene(pos))
            if row != -1:
                self.__selection_manager.selection_update(heatmap, event)
                event.setAccepted(True)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if event.button() == Qt.LeftButton and self.__selecting:
            self.__selection_manager.selection_finish(
                self.heatmapAtPos(pos), event)
            self.__selecting = False
        super().mouseReleaseEvent(event)

    def selectedRows(self) -> Sequence[int]:
        """Return the current selected rows."""
        if self.parts is None:
            return []
        visual_indices = self.__selection_manager.selections
        indices = np.hstack([r.normalized_indices for r in self.parts.rows])
        return indices[visual_indices].tolist()

    def selectRows(self, selection: Sequence[int]):
        """Select the specified rows. Previous selection is cleared."""
        if self.parts is not None:
            indices = np.hstack([r.normalized_indices for r in self.parts.rows])
        else:
            indices = []
        condition = np.in1d(indices, selection)
        visual_indices = np.flatnonzero(condition)
        self.__selection_manager.select_rows(visual_indices.tolist())


class GraphicsHeatmapWidget(QGraphicsWidget):
    __aspectMode = Qt.KeepAspectRatio

    def __init__(
            self, parent=None,
            data: Optional[np.ndarray] = None,
            span: Tuple[float, float] = (0., 1.),
            colormap: Optional[ColorMap] = None,
            aspectRatioMode=Qt.KeepAspectRatio,
            **kwargs
    ) -> None:
        super().__init__(None, **kwargs)
        self.setAcceptHoverEvents(True)
        self.__levels = span
        if colormap is None:
            colormap = GradientColorMap(DefaultContinuousPalette.lookup_table())
        self.__colormap = colormap
        self.__data: Optional[np.ndarray] = None
        self.__pixmap = QPixmap()
        self.__aspectMode = aspectRatioMode

        layout = QGraphicsLinearLayout(Qt.Horizontal)
        layout.setContentsMargins(0, 0, 0, 0)
        self.__pixmapItem = GraphicsPixmapWidget(
            self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio
        )
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sp.setHeightForWidth(True)
        self.__pixmapItem.setSizePolicy(sp)
        layout.addItem(self.__pixmapItem)
        self.setLayout(layout)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setHeatmapData(data)

        if parent is not None:
            self.setParentItem(parent)

    def setAspectMode(self, mode: Qt.AspectRatioMode) -> None:
        if self.__aspectMode != mode:
            self.__aspectMode = mode
            sp = self.sizePolicy()
            sp.setHeightForWidth(mode != Qt.IgnoreAspectRatio)
            self.setSizePolicy(sp)
            self.updateGeometry()

    def aspectMode(self) -> Qt.AspectRatioMode:
        return self.__aspectMode

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize and constraint.width() >= 0:
            sh = super().sizeHint(which)
            return scaled(sh, QSizeF(constraint.width(), -1), self.__aspectMode)
        return super().sizeHint(which, constraint)

    def clear(self):
        """Clear/reset the widget."""
        self.__data = None
        self.__pixmap = QPixmap()
        self.__pixmapItem.setPixmap(self.__pixmap)
        self.updateGeometry()

    def setHeatmapData(self, data):
        """Set the heatmap data for display."""
        if self.__data is not data:
            self.clear()
            self.__data = data
            self.__updatePixmap()
            self.update()

    def heatmapData(self) -> Optional[np.ndarray]:
        if self.__data is not None:
            v = self.__data.view()
            v.flags.writeable = False
            return v
        else:
            return None

    def pixmap(self) -> QPixmap:
        return self.__pixmapItem.pixmap()

    def setLevels(self, levels: Tuple[float, float]) -> None:
        if levels != self.__levels:
            self.__levels = levels
            self.__updatePixmap()
            self.update()

    def setColorMap(self, colormap: ColorMap):
        self.__colormap = colormap
        self.__updatePixmap()

    def colorMap(self,) -> ColorMap:
        return self.__colormap

    def __updatePixmap(self):
        if self.__data is not None:
            ll, lh = self.__levels
            cmap = self.__colormap.replace(span=(ll, lh))
            rgb = cmap.apply(self.__data)
            rgb[np.isnan(self.__data)] = (100, 100, 100)
            qimage = qimage_from_array(rgb)
            self.__pixmap = QPixmap.fromImage(qimage)
        else:
            self.__pixmap = QPixmap()

        self.__pixmapItem.setPixmap(self.__pixmap)
        self.__updateSizeHints()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self.__updateSizeHints()

    def __updateSizeHints(self):
        hmsize = QSizeF(self.__pixmap.size())
        size = QFontMetrics(self.font()).lineSpacing()
        self.__pixmapItem.setMinimumSize(hmsize)
        self.__pixmapItem.setPreferredSize(hmsize * size)

    def heatmapCellAt(self, pos: QPointF) -> Tuple[int, int]:
        """Return the cell row, column from `pos` in local coordinates.
        """
        if self.__pixmap.isNull() or not \
                self.__pixmapItem.geometry().contains(pos):
            return -1, -1
        assert self.__data is not None
        item_clicked = self.__pixmapItem
        pos = self.mapToItem(item_clicked, pos)
        size = self.__pixmapItem.size()

        x, y = pos.x(), pos.y()

        N, M = self.__data.shape
        fx = x / size.width()
        fy = y / size.height()
        i = min(int(math.floor(fy * N)), N - 1)
        j = min(int(math.floor(fx * M)), M - 1)
        return i, j

    def heatmapCellRect(self, row: int, column: int) -> QRectF:
        """Return a rectangle in local coordinates containing the cell
        at `row` and `column`.
        """
        size = self.__pixmap.size()
        if not (0 <= column < size.width() or 0 <= row < size.height()):
            return QRectF()

        topleft = QPointF(column, row)
        bottomright = QPointF(column + 1, row + 1)
        t = self.__pixmapItem.pixmapTransform()
        rect = t.mapRect(QRectF(topleft, bottomright))
        rect.translated(self.__pixmapItem.pos())
        return rect

    def rowRect(self, row):
        """
        Return a QRectF in local coordinates containing the entire row.
        """
        rect = self.heatmapCellRect(row, 0)
        rect.setLeft(0)
        rect.setRight(self.size().width())
        return rect

    def heatmapCellToolTip(self, row, column):
        return "{}, {}: {:g}".format(row, column, self.__data[row, column])

    def hoverMoveEvent(self, event):
        pos = event.pos()
        row, column = self.heatmapCellAt(pos)
        if row != -1:
            tooltip = self.heatmapCellToolTip(row, column)
            self.setToolTip(tooltip)
        return super().hoverMoveEvent(event)


def remove_item(item: QGraphicsItem) -> None:
    scene = item.scene()
    if scene is not None:
        scene.removeItem(item)
    else:
        item.setParentItem(None)


class _GradientLegendAxisItem(pg.AxisItem):
    def boundingRect(self):
        br = super().boundingRect()
        if self.orientation in ["top", "bottom"]:
            # adjust brect (extend in horizontal direction). pg.AxisItem has
            # only fixed constant adjustment for tick text over-flow.
            font = self.style.get("tickFont")
            if font is None:
                font = self.font()
            fm = QFontMetrics(font)
            w = fm.horizontalAdvance('0.0000000') / 2  # bad, should use _tickValues
            geomw = self.geometry().size().width()
            maxw = max(geomw + 2 * w, br.width())
            if br.width() < maxw:
                adjust = (maxw - br.width()) / 2
                br = br.adjusted(-adjust, 0, adjust, 0)
        return br

    def showEvent(self, event):
        super().showEvent(event)
        # AxisItem resizes to 0 width/height when hidden, does not update when
        # shown implicitly (i.e. a parent becomes visible).
        # Use showLabel(False) which should update the size without actually
        # changing anything else (no public interface to explicitly recalc
        # fixed sizes).
        self.showLabel(False)


class GradientLegendWidget(QGraphicsWidget):
    def __init__(
            self, low, high, colormap: GradientColorMap, parent=None, title="",
            **kwargs
    ):
        kwargs.setdefault(
            "sizePolicy", QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        )
        super().__init__(None, **kwargs)
        self.low = low
        self.high = high
        self.colormap = colormap
        self.title = title

        layout = QGraphicsLinearLayout(Qt.Vertical)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        if title:
            titleitem = SimpleLayoutItem(
                QGraphicsSimpleTextItem(title, self), parent=layout,
                anchor=(0.5, 1.), anchorItem=(0.5, 1.0)
            )
            layout.addItem(titleitem)
        self.__axis = axis = _GradientLegendAxisItem(
            orientation="top", maxTickLength=3)
        axis.setRange(low, high)
        layout.addItem(axis)
        pen = QPen(self.palette().color(QPalette.Text))
        axis.setPen(pen)
        self.__pixitem = GraphicsPixmapWidget(
            parent=self, scaleContents=True, aspectMode=Qt.IgnoreAspectRatio
        )
        self.__pixitem.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.__pixitem.setMinimumHeight(12)
        layout.addItem(self.__pixitem)
        self.__update()

        if parent is not None:
            self.setParentItem(parent)

    def setRange(self, low, high):
        if self.low != low or self.high != high:
            self.low = low
            self.high = high
            self.__update()

    def setColorMap(self, colormap: ColorMap) -> None:
        """Set the color map"""
        self.colormap = colormap
        self.__update()

    def colorMap(self) -> ColorMap:
        return self.colormap

    def __update(self):
        low, high = self.low, self.high
        data = np.linspace(low, high, num=1000).reshape((1, -1))
        cmap = self.colormap.replace(span=(low, high))
        qimg = qimage_from_array(cmap.apply(data))
        self.__pixitem.setPixmap(QPixmap.fromImage(qimg))
        if self.colormap.center is not None \
                and low < self.colormap.center < high:
            tick_values = [low, self.colormap.center, high]
        else:
            tick_values = [low, high]
        tickformat = "{:.6g}".format
        ticks = [(val, tickformat(val)) for val in tick_values]
        self.__axis.setRange(low, high)
        self.__axis.setTicks([ticks])

        self.updateGeometry()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.PaletteChange:
            pen = QPen(self.palette().color(QPalette.Text))
            self.__axis.setPen(pen)
        super().changeEvent(event)


class CategoricalColorLegend(QGraphicsWidget):
    def __init__(
            self, colormap: CategoricalColorMap, title="",
            orientation=Qt.Vertical, parent=None, **kwargs,
    ) -> None:
        self.__colormap = colormap
        self.__title = title
        self.__names = colormap.names
        self.__layout = QGraphicsLinearLayout(Qt.Vertical)
        self.__flow = GraphicsFlowLayout()
        self.__layout.addItem(self.__flow)
        self.__flow.setHorizontalSpacing(4)
        self.__flow.setVerticalSpacing(4)
        self.__orientation = orientation
        kwargs.setdefault(
            "sizePolicy", QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        super().__init__(None, **kwargs)
        self.setLayout(self.__layout)
        self._setup()

        if parent is not None:
            self.setParent(parent)

    def setOrientation(self, orientation):
        if self.__orientation != orientation:
            self._clear()
            self._setup()

    def orientation(self):
        return self.__orientation

    def _clear(self):
        items = list(layout_items(self.__flow))
        layout_clear(self.__flow)
        for item in items:
            if isinstance(item, SimpleLayoutItem):
                remove_item(item.item)
        # remove 'title' item if present
        items = [item for item in layout_items(self.__layout)
                 if item is not self.__flow]
        for item in items:
            self.__layout.removeItem(item)
            if isinstance(item, SimpleLayoutItem):
                remove_item(item.item)

    def _setup(self):
        # setup the layout
        colors = self.__colormap.colortable
        names = self.__colormap.names
        title = self.__title
        layout = self.__layout
        flow = self.__flow
        assert flow.count() == 0
        font = self.font()
        fm = QFontMetrics(font)
        size = fm.horizontalAdvance("X")
        headeritem = None
        if title:
            headeritem = QGraphicsSimpleTextItem(title)
            headeritem.setFont(font)

        def centered(item):
            return SimpleLayoutItem(item, anchor=(0.5, 0.5), anchorItem=(0.5, 0.5))

        def legend_item_pair(color: QColor, size: float, text: str):
            coloritem = QGraphicsRectItem(0, 0, size, size)
            coloritem.setBrush(color)
            textitem = QGraphicsSimpleTextItem()
            textitem.setFont(font)
            textitem.setText(text)
            layout = QGraphicsLinearLayout(Qt.Horizontal)
            layout.setSpacing(2)
            layout.addItem(centered(coloritem))
            layout.addItem(SimpleLayoutItem(textitem))
            return coloritem, textitem, layout

        items = [legend_item_pair(QColor(*color), size, name)
                 for color, name in zip(colors, names)]

        for sym, label, pair_layout in items:
            flow.addItem(pair_layout)

        if headeritem:
            layout.insertItem(0, centered(headeritem))

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.FontChange:
            self._updateFont(self.font())
        super().changeEvent(event)

    def _updateFont(self, font):
        w = QFontMetrics(font).horizontalAdvance("X")
        for item in filter(
                lambda item: isinstance(item, SimpleLayoutItem),
                layout_items_recursive(self.__layout)
        ):
            if isinstance(item.item, QGraphicsSimpleTextItem):
                item.item.setFont(font)
            elif isinstance(item.item, QGraphicsRectItem):
                item.item.setRect(QRectF(0, 0, w, w))
            item.updateGeometry()


def layout_items(layout: QGraphicsLayout) -> Iterable[QGraphicsLayoutItem]:
    for item in map(layout.itemAt, range(layout.count())):
        if item is not None:
            yield item


def layout_items_recursive(layout: QGraphicsLayout):
    for item in map(layout.itemAt, range(layout.count())):
        if item is not None:
            if item.isLayout():
                assert isinstance(item, QGraphicsLayout)
                yield from layout_items_recursive(item)
            else:
                yield item


def layout_clear(layout: QGraphicsLayout) -> None:
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        layout.removeAt(i)
        if item is not None and item.graphicsItem() is not None:
            remove_item(item.graphicsItem())


class SelectionManager(QObject):
    """
    Selection manager for heatmap rows
    """
    selection_changed = Signal()
    selection_finished = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.selections = []
        self.selection_ranges = []
        self.selection_ranges_temp = []
        self.selection_rects = []
        self.heatmaps = []
        self._heatmap_ranges: Dict[GraphicsHeatmapWidget, Tuple[int, int]] = {}
        self._start_row = 0

    def clear(self):
        self.remove_rows(self.selection)

    def set_heatmap_widgets(self, widgets):
        # type: (Sequence[Sequence[GraphicsHeatmapWidget]] )-> None
        self.remove_rows(self.selections)
        self.heatmaps = list(zip(*widgets))

        # Compute row ranges for all heatmaps
        self._heatmap_ranges = {}
        for group in zip(*widgets):
            start = end = 0
            for heatmap in group:
                end += heatmap.heatmapData().shape[0]
                self._heatmap_ranges[heatmap] = (start, end)
                start = end

    def select_rows(self, rows, heatmap=None, clear=True):
        """Add `rows` to selection. If `heatmap` is provided the rows
        are mapped from the local indices to global heatmap indices. If `clear`
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
        row, _ = heatmap_widget.heatmapCellAt(pos)

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
        row, _ = heatmap_widget.heatmapCellAt(pos)
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
        if heatmap_widget is not None:
            pos = heatmap_widget.mapFromScene(event.scenePos())
            row, _ = heatmap_widget.heatmapCellAt(pos)
            start, _ = self._heatmap_ranges[heatmap_widget]
            row = start + row
            if event.modifiers() & Qt.ControlModifier:
                pass
            elif event.modifiers() & Qt.AltModifier:
                self.selection_ranges = self.remove_range(
                    self.selection_ranges, self._start_row, row, append=False)
            else:
                if len(self.selection_ranges) > 0:
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
        def group_selections(selections):
            """Group selections along with heatmaps.
            """
            rows2hm = self.rows_to_heatmaps()
            selections = iter(selections)
            try:
                start = end = next(selections)
            except StopIteration:
                return
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
                rect |= heatmap.mapToScene(heatmap.rowRect(start - h_start)).boundingRect()
                rect |= heatmap.mapToScene(heatmap.rowRect(end - h_start)).boundingRect()
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


Parts = HeatmapGridWidget.Parts
RowItem = HeatmapGridWidget.RowItem
ColumnItem = HeatmapGridWidget.ColumnItem
