# pylint: disable=too-many-lines
from collections import namedtuple
from itertools import chain, count
from typing import List, Optional, Tuple, Set, Sequence

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity

from AnyQt.QtCore import QItemSelection, QPointF, QRectF, QSize, Qt, Signal
from AnyQt.QtGui import QBrush, QColor, QPainter, QPainterPath, QPolygonF
from AnyQt.QtWidgets import QCheckBox, QSizePolicy, QGraphicsRectItem, \
    QGraphicsSceneMouseEvent, QApplication, QWidget, QComboBox

import pyqtgraph as pg

from orangewidget.utils.listview import ListViewSearch
from orangewidget.utils.visual_settings_dlg import KeyType, ValueType, \
    VisualSettingsDialog

from Orange.data import ContinuousVariable, DiscreteVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, \
    Setting
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, \
    create_annotated_table
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owboxplot import SortProxyModel
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter, Updater
from Orange.widgets.visualize.utils.plotutils import AxisItem
from Orange.widgets.widget import OWWidget, Input, Output, Msg

# scaling types
AREA, COUNT, WIDTH = range(3)


class ViolinPlotViewBox(pg.ViewBox):
    sigSelectionChanged = Signal(QPointF, QPointF, bool)
    sigDeselect = Signal(bool)

    def __init__(self, _):
        super().__init__()
        self.setMouseMode(self.RectMode)

    def mouseDragEvent(self, ev, axis=None):
        if axis is None:
            ev.accept()
            if ev.button() == Qt.LeftButton:
                p1, p2 = ev.buttonDownPos(), ev.pos()
                self.sigSelectionChanged.emit(self.mapToView(p1),
                                              self.mapToView(p2),
                                              ev.isFinish())
        else:
            ev.ignore()

    def mousePressEvent(self, ev: QGraphicsSceneMouseEvent):
        self.sigDeselect.emit(False)
        super().mousePressEvent(ev)

    def mouseClickEvent(self, ev):
        ev.accept()
        self.sigDeselect.emit(True)


class ParameterSetter(CommonParameterSetter):
    BOTTOM_AXIS_LABEL, IS_VERTICAL_LABEL = "Bottom axis", "Vertical tick text"

    def __init__(self, master):
        self.master: ViolinPlot = master
        self.titles_settings = {}
        self.ticks_settings = {}
        self.is_vertical_setting = False
        super().__init__()

    def update_setters(self):
        def update_titles(**settings):
            self.titles_settings.update(**settings)
            Updater.update_axes_titles_font(self.axis_items, **settings)

        def update_ticks(**settings):
            self.ticks_settings.update(**settings)
            Updater.update_axes_ticks_font(self.axis_items, **settings)

        def update_bottom_axis(**settings):
            self.is_vertical_setting = settings[self.IS_VERTICAL_LABEL]
            self.bottom_axis.setRotateTicks(self.is_vertical_setting)

        self._setters[self.LABELS_BOX][self.AXIS_TITLE_LABEL] = update_titles
        self._setters[self.LABELS_BOX][self.AXIS_TICKS_LABEL] = update_ticks
        self._setters[self.PLOT_BOX] = {
            self.BOTTOM_AXIS_LABEL: update_bottom_axis,
        }

        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
            },
            self.PLOT_BOX: {
                self.BOTTOM_AXIS_LABEL: {
                    self.IS_VERTICAL_LABEL: (None, self.is_vertical_setting),
                },
            },
        }

    @property
    def title_item(self) -> pg.LabelItem:
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self) -> List[AxisItem]:
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]

    @property
    def bottom_axis(self) -> AxisItem:
        return self.master.getAxis("bottom")


def fit_kernel(data: np.ndarray, kernel: str) -> \
        Tuple[Optional[KernelDensity], float]:
    assert np.all(np.isfinite(data))

    if np.unique(data).size < 2:
        return None, 1

    # obtain bandwidth
    try:
        kde = stats.gaussian_kde(data)
        bw = kde.factor * data.std(ddof=1)
    except np.linalg.LinAlgError:
        bw = 1

    # fit selected kernel
    kde = KernelDensity(bandwidth=bw, kernel=kernel)
    kde.fit(data.reshape(-1, 1))
    return kde, bw


def scale_density(scale_type: int, density: np.ndarray, n_data: int,
                  max_density: float) -> np.ndarray:
    if scale_type == AREA:
        return density
    elif scale_type == COUNT:
        return density * n_data / max_density
    elif scale_type == WIDTH:
        return density / max_density
    else:
        raise NotImplementedError


class ViolinItem(pg.GraphicsObject):
    RugPlot = namedtuple("RugPlot", "support, density")

    def __init__(self, data: np.ndarray, color: QColor, kernel: str,
                 scale: int, show_rug: bool, orientation: Qt.Orientations):
        self.__scale = scale
        self.__show_rug_plot = show_rug
        self.__orientation = orientation

        kde, bw = fit_kernel(data, kernel)
        self.__kde: KernelDensity = kde
        self.__bandwidth: float = bw

        path, max_density = self._create_violin(data)
        self.__violin_path: QPainterPath = path
        self.__violin_brush: QBrush = QBrush(color)

        self.__rug_plot_data: ViolinItem.RugPlot = \
            self._create_rug_plot(data, max_density)

        super().__init__()

    @property
    def density(self) -> np.ndarray:
        # density on unique data
        return self.__rug_plot_data.density

    @property
    def violin_width(self) -> float:
        width = self.boundingRect().width() \
            if self.__orientation == Qt.Vertical \
            else self.boundingRect().height()
        return width or 1

    def set_show_rug_plot(self, show: bool):
        self.__show_rug_plot = show
        self.update()

    def boundingRect(self) -> QRectF:
        return self.__violin_path.boundingRect()

    def paint(self, painter: QPainter, *_):
        painter.save()
        painter.setPen(pg.mkPen(QColor(Qt.black)))
        painter.setBrush(self.__violin_brush)
        painter.drawPath(self.__violin_path)

        if self.__show_rug_plot:
            data, density = self.__rug_plot_data
            painter.setPen(pg.mkPen(QColor(Qt.black), width=1))
            for x, y in zip(density, data):
                if self.__orientation == Qt.Vertical:
                    painter.drawLine(QPointF(-x, y), QPointF(x, y))
                else:
                    painter.drawLine(QPointF(y, -x), QPointF(y, x))

        painter.restore()

    def _create_violin(self, data: np.ndarray) -> Tuple[QPainterPath, float]:
        if self.__kde is None:
            x, p, max_density = np.zeros(1), np.zeros(1), 0
        else:
            x = np.linspace(data.min() - self.__bandwidth * 2,
                            data.max() + self.__bandwidth * 2, 1000)
            p = np.exp(self.__kde.score_samples(x.reshape(-1, 1)))
            max_density = p.max()
            p = scale_density(self.__scale, p, len(data), max_density)

        if self.__orientation == Qt.Vertical:
            pts = [QPointF(pi, xi) for xi, pi in zip(x, p)]
            pts += [QPointF(-pi, xi) for xi, pi in reversed(list(zip(x, p)))]
        else:
            pts = [QPointF(xi, pi) for xi, pi in zip(x, p)]
            pts += [QPointF(xi, -pi) for xi, pi in reversed(list(zip(x, p)))]
        pts += pts[:1]

        polygon = QPolygonF(pts)
        path = QPainterPath()
        path.addPolygon(polygon)
        return path, max_density

    def _create_rug_plot(self, data: np.ndarray, max_density: float) -> Tuple:
        if self.__kde is None:
            return self.RugPlot(data, np.zeros(data.size))

        n_data = len(data)
        data = np.unique(data)  # to optimize scoring
        density = np.exp(self.__kde.score_samples(data.reshape(-1, 1)))
        density = scale_density(self.__scale, density, n_data, max_density)
        return self.RugPlot(data, density)


class BoxItem(pg.GraphicsObject):
    def __init__(self, data: np.ndarray, rect: QRectF,
                 orientation: Qt.Orientations):
        self.__bounding_rect = rect
        self.__orientation = orientation

        self.__box_plot_data: Tuple = self._create_box_plot(data)

        super().__init__()

    def boundingRect(self) -> QRectF:
        return self.__bounding_rect

    def paint(self, painter: QPainter, _, widget: Optional[QWidget]):
        painter.save()

        q0, q25, q75, q100 = self.__box_plot_data
        if self.__orientation == Qt.Vertical:
            quartile1 = QPointF(0, q0), QPointF(0, q100)
            quartile2 = QPointF(0, q25), QPointF(0, q75)
        else:
            quartile1 = QPointF(q0, 0), QPointF(q100, 0)
            quartile2 = QPointF(q25, 0), QPointF(q75, 0)

        factor = 1 if widget is None else widget.devicePixelRatio()
        painter.setPen(pg.mkPen(QColor(Qt.black), width=2 * factor))
        painter.drawLine(*quartile1)
        painter.setPen(pg.mkPen(QColor(Qt.black), width=6 * factor))
        painter.drawLine(*quartile2)

        painter.restore()

    @staticmethod
    def _create_box_plot(data: np.ndarray) -> Tuple:
        if data.size == 0:
            return (0,) * 4

        q25, q75 = np.percentile(data, [25, 75])
        whisker_lim = 1.5 * stats.iqr(data)
        min_ = np.min(data[data >= (q25 - whisker_lim)])
        max_ = np.max(data[data <= (q75 + whisker_lim)])
        return min_, q25, q75, max_


class MedianItem(pg.ScatterPlotItem):
    def __init__(self, data: np.ndarray, orientation: Qt.Orientations):
        self.__value = value = 0 if data.size == 0 else np.median(data)
        x, y = (0, value) if orientation == Qt.Vertical else (value, 0)
        super().__init__(x=[x], y=[y], size=5,
                         pen=pg.mkPen(QColor(Qt.white)),
                         brush=pg.mkBrush(QColor(Qt.white)))

    @property
    def value(self) -> float:
        return self.__value

    def setX(self, x: float):
        self.setData(x=[x], y=[self.value])

    def setY(self, y: float):
        self.setData(x=[self.value], y=[y])


class StripItem(pg.ScatterPlotItem):
    def __init__(self, data: np.ndarray, density: np.ndarray,
                 color: QColor, orientation: Qt.Orientations):
        _, indices = np.unique(data, return_inverse=True)
        density = density[indices]
        self.__xdata = x = np.random.RandomState(0).uniform(-density, density)
        self.__ydata = data
        x, y = (x, data) if orientation == Qt.Vertical else (data, x)
        color = color.lighter(150)
        super().__init__(x=x, y=y, size=5, brush=pg.mkBrush(color))

    def setX(self, x: float):
        self.setData(x=self.__xdata + x, y=self.__ydata)

    def setY(self, y: float):
        self.setData(x=self.__ydata, y=self.__xdata + y)


class SelectionRect(pg.GraphicsObject):
    def __init__(self, rect: QRectF, orientation: Qt.Orientations):
        self.__rect: QRectF = rect
        self.__orientation: Qt.Orientations = orientation
        self.__selection_range: Optional[Tuple[float, float]] = None
        super().__init__()

    @property
    def selection_range(self) -> Optional[Tuple[float, float]]:
        return self.__selection_range

    @selection_range.setter
    def selection_range(self, selection_range: Optional[Tuple[float, float]]):
        self.__selection_range = selection_range
        self.update()

    @property
    def selection_rect(self) -> QRectF:
        rect: QRectF = self.__rect
        if self.__selection_range is not None:
            if self.__orientation == Qt.Vertical:
                rect.setTop(self.__selection_range[0])
                rect.setBottom(self.__selection_range[1])
            else:
                rect.setLeft(self.__selection_range[0])
                rect.setRight(self.__selection_range[1])
        return rect

    def boundingRect(self) -> QRectF:
        return self.__rect

    def paint(self, painter: QPainter, *_):
        painter.save()
        painter.setPen(pg.mkPen((255, 255, 100), width=1))
        painter.setBrush(pg.mkBrush(255, 255, 0, 100))
        if self.__selection_range is not None:
            painter.drawRect(self.selection_rect)
        painter.restore()


class ViolinPlot(pg.PlotWidget):
    VIOLIN_PADDING_FACTOR = 1.25
    SELECTION_PADDING_FACTOR = 1.20
    selection_changed = Signal(list, list)

    def __init__(self, parent: OWWidget, kernel: str, scale: int,
                 orientation: Qt.Orientations, show_box_plot: bool,
                 show_strip_plot: bool, show_rug_plot: bool, sort_items: bool):

        # data
        self.__values: Optional[np.ndarray] = None
        self.__value_var: Optional[ContinuousVariable] = None
        self.__group_values: Optional[np.ndarray] = None
        self.__group_var: Optional[DiscreteVariable] = None

        # settings
        self.__kernel = kernel
        self.__scale = scale
        self.__orientation = orientation
        self.__show_box_plot = show_box_plot
        self.__show_strip_plot = show_strip_plot
        self.__show_rug_plot = show_rug_plot
        self.__sort_items = sort_items

        # items
        self.__violin_items: List[ViolinItem] = []
        self.__box_items: List[BoxItem] = []
        self.__median_items: List[MedianItem] = []
        self.__strip_items: List[pg.ScatterPlotItem] = []

        # selection
        self.__selection: Set[int] = set()
        self.__selection_rects: List[SelectionRect] = []

        view_box = ViolinPlotViewBox(self)
        super().__init__(parent, viewBox=view_box,
                         background="w", enableMenu=False,
                         axisItems={"bottom": AxisItem("bottom"),
                                    "left": AxisItem("left")})
        self.setAntialiasing(True)
        self.hideButtons()
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.setMouseEnabled(False, False)
        view_box.sigSelectionChanged.connect(self._update_selection)
        view_box.sigDeselect.connect(self._deselect)

        self.parameter_setter = ParameterSetter(self)

    @property
    def _selection_ranges(self) -> List[Optional[Tuple[float, float]]]:
        return [rect.selection_range for rect in self.__selection_rects]

    @_selection_ranges.setter
    def _selection_ranges(self, ranges: List[Optional[Tuple[float, float]]]):
        for min_max, sel_rect in zip(ranges, self.__selection_rects):
            sel_rect.selection_range = min_max

    @property
    def _sorted_group_indices(self) -> Sequence[int]:
        medians = [item.value for item in self.__median_items]
        return np.argsort(medians) if self.__sort_items \
            else range(len(medians))

    @property
    def _max_item_width(self) -> float:
        if not self.__violin_items:
            return 0
        return max(item.violin_width * self.VIOLIN_PADDING_FACTOR
                   for item in self.__violin_items)

    def set_data(self, values: np.ndarray, value_var: ContinuousVariable,
                 group_values: Optional[np.ndarray],
                 group_var: Optional[DiscreteVariable]):
        self.__values = values
        self.__value_var = value_var
        self.__group_values = group_values
        self.__group_var = group_var
        self._set_axes()
        self._plot_data()

    def set_kernel(self, kernel: str):
        if self.__kernel != kernel:
            self.__kernel = kernel
            self._plot_data()

    def set_scale(self, scale: int):
        if self.__scale != scale:
            self.__scale = scale
            self._plot_data()

    def set_orientation(self, orientation: Qt.Orientations):
        if self.__orientation != orientation:
            self.__orientation = orientation
            self._clear_axes()
            self._set_axes()
            self._plot_data()

    def set_show_box_plot(self, show: bool):
        if self.__show_box_plot != show:
            self.__show_box_plot = show
            for item in self.__box_items:
                item.setVisible(show)
            for item in self.__median_items:
                item.setVisible(show)

    def set_show_strip_plot(self, show: bool):
        if self.__show_strip_plot != show:
            self.__show_strip_plot = show
            for item in self.__strip_items:
                item.setVisible(show)

    def set_show_rug_plot(self, show: bool):
        if self.__show_rug_plot != show:
            self.__show_rug_plot = show
            for item in self.__violin_items:
                item.set_show_rug_plot(show)

    def set_sort_items(self, sort_items: bool):
        if self.__sort_items != sort_items:
            self.__sort_items = sort_items
            if self.__group_var is not None:
                self.order_items()

    def order_items(self):
        assert self.__group_var is not None

        indices = self._sorted_group_indices

        for i, index in enumerate(indices):
            violin: ViolinItem = self.__violin_items[index]
            box: BoxItem = self.__box_items[index]
            median: MedianItem = self.__median_items[index]
            strip: StripItem = self.__strip_items[index]
            sel_rect: QGraphicsRectItem = self.__selection_rects[index]

            if self.__orientation == Qt.Vertical:
                x = i * self._max_item_width
                violin.setX(x)
                box.setX(x)
                median.setX(x)
                strip.setX(x)
                sel_rect.setX(x)
            else:
                y = - i * self._max_item_width
                violin.setY(y)
                box.setY(y)
                median.setY(y)
                strip.setY(y)
                sel_rect.setY(y)

        sign = 1 if self.__orientation == Qt.Vertical else -1
        side = "bottom" if self.__orientation == Qt.Vertical else "left"
        ticks = [[(i * self._max_item_width * sign,
                   self.__group_var.values[index])
                  for i, index in enumerate(indices)]]
        self.getAxis(side).setTicks(ticks)

    def set_selection(self, ranges: List[Optional[Tuple[float, float]]]):
        if self.__values is None:
            return

        self._selection_ranges = ranges

        self.__selection = set()
        for index, min_max in enumerate(ranges):
            if min_max is None:
                continue
            mask = np.bitwise_and(self.__values >= min_max[0],
                                  self.__values <= min_max[1])
            if self.__group_values is not None:
                mask = np.bitwise_and(mask, self.__group_values == index)
            self.__selection |= set(np.flatnonzero(mask))

        self.selection_changed.emit(sorted(self.__selection),
                                    self._selection_ranges)

    def _set_axes(self):
        if self.__value_var is None:
            return
        value_title = self.__value_var.name
        group_title = self.__group_var.name if self.__group_var else ""
        vertical = self.__orientation == Qt.Vertical
        self.getAxis("left" if vertical else "bottom").setLabel(value_title)
        self.getAxis("bottom" if vertical else "left").setLabel(group_title)

        if self.__group_var is None:
            self.getAxis("bottom" if vertical else "left").setTicks([])

    def _plot_data(self):
        # save selection ranges
        ranges = self._selection_ranges

        self._clear_data_items()
        if self.__values is None:
            return

        if not self.__group_var:
            self._set_violin_item(self.__values, QColor(Qt.lightGray))
        else:
            assert self.__group_values is not None
            for index in range(len(self.__group_var.values)):
                mask = self.__group_values == index
                color = QColor(*self.__group_var.colors[index])
                self._set_violin_item(self.__values[mask], color)

            self.order_items()

        # apply selection ranges
        self._selection_ranges = ranges

    def _set_violin_item(self, values: np.ndarray, color: QColor):
        values = values[~np.isnan(values)]

        violin = ViolinItem(values, color, self.__kernel, self.__scale,
                            self.__show_rug_plot, self.__orientation)
        self.addItem(violin)
        self.__violin_items.append(violin)

        box = BoxItem(values, violin.boundingRect(), self.__orientation)
        box.setVisible(self.__show_box_plot)
        self.addItem(box)
        self.__box_items.append(box)

        median = MedianItem(values, self.__orientation)
        median.setVisible(self.__show_box_plot)
        self.addItem(median)
        self.__median_items.append(median)

        strip = StripItem(values, violin.density, color, self.__orientation)
        strip.setVisible(self.__show_strip_plot)
        self.addItem(strip)
        self.__strip_items.append(strip)

        width = self._max_item_width * self.SELECTION_PADDING_FACTOR / \
                self.VIOLIN_PADDING_FACTOR
        if self.__orientation == Qt.Vertical:
            rect = QRectF(-width / 2, median.value, width, 0)
        else:
            rect = QRectF(median.value, -width / 2, 0, width)
        sel_rect = SelectionRect(rect, self.__orientation)
        self.addItem(sel_rect)
        self.__selection_rects.append(sel_rect)

    def clear_plot(self):
        self.clear()
        self._clear_data()
        self._clear_data_items()
        self._clear_axes()
        self._clear_selection()

    def _clear_data(self):
        self.__values = None
        self.__value_var = None
        self.__group_values = None
        self.__group_var = None

    def _clear_data_items(self):
        for i in range(len(self.__violin_items)):
            self.removeItem(self.__violin_items[i])
            self.removeItem(self.__box_items[i])
            self.removeItem(self.__median_items[i])
            self.removeItem(self.__strip_items[i])
            self.removeItem(self.__selection_rects[i])
        self.__violin_items.clear()
        self.__box_items.clear()
        self.__median_items.clear()
        self.__strip_items.clear()
        self.__selection_rects.clear()

    def _clear_axes(self):
        self.setAxisItems({"bottom": AxisItem(orientation="bottom"),
                           "left": AxisItem(orientation="left")})
        Updater.update_axes_titles_font(
            self.parameter_setter.axis_items,
            **self.parameter_setter.titles_settings
        )
        Updater.update_axes_ticks_font(
            self.parameter_setter.axis_items,
            **self.parameter_setter.ticks_settings
        )
        self.getAxis("bottom").setRotateTicks(
            self.parameter_setter.is_vertical_setting
        )

    def _clear_selection(self):
        self.__selection = set()

    def _update_selection(self, p1: QPointF, p2: QPointF, finished: bool):
        # When finished, emit selection_changed.
        if len(self.__selection_rects) == 0:
            return
        assert self._max_item_width > 0

        rect = QRectF(p1, p2).normalized()
        if self.__orientation == Qt.Vertical:
            min_max = rect.y(), rect.y() + rect.height()
            index = int((p1.x() + self._max_item_width / 2) /
                        self._max_item_width)
        else:
            min_max = rect.x(), rect.x() + rect.width()
            index = int((-p1.y() + self._max_item_width / 2) /
                        self._max_item_width)

        index = min(index, len(self.__selection_rects) - 1)
        index = self._sorted_group_indices[index]

        self.__selection_rects[index].selection_range = min_max

        if not finished:
            return

        mask = np.bitwise_and(self.__values >= min_max[0],
                              self.__values <= min_max[1])
        if self.__group_values is not None:
            mask = np.bitwise_and(mask, self.__group_values == index)

        selection = set(np.flatnonzero(mask))
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ShiftModifier:
            remove_mask = self.__group_values == index
            selection |= self.__selection - set(np.flatnonzero(remove_mask))
        if self.__selection != selection:
            self.__selection = selection
            self.selection_changed.emit(sorted(self.__selection),
                                        self._selection_ranges)

    def _deselect(self, finished: bool):
        # When finished, emit selection_changed.
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ShiftModifier:
            return

        for index in range(len(self.__selection_rects)):
            self.__selection_rects[index].selection_range = None
        if self.__selection and finished:
            self.__selection = set()
            self.selection_changed.emit([], [])

    @staticmethod
    def sizeHint() -> QSize:
        return QSize(800, 600)


class OWViolinPlot(OWWidget):
    name = "Violin Plot"
    description = "Visualize the distribution of feature" \
                  " values in a violin plot."
    icon = "icons/ViolinPlot.svg"
    priority = 110
    keywords = ["kernel", "density"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Error(OWWidget.Error):
        no_cont_features = Msg("Plotting requires a numeric feature.")
        not_enough_instances = Msg("Plotting requires at least two instances.")

    KERNELS = ["gaussian", "epanechnikov", "linear"]
    KERNEL_LABELS = ["Normal", "Epanechnikov", "Linear"]
    SCALE_LABELS = ["Area", "Count", "Width"]

    settingsHandler = DomainContextHandler()
    value_var = ContextSetting(None)
    order_by_importance = Setting(False)
    group_var = ContextSetting(None)
    order_grouping_by_importance = Setting(False)
    show_box_plot = Setting(True)
    show_strip_plot = Setting(False)
    show_rug_plot = Setting(False)
    order_violins = Setting(False)
    orientation_index = Setting(1)  # Vertical
    kernel_index = Setting(0)  # Normal kernel
    scale_index = Setting(AREA)
    selection_ranges = Setting([], schema_only=True)
    visual_settings = Setting({}, schema_only=True)

    graph_name = "graph.plotItem"
    buttons_area_orientation = None

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.orig_data: Optional[Table] = None
        self.graph: ViolinPlot = None
        self._value_var_model: VariableListModel = None
        self._group_var_model: VariableListModel = None
        self._value_var_view: ListViewSearch = None
        self._group_var_view: ListViewSearch = None
        self._order_violins_cb: QCheckBox = None
        self._scale_combo: QComboBox = None
        self.selection = []
        self.__pending_selection: List = self.selection_ranges

        self.setup_gui()
        VisualSettingsDialog(
            self, self.graph.parameter_setter.initial_settings
        )

    def setup_gui(self):
        self._add_graph()
        self._add_controls()

    def _add_graph(self):
        box = gui.vBox(self.mainArea)
        self.graph = ViolinPlot(self, self.kernel,
                                self.scale_index, self.orientation,
                                self.show_box_plot, self.show_strip_plot,
                                self.show_rug_plot, self.order_violins)
        self.graph.selection_changed.connect(self.__selection_changed)
        box.layout().addWidget(self.graph)

    def __selection_changed(self, indices: List, ranges: List):
        self.selection_ranges = ranges
        if self.selection != indices:
            self.selection = indices
            self.commit()

    def _add_controls(self):
        self._value_var_model = VariableListModel()
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self._value_var_model)
        sorted_model.sort(0)

        view = self._value_var_view = ListViewSearch()
        view.setModel(sorted_model)
        view.setMinimumSize(QSize(30, 100))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        view.selectionModel().selectionChanged.connect(
            self.__value_var_changed
        )

        self._group_var_model = VariableListModel(placeholder="None")
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self._group_var_model)
        sorted_model.sort(0)

        view = self._group_var_view = ListViewSearch()
        view.setModel(sorted_model)
        view.setMinimumSize(QSize(30, 100))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        view.selectionModel().selectionChanged.connect(
            self.__group_var_changed
        )

        box = gui.vBox(self.controlArea, "Variable")
        box.layout().addWidget(self._value_var_view)
        gui.checkBox(box, self, "order_by_importance",
                     "Order by relevance to subgroups",
                     tooltip="Order by 𝜒² or ANOVA over the subgroups",
                     callback=self.apply_value_var_sorting)

        box = gui.vBox(self.controlArea, "Subgroups")
        box.layout().addWidget(self._group_var_view)
        gui.checkBox(box, self, "order_grouping_by_importance",
                     "Order by relevance to variable",
                     tooltip="Order by 𝜒² or ANOVA over the variable values",
                     callback=self.apply_group_var_sorting)

        box = gui.vBox(self.controlArea, "Display",
                       sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum))
        gui.checkBox(box, self, "show_box_plot", "Box plot",
                     callback=self.__show_box_plot_changed)
        gui.checkBox(box, self, "show_strip_plot", "Strip plot",
                     callback=self.__show_strip_plot_changed)
        gui.checkBox(box, self, "show_rug_plot", "Rug plot",
                     callback=self.__show_rug_plot_changed)
        self._order_violins_cb = gui.checkBox(
            box, self, "order_violins", "Order subgroups",
            callback=self.__order_violins_changed,
        )
        gui.radioButtons(box, self, "orientation_index",
                         ["Horizontal", "Vertical"], label="Orientation: ",
                         orientation=Qt.Horizontal,
                         callback=self.__orientation_changed)

        box = gui.vBox(self.controlArea, "Density Estimation",
                       sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum))
        gui.comboBox(box, self, "kernel_index", items=self.KERNEL_LABELS,
                     label="Kernel:", labelWidth=60, orientation=Qt.Horizontal,
                     callback=self.__kernel_changed)
        self._scale_combo = gui.comboBox(
            box, self, "scale_index", items=self.SCALE_LABELS,
            label="Scale:", labelWidth=60, orientation=Qt.Horizontal,
            callback=self.__scale_changed
        )

    def __value_var_changed(self, selection: QItemSelection):
        if not selection:
            return
        self.value_var = selection.indexes()[0].data(gui.TableVariable)
        self.apply_group_var_sorting()
        self.setup_plot()
        self.__selection_changed([], [])

    def __group_var_changed(self, selection: QItemSelection):
        if not selection:
            return
        self.group_var = selection.indexes()[0].data(gui.TableVariable)
        self.apply_value_var_sorting()
        self.enable_controls()
        self.setup_plot()
        self.__selection_changed([], [])

    def __show_box_plot_changed(self):
        self.graph.set_show_box_plot(self.show_box_plot)

    def __show_strip_plot_changed(self):
        self.graph.set_show_strip_plot(self.show_strip_plot)

    def __show_rug_plot_changed(self):
        self.graph.set_show_rug_plot(self.show_rug_plot)

    def __order_violins_changed(self):
        self.graph.set_sort_items(self.order_violins)

    def __orientation_changed(self):
        self.graph.set_orientation(self.orientation)

    def __kernel_changed(self):
        self.graph.set_kernel(self.kernel)

    def __scale_changed(self):
        self.graph.set_scale(self.scale_index)

    @property
    def kernel(self) -> str:
        # pylint: disable=invalid-sequence-index
        return self.KERNELS[self.kernel_index]

    @property
    def orientation(self) -> Qt.Orientations:
        # pylint: disable=invalid-sequence-index
        return [Qt.Horizontal, Qt.Vertical][self.orientation_index]

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.clear()
        self.orig_data = self.data = data
        self.check_data()
        self.init_list_view()
        self.openContext(self.data)
        self.set_list_view_selection()
        self.apply_value_var_sorting()
        self.apply_group_var_sorting()
        self.enable_controls()
        self.setup_plot()
        self.apply_selection()

    def check_data(self):
        self.clear_messages()
        if self.data is not None:
            if self.data.domain.has_continuous_attributes(True, True) == 0:
                self.Error.no_cont_features()
                self.data = None
            elif len(self.data) < 2:
                self.Error.not_enough_instances()
                self.data = None

    def init_list_view(self):
        if not self.data:
            return

        domain = self.data.domain
        self._value_var_model[:] = [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_continuous and not var.attributes.get("hidden", False)]
        self._group_var_model[:] = [None] + [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_discrete and not var.attributes.get("hidden", False)]

        if len(self._value_var_model) > 0:
            self.value_var = self._value_var_model[0]

        self.group_var = self._group_var_model[0]
        if domain.class_var and domain.class_var.is_discrete:
            self.group_var = domain.class_var

    def set_list_view_selection(self):
        for view, var, callback in ((self._value_var_view, self.value_var,
                                     self.__value_var_changed),
                                    (self._group_var_view, self.group_var,
                                     self.__group_var_changed)):
            src_model = view.model().sourceModel()
            if var not in src_model:
                continue
            sel_model = view.selectionModel()
            sel_model.selectionChanged.disconnect(callback)
            row = src_model.indexOf(var)
            index = view.model().index(row, 0)
            sel_model.select(index, sel_model.ClearAndSelect)
            self._ensure_selection_visible(view)
            sel_model.selectionChanged.connect(callback)

    def apply_value_var_sorting(self):
        def compute_score(attr):
            if attr is group_var:
                return 3
            col = self.data.get_column_view(attr)[0].astype(float)
            groups = (col[group_col == i] for i in range(n_groups))
            groups = (col[~np.isnan(col)] for col in groups)
            groups = [group for group in groups if len(group)]
            p = stats.f_oneway(*groups)[1] if len(groups) > 1 else 2
            if np.isnan(p):
                return 2
            return p

        if self.data is None:
            return
        group_var = self.group_var
        if self.order_by_importance and group_var is not None:
            n_groups = len(group_var.values)
            group_col = self.data.get_column_view(group_var)[0].astype(float)
            self._sort_list(self._value_var_model, self._value_var_view,
                            compute_score)
        else:
            self._sort_list(self._value_var_model, self._value_var_view, None)

    def apply_group_var_sorting(self):
        def compute_stat(group):
            if group is value_var:
                return 3
            if group is None:
                return -1
            col = self.data.get_column_view(group)[0].astype(float)
            groups = (value_col[col == i] for i in range(len(group.values)))
            groups = (col[~np.isnan(col)] for col in groups)
            groups = [group for group in groups if len(group)]
            p = stats.f_oneway(*groups)[1] if len(groups) > 1 else 2
            if np.isnan(p):
                return 2
            return p

        if self.data is None:
            return
        value_var = self.value_var
        if self.order_grouping_by_importance:
            value_col = self.data.get_column_view(value_var)[0].astype(float)
            self._sort_list(self._group_var_model, self._group_var_view,
                            compute_stat)
        else:
            self._sort_list(self._group_var_model, self._group_var_view, None)

    def _sort_list(self, source_model, view, key=None):
        if key is None:
            c = count()

            def key(_):  # pylint: disable=function-redefined
                return next(c)

        for i, attr in enumerate(source_model):
            source_model.setData(source_model.index(i), key(attr), Qt.UserRole)
        self._ensure_selection_visible(view)

    @staticmethod
    def _ensure_selection_visible(view):
        selection = view.selectedIndexes()
        if len(selection) == 1:
            view.scrollTo(selection[0])

    def enable_controls(self):
        enable = self.group_var is not None or not self.data
        self._order_violins_cb.setEnabled(enable)
        self._scale_combo.setEnabled(enable)

    def setup_plot(self):
        self.graph.clear_plot()
        if not self.data:
            return

        y = self.data.get_column_view(self.value_var)[0].astype(float)
        x = None
        if self.group_var:
            x = self.data.get_column_view(self.group_var)[0].astype(float)
        self.graph.set_data(y, self.value_var, x, self.group_var)

    def apply_selection(self):
        if self.__pending_selection:
            # commit is invoked on selection_changed
            self.selection_ranges = self.__pending_selection
            self.__pending_selection = []
            self.graph.set_selection(self.selection_ranges)
        else:
            self.commit()

    def commit(self):
        selected = None
        if self.data is not None and bool(self.selection):
            selected = self.data[self.selection]
        annotated = create_annotated_table(self.orig_data, self.selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def clear(self):
        self._value_var_model[:] = []
        self._group_var_model[:] = []
        self.selection = []
        self.selection_ranges = []
        self.graph.clear_plot()

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        # pylint: disable=unsupported-assignment-operation
        self.visual_settings[key] = value


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWViolinPlot).run(set_data=Table("heart_disease"))
