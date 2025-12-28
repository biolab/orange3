import sys
import itertools
import warnings
from collections import Counter
from typing import Callable
from xml.sax.saxutils import escape
from datetime import datetime, timezone

import numpy as np
from AnyQt.QtCore import Qt, QRectF, QSize, QTimer, pyqtSignal as Signal, \
    QObject, QEvent
from AnyQt.QtGui import QColor, QPen, QBrush, QPainterPath, QTransform, \
    QPainter, QPalette, QTextOption
from AnyQt.QtWidgets import QApplication, QToolTip, QGraphicsTextItem, \
    QGraphicsRectItem, QGraphicsItemGroup

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
from pyqtgraph.graphicsItems.LegendItem import LegendItem as PgLegendItem
from pyqtgraph.graphicsItems.TextItem import TextItem

from Orange.preprocess.discretize import _time_binnings
from Orange.util import utc_from_timestamp
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import classdensity, colorpalettes
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import (
    HelpEventDelegate as EventDelegate, InteractiveViewBox as ViewBox,
    PaletteItemSample, SymbolItemSample, AxisItem, PlotWidget, DiscretizedScale
)

SELECTION_WIDTH = 5
MAX_N_VALID_SIZE_ANIMATE = 1000

# maximum number of colors (including Other)
MAX_COLORS = 11


class LegendItem(PgLegendItem):
    items = []  # Accessed in changeEvent after delete in tests??

    def __init__(
            self, size=None, offset=None, pen=None, brush=None,
    ):
        super().__init__(size, offset)
        self.items = []
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setHorizontalSpacing(15)
        self.layout.setColumnAlignment(1, Qt.AlignLeft | Qt.AlignVCenter)
        if pen is not None:
            pen = QPen(pen)
        if brush is not None:
            brush = QBrush(brush)
        self.__pen = pen
        self.__brush = brush

    def restoreAnchor(self, anchors):
        """
        Restore (parent) relative position from stored anchors.

        The restored position is within the parent bounds.
        """
        anchor, parentanchor = anchors
        self.anchor(*bound_anchor_pos(anchor, parentanchor))

    # pylint: disable=arguments-differ
    def paint(self, painter, _option, _widget=None):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        rect = self.contentsRect()
        painter.drawRoundedRect(rect, 2, 2)

    def addItem(self, item, name):
        super().addItem(item, name)
        # Fix-up the label alignment, and color
        color = self.palette().color(QPalette.Text)
        _, label = self.items[-1]
        label.setText(name, justify="left", color=color)

    def clear(self):
        """
        Clear all legend items.
        """
        items = list(self.items)
        self.items = []
        for sample, label in items:
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
            sample.hide()
            label.hide()

        self.updateSize()

    def pen(self):
        if self.__pen is not None:
            return QPen(self.__pen)
        else:
            color = self.palette().color(QPalette.Disabled, QPalette.Text)
            color.setAlpha(100)
            pen = QPen(color, 1)
            pen.setCosmetic(True)
            return pen

    def brush(self):
        if self.__brush is not None:
            return QBrush(self.__brush)
        else:
            color = self.palette().color(QPalette.Window)
            color.setAlpha(150)
            return QBrush(color)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.PaletteChange:
            color = self.palette().color(QPalette.Text)
            for _, label in self.items:
                label.setText(label.text, color=color)
        super().changeEvent(event)


def bound_anchor_pos(corner, parentpos):
    corner = np.clip(corner, 0, 1)
    parentpos = np.clip(parentpos, 0, 1)

    irx, iry = corner
    prx, pry = parentpos

    if irx > 0.9 and prx < 0.1:
        irx = prx = 0.0
    if iry > 0.9 and pry < 0.1:
        iry = pry = 0.0
    if irx < 0.1 and prx > 0.9:
        irx = prx = 1.0
    if iry < 0.1 and pry > 0.9:
        iry = pry = 1.0
    return (irx, iry), (prx, pry)


class ScatterPlotItem(pg.ScatterPlotItem):
    """
    Modifies the behaviour of ScatterPlotItem as follows:

    - Add z-index. ScatterPlotItem paints points in order of appearance in
      self.data. Plotting by z-index is achieved by sorting before calling
      super().paint() and re-sorting afterwards. Re-sorting (instead of
      storing the original data) is needed because the inherited paint
      may modify the data.

    - Prevent multiple calls to updateSpots. ScatterPlotItem calls updateSpots
      at any change of sizes/colors/symbols, which then rebuilds the stored
      pixmaps for each symbol. Orange calls set* functions in succession,
      so we postpone updateSpots() to paint()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_spots_in_paint = False
        self._z_mapping = None
        self._inv_mapping = None

    def setZ(self, z):
        """
        Set z values for all points.

        Points with higher values are plotted on top of those with lower.

        Args:
            z (np.ndarray or None): a vector of z values
        """
        if z is None:
            self._z_mapping = self._inv_mapping = None
        else:
            assert len(z) == len(self.data)
            self._z_mapping = np.argsort(z)
            self._inv_mapping = np.argsort(self._z_mapping)

    def setCoordinates(self, x, y):
        """
        Change the coordinates of points while keeping other properties.

        Asserts that the number of points stays the same.

        Note. Pyqtgraph does not offer a method for this: setting coordinates
        invalidates other data. We therefore retrieve the data to set it
        together with the coordinates. Pyqtgraph also does not offer a
        (documented) method for retrieving the data, yet using
        data[prop]` looks reasonably safe.

        The alternative, updating the whole scatterplot from the Orange Table,
        is too slow.
        """
        assert len(self.data) == len(x) == len(y)
        data = dict(x=x, y=y)
        for prop in ('pen', 'brush', 'size', 'symbol', 'data'):
            data[prop] = self.data[prop]
        self.setData(**data)

    def updateSpots(self, dataSet=None):  # pylint: disable=unused-argument
        self._update_spots_in_paint = True
        self.update()

    # pylint: disable=arguments-differ
    def paint(self, painter, option, widget=None):
        # super().paint will reset painter.transform, so we must compute
        # all transformations in advance
        viewmask = self._maskAt(self.viewRect())
        data = self.data[viewmask]
        if len(data) == 0:
            super().paint(painter, option, widget)
            return
        x, y = data["x"], data["y"]
        xmi, xma = np.min(x), np.max(x)
        ymi, yma = np.min(y), np.max(y)

        tr = fn.transformCoordinates(
            painter.transform(),
            np.array([[xmi, ymi], [xma, yma]]).T).T
        width = tr[1, 0] - tr[0, 0] or 1
        height = tr[0, 1] - tr[1, 1] or 1

        NX = int(width / 50) or 1
        NY = int(height / 50) or 1
        cellx = (xma - xmi) / NX or 1
        celly = (yma - ymi) / NY or 1
        xg = np.clip((x - xmi) // cellx, 0, NX - 1)
        yg = np.clip((y - ymi) // celly, 0, NY - 1)
        idx = (xg + NX * yg).astype(int)
        counts = np.bincount(idx, minlength=NX * NY)

        agg_pts = []
        agg_colors = []
        brushes = data["brush"]
        for i, count in enumerate(counts):
            if count < 10:
                continue
            mask = idx == i
            agg_pts.append([np.mean(x[mask]), np.mean(y[mask])])
            agg_colors.append(Counter(brush.color().getRgb() for brush in brushes[mask]))
        if agg_pts:
            agg_pts = fn.transformCoordinates(painter.transform(), np.array(agg_pts).T).T

        visible = np.ones(len(self.data), dtype=bool)
        visible[viewmask] = counts[idx] < 10
        self.data["visible"] = visible
        try:
            if self._z_mapping is not None:
                assert len(self._z_mapping) == len(self.data)
                self.data = self.data[self._z_mapping]
            if self._update_spots_in_paint:
                self._update_spots_in_paint = False
                super().updateSpots()
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            super().paint(painter, option, widget)
        finally:
            self.data["visible"] = True
            if self._inv_mapping is not None:
                self.data = self.data[self._inv_mapping]

        if len(agg_pts) == 0:
            return

        countf = 8 / np.max(counts)
        text_opt = QTextOption()
        text_opt.setAlignment(Qt.AlignCenter)
        for (pt, spot_colors) in zip(agg_pts, agg_colors):
            painter.resetTransform()
            painter.translate(*pt)
            total = sum(spot_colors.values())
            r = int(6 + countf * total)
            if len(spot_colors) == 1:
                color = QColor(*next(iter(spot_colors)))
                painter.setBrush(fn.mkBrush(color))
                painter.setPen(fn.mkPen(color))
                painter.drawEllipse(-r, -r, 2 * r, 2 * r)
                painter.setBrush(fn.mkBrush(None))
                for r in (r + 3, r + 6):
                    painter.setPen(fn.mkPen(color, width=5))
                    painter.drawEllipse(-r, -r, 2 * r, 2 * r)
            else:
                start_angle = 0
                for color, count in spot_colors.items():
                    angle_span = 360 * count / total
                    painter.setBrush(fn.mkBrush(color))
                    painter.setPen(fn.mkPen(color))
                    start = int(16 * start_angle)
                    span = int(16 * angle_span)
                    painter.drawPie(-r, -r, 2 * r, 2 * r, start, span)
                    for nr in (r + 3, r + 6):
                        painter.setPen(fn.mkPen(color, width=5))
                        painter.drawArc(-nr, -nr, 2 * nr, 2 * nr, start, span)
                    start_angle += angle_span

            painter.setBrush(fn.mkBrush(QColor(0, 0, 0)))
            painter.setPen(fn.mkPen(QColor(0, 0, 0)))
            painter.drawText(QRectF(-15, -15, 30, 30), str(total), text_opt)


def _define_symbols():
    """
    Add symbol ? to ScatterPlotItemSymbols,
    reflect the triangle to point upwards
    """
    path = QPainterPath()
    path.addEllipse(QRectF(-0.35, -0.35, 0.7, 0.7))
    path.moveTo(-0.5, 0.5)
    path.lineTo(0.5, -0.5)
    path.moveTo(-0.5, -0.5)
    path.lineTo(0.5, 0.5)
    Symbols["?"] = path

    path = QPainterPath()
    plusCoords = [
        (-0.5, -0.1), (-0.5, 0.1), (-0.1, 0.1), (-0.1, 0.5),
        (0.1, 0.5), (0.1, 0.1), (0.5, 0.1), (0.5, -0.1),
        (0.1, -0.1), (0.1, -0.5), (-0.1, -0.5), (-0.1, -0.1)
    ]
    path.moveTo(*plusCoords[0])
    for x, y in plusCoords[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    Symbols["+"] = path

    tr = QTransform()
    tr.rotate(180)
    Symbols['t'] = tr.map(Symbols['t'])

    tr = QTransform()
    tr.rotate(45)
    Symbols['x'] = tr.map(Symbols["+"])


_define_symbols()


def _make_pen(color, width):
    p = QPen(color, width)
    p.setCosmetic(True)
    return p


class AxisItem(AxisItem):
    """
    Axis that if needed displays ticks appropriate for time data.
    """

    _label_width = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_time = False

    def use_time(self, enable):
        """Enables axes to display ticks for time data."""
        self._use_time = enable
        self.enableAutoSIPrefix(not enable)

    def is_time(self):
        return self._use_time

    def tickValues(self, minVal, maxVal, size):
        """Find appropriate tick locations."""
        if not self._use_time:
            return super().tickValues(minVal, maxVal, size)

        # if timezone is not set, then local is used which cause exceptions
        minVal = max(minVal,
                     datetime.min.replace(tzinfo=timezone.utc).timestamp() + 1)
        maxVal = min(maxVal,
                     datetime.max.replace(tzinfo=timezone.utc).timestamp() - 1)
        mn = utc_from_timestamp(minVal).timetuple()
        mx = utc_from_timestamp(maxVal).timetuple()
        try:
            bins = _time_binnings(mn, mx, 6, 30)[-1]
        except (IndexError, ValueError):
            # cannot handle very large and very small time intervals
            return super().tickValues(minVal, maxVal, size)

        ticks = bins.thresholds
        # Remove ticks that will later be removed in AxisItem.generateDrawSpecs
        # because they are out of range. Removing them here is needed so that
        # they do not affect spaces and label format
        ticks = ticks[int((ticks[0] < minVal))
                      :len(ticks) - int((ticks[-1] > maxVal))]

        max_steps = max(int(size / self._label_width), 1)
        if len(ticks) > max_steps:
            # remove some of ticks so that they don't overlap
            step = int(np.ceil(float(len(ticks)) / max_steps))
            ticks = ticks[::step]

        # In case of a single tick, `default` will inform tickStrings
        # about the appropriate scale.
        spacing = min((b - a for a, b in zip(ticks[:-1], ticks[1:])),
                      default=maxVal - minVal)
        return [(spacing, ticks)]

    def tickStrings(self, values, scale, spacing):
        """Format tick values according to space between them."""
        if not self._use_time:
            return super().tickStrings(values, scale, spacing)

        if spacing >= 3600 * 24 * 365:
            fmt = "%Y"
        elif spacing >= 3600 * 24 * 28:
            fmt = "%Y %b"
        elif spacing >= 3600 * 24:
            fmt = "%Y %b %d"
        elif spacing >= 3600:
            min_day = max_day = 1
            if len(values) > 0:
                min_day = datetime.fromtimestamp(
                    min(values), tz=timezone.utc).day
                max_day = datetime.fromtimestamp(
                    max(values), tz=timezone.utc).day
            if min_day == max_day:
                fmt = "%Hh"
            else:
                fmt = "%d %Hh"
        elif spacing >= 60:
            fmt = "%H:%M"
        elif spacing >= 1:
            fmt = "%H:%M:%S"
        else:
            fmt = '%S.%f'

        return [utc_from_timestamp(x).strftime(fmt) for x in values]


class ScatterBaseParameterSetter(CommonParameterSetter):
    CAT_LEGEND_LABEL = "Categorical legend"
    NUM_LEGEND_LABEL = "Numerical legend"
    NUM_LEGEND_SETTING = {
        Updater.SIZE_LABEL: (range(4, 50), 11),
        Updater.IS_ITALIC_LABEL: (None, False),
    }

    def __init__(self, master):
        super().__init__()
        self.master = master
        self.cat_legend_settings = {}
        self.num_legend_settings = {}

    def update_setters(self):
        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.LABEL_LABEL: self.FONT_SETTING,
                self.CAT_LEGEND_LABEL: self.FONT_SETTING,
                self.NUM_LEGEND_LABEL: self.NUM_LEGEND_SETTING,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
            }
        }

        def update_cat_legend(**settings):
            self.cat_legend_settings.update(**settings)
            Updater.update_legend_font(self.cat_legend_items, **settings)

        def update_num_legend(**settings):
            self.num_legend_settings.update(**settings)
            Updater.update_num_legend_font(self.num_legend, **settings)

        labels = self.LABELS_BOX
        self._setters[labels][self.CAT_LEGEND_LABEL] = update_cat_legend
        self._setters[labels][self.NUM_LEGEND_LABEL] = update_num_legend

    @property
    def title_item(self):
        return self.master.plot_widget.getPlotItem().titleLabel

    @property
    def cat_legend_items(self):
        items = self.master.color_legend.items
        if items and items[0] and isinstance(items[0][0], PaletteItemSample):
            items = []
        return itertools.chain(self.master.shape_legend.items, items)

    @property
    def num_legend(self):
        items = self.master.color_legend.items
        if items and items[0] and isinstance(items[0][0], PaletteItemSample):
            return self.master.color_legend
        return None

    @property
    def labels(self):
        return self.master.labels


class OWScatterPlotBase(gui.OWComponent, QObject):
    """
    Provide a graph component for widgets that show any kind of point plot

    The component plots a set of points with given coordinates, shapes,
    sizes and colors. Its function is similar to that of a *view*, whereas
    the widget represents a *model* and a *controler*.

    The model (widget) needs to provide methods:

    - `get_coordinates_data`, `get_size_data`, `get_color_data`,
      `get_shape_data`, `get_label_data`, which return a 1d array (or two
      arrays, for `get_coordinates_data`) of `dtype` `float64`, except for
      `get_label_data`, which returns formatted labels;
    - `get_shape_labels` returns a list of strings for shape legend
    - `get_color_labels` returns strings for color legend, or a function for
       formatting numbers if the legend is continuous, or None for default
       formatting
    - `get_tooltip`, which gives a tooltip for a single data point
    - (optional) `impute_sizes`, `impute_shapes` get final coordinates and
      shapes, and replace nans;
    - `get_subset_mask` returns a bool array indicating whether a
      data point is in the subset or not (e.g. in the 'Data Subset' signal
      in the Scatter plot and similar widgets);
    - `get_palette` returns a palette appropriate for visualizing the
      current color data;
    - `is_continuous_color` decides the type of the color legend;

    The widget (in a role of controller) must also provide methods
    - `selection_changed`

    If `get_coordinates_data` returns `(None, None)`, the plot is cleared. If
    `get_size_data`, `get_color_data` or `get_shape_data` return `None`,
    all points will have the same size, color or shape, respectively.
    If `get_label_data` returns `None`, there are no labels.

    The view (this compomnent) provides methods `update_coordinates`,
    `update_sizes`, `update_colors`, `update_shapes` and `update_labels`
    that the widget (in a role of a controler) should call when any of
    these properties are changed. If the widget calls, for instance, the
    plot's `update_colors`, the plot will react by calling the widget's
    `get_color_data` as well as the widget's methods needed to construct the
    legend.

    The view also provides a method `reset_graph`, which should be called only
    when
    - the widget gets entirely new data
    - the number of points may have changed, for instance when selecting
    a different attribute for x or y in the scatter plot, where the points
    with missing x or y coordinates are hidden.

    Every `update_something` calls the plot's `get_something`, which
    calls the model's `get_something_data`, then it transforms this data
    into whatever is needed (colors, shapes, scaled sizes) and changes the
    plot. For the simplest example, here is `update_shapes`:

    ```
        def update_shapes(self):
            if self.scatterplot_item:
                shape_data = self.get_shapes()
                self.scatterplot_item.setSymbol(shape_data)
            self.update_legends()

        def get_shapes(self):
            shape_data = self.master.get_shape_data()
            shape_data = self.master.impute_shapes(
                shape_data, len(self.CurveSymbols) - 1)
            return self.CurveSymbols[shape_data]
    ```

    On the widget's side, `get_something_data` is essentially just:

    ```
        def get_size_data(self):
            return self.get_column(self.attr_size)
    ```

    where `get_column` retrieves a column while also filtering out the
    points with missing x and y and so forth. (Here we present the simplest
    two cases, "shapes" for the view and "sizes" for the model. The colors
    for the view are more complicated since they deal with discrete and
    continuous palettes, and the shapes for the view merge infrequent shapes.)

    The plot can also show just a random sample of the data. The sample size is
    set by `set_sample_size`, and the rest is taken care by the plot: the
    widget keeps providing the data for all points, selection indices refer
    to the entire set etc. Internally, sampling happens as early as possible
    (in methods `get_<something>`).
    """
    too_many_labels = Signal(bool)
    begin_resizing = Signal()
    step_resizing = Signal()
    end_resizing = Signal()

    label_only_selected = Setting(False)
    point_width = Setting(10)
    alpha_value = Setting(128)
    show_grid = Setting(False)
    show_legend = Setting(True)
    class_density = Setting(False)
    jitter_size = Setting(0)

    resolution = 256

    CurveSymbols = np.array("o x t + d star ?".split())
    MinShapeSize = 6
    DarkerValue = 120
    UnknownColor = (168, 50, 168)

    COLOR_DEFAULT = (128, 128, 128)

    MAX_VISIBLE_LABELS = 500

    def __init__(self, scatter_widget, parent=None, view_box=ViewBox):
        QObject.__init__(self)
        gui.OWComponent.__init__(self, scatter_widget)

        self.subset_is_shown = False
        self.jittering_suspended = False

        self.view_box = view_box(self)
        _axis = {"left": AxisItem("left"), "bottom": AxisItem("bottom")}
        self.plot_widget = PlotWidget(
            viewBox=self.view_box, parent=parent, background=None,
            axisItems=_axis
        )
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.getPlotItem().buttonsHidden = True
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QSize(500, 500)

        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.labels = []

        self.master = scatter_widget
        tooltip = self._create_drag_tooltip()
        self.view_box.setDragTooltip(tooltip)

        self.selection = None  # np.ndarray

        self.n_valid = 0
        self.n_shown = 0
        self.sample_size = None
        self.sample_indices = None

        self.palette = None

        self.shape_legend = self._create_legend(((1, 0), (1, 0)))
        self.color_legend = self._create_legend(((1, 1), (1, 1)))
        self.update_legend_visibility()

        self.scale = None  # DiscretizedScale
        self._too_many_labels = False

        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.update_grid_visibility()

        self._tooltip_delegate = EventDelegate(self.help_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.view_box.sigTransformChanged.connect(self.update_density)
        self.view_box.sigRangeChangedManually.connect(self.update_labels)

        self.timer = None

        self.parameter_setter = ScatterBaseParameterSetter(self)

    def _create_legend(self, anchor):
        legend = LegendItem()
        legend.setParentItem(self.plot_widget.getViewBox())
        legend.restoreAnchor(anchor)
        return legend

    def _create_drag_tooltip(self):
        tip_parts = [
            (Qt.ControlModifier,
             "{}: Append to group".
             format("Cmd" if sys.platform == "darwin" else "Ctrl")),
            (Qt.ShiftModifier, "Shift: Add group"),
            (Qt.AltModifier, "Alt: Remove")
        ]
        all_parts = "<center>" + \
                    ", ".join(part for _, part in tip_parts) + \
                    "</center>"
        self.tiptexts = {
            modifier: all_parts.replace(part, "<b>{}</b>".format(part))
            for modifier, part in tip_parts
        }
        self.tiptexts[Qt.NoModifier] = all_parts

        self.tip_textitem = text = QGraphicsTextItem()
        # Set to the longest text
        text.setHtml(self.tiptexts[Qt.ControlModifier])
        text.setPos(4, 2)
        r = text.boundingRect()
        text.setTextWidth(r.width())
        rect = QGraphicsRectItem(0, 0, r.width() + 8, r.height() + 4)
        color = self.plot_widget.palette().color(QPalette.Disabled, QPalette.Window)
        color.setAlpha(212)
        rect.setBrush(color)
        rect.setPen(QPen(Qt.NoPen))
        self.update_tooltip()

        tooltip_group = QGraphicsItemGroup()
        tooltip_group.addToGroup(rect)
        tooltip_group.addToGroup(text)
        return tooltip_group

    def update_tooltip(self, modifiers=Qt.NoModifier):
        text = self.tiptexts[Qt.NoModifier]
        for mod in [Qt.ControlModifier,
                    Qt.ShiftModifier,
                    Qt.AltModifier]:
            if modifiers & mod:
                text = self.tiptexts.get(mod)
                break
        self.tip_textitem.setHtml(text)

    def suspend_jittering(self):
        if self.jittering_suspended:
            return
        self.jittering_suspended = True
        if self.jitter_size != 0:
            self.update_jittering()

    def unsuspend_jittering(self):
        if not self.jittering_suspended:
            return
        self.jittering_suspended = False
        if self.jitter_size != 0:
            self.update_jittering()

    def update_jittering(self):
        x, y = self.get_coordinates()
        if x is None or len(x) == 0 or self.scatterplot_item is None:
            return
        self.scatterplot_item.setCoordinates(x, y)
        self.scatterplot_item_sel.setCoordinates(x, y)
        self.update_labels()

    # TODO: Rename to remove_plot_items
    def clear(self):
        """
        Remove all graphical elements from the plot

        Calls the pyqtgraph's plot widget's clear, sets all handles to `None`,
        removes labels and selections.

        This method should generally not be called by the widget. If the data
        is gone (*e.g.* upon receiving `None` as an input data signal), this
        should be handler by calling `reset_graph`, which will in turn call
        `clear`.

        Derived classes should override this method if they add more graphical
        elements. For instance, the regression line in the scatterplot adds
        `self.reg_line_item = None` (the line in the plot is already removed
        in this method).
        """
        self.plot_widget.clear()

        self.density_img = None
        if self.timer is not None and self.timer.isActive():
            self.timer.stop()
            self.timer = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.labels = []
        self._signal_too_many_labels(False)
        self.view_box.init_history()
        self.view_box.tag_history()

    # TODO: I hate `keep_something` and `reset_something` arguments
    # __keep_selection is used exclusively be set_sample size which would
    # otherwise just repeat the code from reset_graph except for resetting
    # the selection. I'm uncomfortable with this; we may prefer to have a
    # method _reset_graph which does everything except resetting the selection,
    # and reset_graph would call it.
    def reset_graph(self, __keep_selection=False):
        """
        Reset the graph to new data (or no data)

        The method must be called when the plot receives new data, in
        particular when the number of points change. If only their properties
        - like coordinates or shapes - change, an update method
        (`update_coordinates`, `update_shapes`...) should be called instead.

        The method must also be called when the data is gone.

        The method calls `clear`, followed by calls of all update methods.

        NB. Argument `__keep_selection` is for internal use only
        """
        self.clear()
        if not __keep_selection:
            self.selection = None
        self.sample_indices = None
        self.update_coordinates()
        self.update_point_props()

    def set_sample_size(self, sample_size):
        """
        Set the sample size

        Args:
            sample_size (int or None): sample size or `None` to show all points
        """
        if self.sample_size != sample_size:
            self.sample_size = sample_size
            self.reset_graph(True)

    def update_point_props(self):
        """
        Update the sizes, colors, shapes and labels

        The method calls the appropriate update methods for individual
        properties.
        """
        self.update_sizes()
        self.update_colors()
        self.update_selection_colors()
        self.update_shapes()
        self.update_labels()

    # Coordinates
    # TODO: It could be nice if this method was run on entire data, not just
    # a sample. For this, however, it would need to either be called from
    # `get_coordinates` before sampling (very ugly) or call
    # `self.master.get_coordinates_data` (beyond ugly) or the widget would
    # have to store the ranges of unsampled data (ugly).
    # Maybe we leave it as it is.
    def _reset_view(self, x_data, y_data):
        """
        Set the range of the view box

        Args:
            x_data (np.ndarray): x coordinates
            y_data (np.ndarray) y coordinates
        """
        min_x, max_x = np.min(x_data), np.max(x_data)
        min_y, max_y = np.min(y_data), np.max(y_data)
        self.view_box.setRange(
            QRectF(min_x, min_y, max_x - min_x or 1, max_y - min_y or 1),
            padding=0.025)

    def _filter_visible(self, data):
        """Return the sample from the data using the stored sample_indices"""
        if data is None or self.sample_indices is None:
            return data
        else:
            return np.asarray(data[self.sample_indices])

    def get_coordinates(self):
        """
        Prepare coordinates of the points in the plot

        The method is called by `update_coordinates`. It gets the coordinates
        from the widget, jitters them and return them.

        The methods also initializes the sample indices if neededd and stores
        the original and sampled number of points.

        Returns:
            (tuple): a pair of numpy arrays containing (sampled) coordinates,
                or `(None, None)`.
        """
        x, y = self.master.get_coordinates_data()
        if x is None:
            self.n_valid = self.n_shown = 0
            return None, None
        self.n_valid = len(x)
        self._create_sample()
        x = self._filter_visible(x)
        y = self._filter_visible(y)
        # Jittering after sampling is OK if widgets do not change the sample
        # semi-permanently, e.g. take a sample for the duration of some
        # animation. If the sample size changes dynamically (like by adding
        # a "sample size" slider), points would move around when the sample
        # size changes. To prevent this, jittering should be done before
        # sampling (i.e. two lines earlier). This would slow it down somewhat.
        x, y = self.jitter_coordinates(x, y)
        return x, y

    def _create_sample(self):
        """
        Create a random sample if the data is larger than the set sample size
        """
        self.n_shown = min(self.n_valid, self.sample_size or self.n_valid)
        if self.sample_size is not None \
                and self.sample_indices is None \
                and self.n_valid != self.n_shown:
            random = np.random.RandomState(seed=0)
            self.sample_indices = random.choice(
                self.n_valid, self.n_shown, replace=False)
            # TODO: Is this really needed?
            np.sort(self.sample_indices)

    def jitter_coordinates(self, x, y):
        """
        Display coordinates to random positions within ellipses with
        radiuses of `self.jittter_size` percents of spans
        """
        if self.jitter_size == 0 or self.jittering_suspended:
            return x, y
        return self._jitter_data(x, y)

    def _jitter_data(self, x, y, span_x=None, span_y=None):
        if span_x is None:
            span_x = np.max(x) - np.min(x)
        if span_y is None:
            span_y = np.max(y) - np.min(y)
        random = np.random.RandomState(seed=0)
        rs = random.uniform(0, 1, len(x))
        phis = random.uniform(0, 2 * np.pi, len(x))
        magnitude = self.jitter_size / 100
        return (x + magnitude * span_x * rs * np.cos(phis),
                y + magnitude * span_y * rs * np.sin(phis))

    def update_coordinates(self):
        """
        Trigger the update of coordinates while keeping other features intact.

        The method gets the coordinates by calling `self.get_coordinates`,
        which in turn calls the widget's `get_coordinate_data`. The number of
        coordinate pairs returned by the latter must match the current number
        of points. If this is not the case, the widget should trigger
        the complete update by calling `reset_graph` instead of this method.
        """
        x, y = self.get_coordinates()
        if x is None or len(x) == 0:
            return

        self._reset_view(x, y)
        if self.scatterplot_item is None:
            if self.sample_indices is None:
                indices = np.arange(self.n_valid)
            else:
                indices = self.sample_indices
            kwargs = dict(x=x, y=y, data=indices)
            self.scatterplot_item = ScatterPlotItem(**kwargs)
            self.scatterplot_item.sigClicked.connect(self.select_by_click)
            self.scatterplot_item_sel = ScatterPlotItem(**kwargs)
            self.plot_widget.addItem(self.scatterplot_item_sel)
            self.plot_widget.addItem(self.scatterplot_item)
        else:
            self.scatterplot_item.setCoordinates(x, y)
            self.scatterplot_item_sel.setCoordinates(x, y)
            self.update_labels()

        self.update_density()  # Todo: doesn't work: try MDS with density on

    # Sizes
    def get_sizes(self):
        """
        Prepare data for sizes of points in the plot

        The method is called by `update_sizes`. It gets the sizes
        from the widget and performs the necessary scaling and sizing.
        The output is rounded to half a pixel for faster drawing.

        Returns:
            (np.ndarray): sizes
        """
        size_column = self.master.get_size_data()
        if size_column is None:
            return np.full((self.n_shown,),
                           self.MinShapeSize + (5 + self.point_width) * 0.5)
        size_column = self._filter_visible(size_column)
        size_column = size_column.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            size_column -= np.nanmin(size_column)
            mx = np.nanmax(size_column)
        if mx > 0:
            size_column /= mx
        else:
            size_column[:] = 0.5

        sizes = self.MinShapeSize + (5 + self.point_width) * size_column
        # round sizes to half pixel for smaller pyqtgraph's symbol pixmap atlas
        sizes = (sizes * 2).round() / 2
        return sizes

    def update_sizes(self):
        """
        Trigger an update of point sizes

        The method calls `self.get_sizes`, which in turn calls the widget's
        `get_size_data`. The result are properly scaled and then passed
        back to widget for imputing (`master.impute_sizes`).
        """
        if self.scatterplot_item:
            size_data = self.get_sizes()
            size_imputer = getattr(
                self.master, "impute_sizes", self.default_impute_sizes)
            size_imputer(size_data)

            if self.timer is not None and self.timer.isActive():
                self.timer.stop()
                self.timer = None

            current_size_data = self.scatterplot_item.data["size"].copy()
            diff = size_data - current_size_data
            widget = self

            class Timeout:
                # 0.5 - np.cos(np.arange(0.17, 1, 0.09) * np.pi) / 2
                factors = [0.07, 0.16, 0.27, 0.41, 0.55,
                           0.68, 0.81, 0.9, 0.97, 1]

                def __init__(self):
                    self._counter = 0

                def __call__(self):
                    factor = self.factors[self._counter]
                    self._counter += 1
                    size = current_size_data + diff * factor
                    if len(self.factors) == self._counter:
                        widget.timer.stop()
                        widget.timer = None
                        size = size_data
                    widget.scatterplot_item.setSize(size)
                    widget.scatterplot_item_sel.setSize(size + SELECTION_WIDTH)
                    if widget.timer is None:
                        widget.end_resizing.emit()
                    else:
                        widget.step_resizing.emit()

            if self.n_valid <= MAX_N_VALID_SIZE_ANIMATE and \
                    np.all(current_size_data > 0) and np.any(diff != 0):
                # If encountered any strange behaviour when updating sizes,
                # implement it with threads
                self.begin_resizing.emit()
                interval = int(500 / len(Timeout.factors))
                self.timer = QTimer(self.scatterplot_item, interval=interval)
                self.timer.timeout.connect(Timeout())
                self.timer.start()
            else:
                self.begin_resizing.emit()
                self.scatterplot_item.setSize(size_data)
                self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)
                self.end_resizing.emit()

    update_point_size = update_sizes  # backward compatibility (needed?!)
    update_size = update_sizes

    @classmethod
    def default_impute_sizes(cls, size_data):
        """
        Fallback imputation for sizes.

        Set the size to two pixels smaller than the minimal size

        Returns:
            (bool): True if there was any missing data
        """
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = cls.MinShapeSize - 2
            return True
        else:
            return False

    # Colors
    def get_colors(self):
        """
        Prepare data for colors of the points in the plot

        The method is called by `update_colors`. It gets the colors and the
        indices of the data subset from the widget (`get_color_data`,
        `get_subset_mask`), and constructs lists of pens and brushes for
        each data point.

        The method uses different palettes for discrete and continuous data,
        as determined by calling the widget's method `is_continuous_color`.

        If also marks the points that are in the subset as defined by, for
        instance the 'Data Subset' signal in the Scatter plot and similar
        widgets. (Do not confuse this with *selected points*, which are
        marked by circles around the points, which are colored by groups
        and thus independent of this method.)

        Returns:
            (tuple): a list of pens and list of brushes
        """
        c_data = self.master.get_color_data()
        c_data = self._filter_visible(c_data)
        subset = self.master.get_subset_mask()
        subset = self._filter_visible(subset)
        self.subset_is_shown = subset is not None
        if c_data is None:  # same color
            self.palette = None
            return self._get_same_colors(subset)
        elif self.master.is_continuous_color():
            return self._get_continuous_colors(c_data, subset)
        else:
            return self._get_discrete_colors(c_data, subset)

    def _get_same_colors(self, subset, color=COLOR_DEFAULT):
        """
        Return the same pen for all points while the brush color depends
        upon whether the point is in the subset or not

        Args:
            subset (np.ndarray): a bool array indicating whether a data point
                is in the subset or not (e.g. in the 'Data Subset' signal
                in the Scatter plot and similar widgets);

        Returns:
            (tuple): a list of pens and list of brushes
        """
        alpha_subset, alpha_unset = self._alpha_for_subsets()
        if subset is not None:
            qcolor = QColor(*color, alpha_subset)
            brush = np.where(subset, QBrush(qcolor), QBrush(QColor(0, 0, 0, 0)))
            pen = np.where(subset,
                           _make_pen(qcolor, 1.5),
                           _make_pen(QColor(*color, alpha_unset), 1.5))
        else:
            qcolor = QColor(*color, self.alpha_value)
            brush = np.full(self.n_shown, QBrush(qcolor))
            pen = [_make_pen(qcolor, 1.5)] * self.n_shown
        return pen, brush

    def _get_continuous_colors(self, c_data, subset):
        """
        Return the pens and colors whose color represent an index into
        a continuous palette. The same color is used for pen and brush,
        except the former is darker. If the data has a subset, the brush
        is transparent for points that are not in the subset.
        """
        palette = self.master.get_palette()

        if np.isnan(c_data).all():
            self.palette = palette
            return self._get_same_colors(subset, self.palette.nan_color)

        self.scale = DiscretizedScale(np.nanmin(c_data), np.nanmax(c_data))
        bins = self.scale.get_bins()
        self.palette = \
            colorpalettes.BinnedContinuousPalette.from_palette(palette, bins)
        colors = self.palette.values_to_colors(c_data)
        alphas = np.full((len(c_data), 1), self.alpha_value, dtype=np.ubyte)
        brush = np.hstack((colors, alphas))
        pen = np.hstack(
            ((colors.astype(dtype=float) * 100 / self.DarkerValue).astype(np.ubyte),
             alphas))

        # Reuse pens and brushes with the same colors because PyQtGraph then
        # builds smaller pixmap atlas, which makes the drawing faster

        def reuse(cache, fun, *args):
            if args not in cache:
                cache[args] = fun(args)
            return cache[args]

        def create_pen(col):
            return _make_pen(QColor(*col), 1.5)

        def create_brush(col):
            return QBrush(QColor(*col))

        if subset is not None:
            alpha_subset, alpha_unset = self._alpha_for_subsets()
            brush[:, 3] = 0
            brush[subset, 3] = alpha_subset
            pen[:, 3] = alpha_unset
            brush[subset, 3] = alpha_subset

        cached_pens = {}
        pen = [reuse(cached_pens, create_pen, *col) for col in pen.tolist()]
        cached_brushes = {}
        brush = np.array([reuse(cached_brushes, create_brush, *col)
                          for col in brush.tolist()])

        return pen, brush

    def _get_discrete_colors(self, c_data, subset):
        """
        Return the pens and colors whose color represent an index into
        a discrete palette. The same color is used for pen and brush,
        except the former is darker. If the data has a subset, the brush
        is transparent for points that are not in the subset.
        """
        self.palette = self.master.get_palette()
        c_data = c_data.copy()
        c_data[np.isnan(c_data)] = len(self.palette)
        c_data = c_data.astype(int)
        colors = self.palette.qcolors_w_nan
        if subset is None:
            for col in colors:
                col.setAlpha(self.alpha_value)
            pens = np.array(
                [_make_pen(col.darker(self.DarkerValue), 1.5)
                 for col in colors])
            pen = pens[c_data]
            brushes = np.array([QBrush(col) for col in colors])
            brush = brushes[c_data]
        else:
            subset_colors = [QColor(col) for col in colors]
            alpha_subset, alpha_unset = self._alpha_for_subsets()
            for col in subset_colors:
                col.setAlpha(alpha_subset)
            for col in colors:
                col.setAlpha(alpha_unset)

            pens, subset_pens = (
                np.array(
                    [_make_pen(col.darker(self.DarkerValue), 1.5)
                     for col in cols])
                for cols in (colors, subset_colors))
            pen = np.where(subset, subset_pens[c_data], pens[c_data])

            brushes = np.array([QBrush(col) for col in subset_colors])
            brush = brushes[c_data]
            black = np.full(len(brush), QBrush(QColor(0, 0, 0, 0)))
            brush = np.where(subset, brush, black)
        return pen, brush

    def _alpha_for_subsets(self):
        a, b, c = 1.2, -3.2, 3
        x = self.alpha_value / 255
        alpha_subset = 31 + int(224 * (a * x ** 3 + b * x ** 2 + c * x))
        x = 1 - x
        alpha_unset = int(255 - 224 * (a * x ** 3 + b * x ** 2 + c * x))
        return alpha_subset, alpha_unset

    def update_colors(self):
        """
        Trigger an update of point colors

        The method calls `self.get_colors`, which in turn calls the widget's
        `get_color_data` to get the indices in the pallette. `get_colors`
        returns a list of pens and brushes to which this method uses to
        update the colors. Finally, the method triggers the update of the
        legend and the density plot.
        """
        if self.scatterplot_item is not None:
            pen_data, brush_data = self.get_colors()
            self.scatterplot_item.setPen(pen_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
        self.update_z_values()
        self.update_legends()
        self.update_density()

    update_alpha_value = update_colors

    def update_density(self):
        """
        Remove the existing density plot (if there is one) and replace it
        with a new one (if enabled).

        The method gets the colors from the pens of the currently plotted
        points.
        """
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
            self.density_img = None
        if self.class_density and self.scatterplot_item is not None:
            c_data = self.master.get_color_data()
            if c_data is None:
                return
            visible_c_data = self._filter_visible(c_data)
            mask = np.isfinite(visible_c_data)
            if not self.master.is_continuous_color():
                mask = np.bitwise_and(mask, visible_c_data < MAX_COLORS - 1)
            pens = self.scatterplot_item.data['pen']
            rgb_data = [
                pen.color().getRgb()[:3] if pen is not None else (255, 255, 255)
                for known, pen in zip(mask, pens)
                if known]
            if len(set(rgb_data)) <= 1:
                return
            [min_x, max_x], [min_y, max_y] = self.view_box.viewRange()
            x_data, y_data = self.scatterplot_item.getData()
            self.density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution,
                x_data[mask], y_data[mask], rgb_data)
            self.plot_widget.addItem(self.density_img, ignoreBounds=True)

    def update_selection_colors(self):
        """
        Trigger an update of selection markers

        This update method is usually not called by the widget but by the
        plot, since it is the plot that handles the selections.

        Like other update methods, it calls the corresponding get method
        (`get_colors_sel`) which returns a list of pens and brushes.
        """
        if self.scatterplot_item_sel is None:
            return
        pen, brush = self.get_colors_sel()
        self.scatterplot_item_sel.setPen(pen, update=False, mask=None)
        self.scatterplot_item_sel.setBrush(brush, mask=None)
        self.update_z_values()

    def get_colors_sel(self):
        """
        Return pens and brushes for selection markers.

        A pen can is set to `Qt.NoPen` if a point is not selected.

        All brushes are completely transparent whites.

        Returns:
            (tuple): a list of pens and a list of brushes
        """
        nopen = QPen(Qt.NoPen)
        if self.selection is None:
            pen = [nopen] * self.n_shown
        else:
            sels = np.max(self.selection)
            if sels == 1:
                pen = np.where(
                    self._filter_visible(self.selection),
                    _make_pen(QColor(255, 190, 0, 255), SELECTION_WIDTH),
                    nopen)
            else:
                palette = colorpalettes.LimitedDiscretePalette(
                    number_of_colors=sels + 1)
                pen = np.choose(
                    self._filter_visible(self.selection),
                    [nopen] + [_make_pen(palette[i], SELECTION_WIDTH)
                               for i in range(sels)])
        return pen, [QBrush(QColor(255, 255, 255, 0))] * self.n_shown

    # Labels
    def get_labels(self):
        """
        Prepare data for labels for points

        The method returns the results of the widget's `get_label_data`

        Returns:
            (labels): a sequence of labels
        """
        return self._filter_visible(self.master.get_label_data())

    def update_labels(self):
        """
        Trigger an update of labels

        The method calls `get_labels` which in turn calls the widget's
        `get_label_data`. The obtained labels are shown if the corresponding
        points are selected or if `label_only_selected` is `false`.
        """
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []

        mask = None
        if self.scatterplot_item is not None:
            x, y = self.scatterplot_item.getData()
            mask = self._label_mask(x, y)

        if mask is not None:
            labels = self.get_labels()
            if labels is None:
                mask = None

        self._signal_too_many_labels(
            bool(mask is not None and mask.sum() > self.MAX_VISIBLE_LABELS))
        if self._too_many_labels or mask is None or not np.any(mask):
            return

        foreground = self.plot_widget.palette().color(QPalette.Text)
        labels = labels[mask]
        x = x[mask]
        y = y[mask]
        for label, xp, yp in zip(labels, x, y):
            ti = TextItem(label, foreground)
            ti.setPos(xp, yp)
            self.plot_widget.addItem(ti)
            self.labels.append(ti)
            ti.setFont(self.parameter_setter.label_font)

    def _signal_too_many_labels(self, too_many):
        if self._too_many_labels != too_many:
            self._too_many_labels = too_many
            self.too_many_labels.emit(too_many)

    def _label_mask(self, x, y):
        (x0, x1), (y0, y1) = self.view_box.viewRange()
        mask = np.logical_and(
            np.logical_and(x >= x0, x <= x1),
            np.logical_and(y >= y0, y <= y1))
        if self.label_only_selected:
            sub_mask = self._filter_visible(self.master.get_subset_mask())
            if self.selection is None:
                if sub_mask is None:
                    return None
                else:
                    sel_mask = sub_mask
            else:
                sel_mask = self._filter_visible(self.selection) != 0
                if sub_mask is not None:
                    sel_mask = np.logical_or(sel_mask, sub_mask)
            mask = np.logical_and(mask, sel_mask)
        return mask

    # Shapes
    def get_shapes(self):
        """
        Prepare data for shapes of points in the plot

        The method is called by `update_shapes`. It gets the data from
        the widget's `get_shape_data`, and then calls its `impute_shapes`
        to impute the missing shape (usually with some default shape).

        Returns:
            (np.ndarray): an array of symbols (e.g. o, x, + ...)
        """
        shape_data = self.master.get_shape_data()
        shape_data = self._filter_visible(shape_data)
        # Data has to be copied so the imputation can change it in-place
        # TODO: Try avoiding this when we move imputation to the widget
        if shape_data is not None:
            shape_data = np.copy(shape_data)
        shape_imputer = getattr(
            self.master, "impute_shapes", self.default_impute_shapes)
        shape_imputer(shape_data, len(self.CurveSymbols) - 1)
        if isinstance(shape_data, np.ndarray):
            shape_data = shape_data.astype(int)
        else:
            shape_data = np.zeros(self.n_shown, dtype=int)
        return self.CurveSymbols[shape_data]

    @staticmethod
    def default_impute_shapes(shape_data, default_symbol):
        """
        Fallback imputation for shapes.

        Use the default symbol, usually the last symbol in the list.

        Returns:
            (bool): True if there was any missing data
        """
        if shape_data is None:
            return False
        nans = np.isnan(shape_data)
        if np.any(nans):
            shape_data[nans] = default_symbol
            return True
        else:
            return False

    def update_shapes(self):
        """
        Trigger an update of point symbols

        The method calls `get_shapes` to obtain an array with a symbol
        for each point and uses it to update the symbols.

        Finally, the method updates the legend.
        """
        if self.scatterplot_item:
            shape_data = self.get_shapes()
            self.scatterplot_item.setSymbol(shape_data)
        self.update_legends()

    def update_z_values(self):
        """
        Set z-values for point in the plot

        The order is as follows:
        - selected points that are also in the subset on top,
        - followed by selected points,
        - followed by points from the subset,
        - followed by the rest.
        Within each of these four groups, points are ordered by their colors.

        Points with less frequent colors are above those with more frequent.
        The points for which the value for the color is missing are at the
        bottom of their respective group.
        """
        if not self.scatterplot_item:
            return

        subset = self.master.get_subset_mask()
        c_data = self.master.get_color_data()
        if subset is None and self.selection is None and c_data is None:
            self.scatterplot_item.setZ(None)
            return

        z = np.zeros(self.n_shown)

        if subset is not None:
            subset = self._filter_visible(subset)
            z[subset] += 1000

        if self.selection is not None:
            z[self._filter_visible(self.selection) != 0] += 2000

        if c_data is not None:
            c_nan = np.isnan(c_data)
            vis_data = self._filter_visible(c_data)
            vis_nan = np.isnan(vis_data)
            z[vis_nan] -= 999
            if not self.master.is_continuous_color():
                dist = np.bincount(c_data[~c_nan].astype(int))
                vis_knowns = vis_data[~vis_nan].astype(int)
                argdist = np.argsort(dist)
                z[~vis_nan] -= argdist[vis_knowns]

        self.scatterplot_item.setZ(z)

    def update_grid_visibility(self):
        """Show or hide the grid"""
        self.plot_widget.showGrid(x=self.show_grid, y=self.show_grid)

    def update_legend_visibility(self):
        """
        Show or hide legends based on whether they are enabled and non-empty
        """
        self.shape_legend.setVisible(
            self.show_legend and bool(self.shape_legend.items))
        self.color_legend.setVisible(
            self.show_legend and bool(self.color_legend.items))

    def update_legends(self):
        """Update content of legends and their visibility"""
        cont_color = self.master.is_continuous_color()
        shape_labels = self.master.get_shape_labels()
        color_labels = self.master.get_color_labels()
        if not cont_color and shape_labels is not None \
                and shape_labels == color_labels:
            colors = self.master.get_color_data()
            shapes = self.master.get_shape_data()
            mask = np.isfinite(colors) * np.isfinite(shapes)
            combined = (colors == shapes)[mask].all()
        else:
            combined = False
        if combined:
            self._update_combined_legend(shape_labels)
        else:
            self._update_shape_legend(shape_labels)
            if cont_color:
                self._update_continuous_color_legend(color_labels)
            else:
                self._update_color_legend(color_labels)
        self.update_legend_visibility()
        Updater.update_legend_font(self.parameter_setter.cat_legend_items,
                                   **self.parameter_setter.cat_legend_settings)
        Updater.update_num_legend_font(self.parameter_setter.num_legend,
                                       **self.parameter_setter.num_legend_settings)

    def _update_shape_legend(self, labels):
        self.shape_legend.clear()
        if labels is None or self.scatterplot_item is None:
            return
        color = QColor(0, 0, 0)
        color.setAlpha(self.alpha_value)
        for label, symbol in zip(labels, self.CurveSymbols):
            self.shape_legend.addItem(
                SymbolItemSample(pen=color, brush=color, size=10, symbol=symbol),
                escape(label))

    def _update_continuous_color_legend(self, label_formatter: Callable[[float], str]):
        self.color_legend.clear()
        if self.scale is None or self.scatterplot_item is None:
            return
        label = PaletteItemSample(self.palette, self.scale, label_formatter)
        self.color_legend.addItem(label, "")
        self.color_legend.setGeometry(label.boundingRect())

    def _update_color_legend(self, labels):
        self.color_legend.clear()
        if labels is None:
            return
        self._update_colored_legend(self.color_legend, labels, 'o')

    def _update_combined_legend(self, labels):
        # update_colored_legend will already clear the shape legend
        # so we remove colors here
        use_legend = \
            self.shape_legend if self.shape_legend.items else self.color_legend
        self.color_legend.clear()
        self.shape_legend.clear()
        self._update_colored_legend(use_legend, labels, self.CurveSymbols)

    def _update_colored_legend(self, legend, labels, symbols):
        if self.scatterplot_item is None or not self.palette:
            return
        if isinstance(symbols, str):
            symbols = itertools.repeat(symbols, times=len(labels))
        colors = self.palette.values_to_colors(np.arange(len(labels)))
        for color, label, symbol in zip(colors, labels, symbols):
            color = QColor(*color)
            pen = _make_pen(color.darker(self.DarkerValue), 1.5)
            color.setAlpha(self.alpha_value)
            brush = QBrush(color)
            legend.addItem(
                SymbolItemSample(pen=pen, brush=brush, size=10, symbol=symbol),
                escape(label))

    def zoom_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().RectMode)

    def pan_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().PanMode)

    def select_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().RectMode)

    def reset_button_clicked(self):
        self.plot_widget.getViewBox().autoRange()
        self.update_labels()

    def select_by_click(self, _, points):
        if self.scatterplot_item is not None:
            self.select(points)

    def select_by_rectangle(self, rect):
        if self.scatterplot_item is not None:
            x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
            y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
            x, y = self.master.get_coordinates_data()
            indices = np.flatnonzero(
                (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1))
            self.select_by_indices(indices.astype(int))

    def unselect_all(self):
        if self.selection is not None:
            self.selection = None
            self.update_selection_colors()
            if self.label_only_selected:
                self.update_labels()
            self.master.selection_changed()

    def select(self, points):
        # noinspection PyArgumentList
        if self.scatterplot_item is None:
            return
        indices = [p.data() for p in points]
        self.select_by_indices(indices)

    def select_by_indices(self, indices):
        if self.selection is None:
            self.selection = np.zeros(self.n_valid, dtype=np.uint8)
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ControlModifier:
            self.selection_append(indices)
        elif keys & Qt.ShiftModifier:
            self.selection_new_group(indices)
        elif keys & Qt.AltModifier:
            self.selection_remove(indices)
        else:
            self.selection_select(indices)

    def selection_select(self, indices):
        self.selection = np.zeros(self.n_valid, dtype=np.uint8)
        self.selection[indices] = 1
        self._update_after_selection()

    def selection_append(self, indices):
        self.selection[indices] = max(np.max(self.selection), 1)
        self._update_after_selection()

    def selection_new_group(self, indices):
        self.selection[indices] = np.max(self.selection) + 1
        self._update_after_selection()

    def selection_remove(self, indices):
        self.selection[indices] = 0
        self._update_after_selection()

    def _update_after_selection(self):
        self._compress_indices()
        self.update_selection_colors()
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def _compress_indices(self):
        indices = sorted(set(self.selection) | {0})
        if len(indices) == max(indices) + 1:
            return
        mapping = np.zeros((max(indices) + 1,), dtype=int)
        for i, ind in enumerate(indices):
            mapping[ind] = i
        self.selection = mapping[self.selection]

    def get_selection(self):
        if self.selection is None:
            return np.array([], dtype=np.uint8)
        else:
            return np.flatnonzero(self.selection)

    def help_event(self, event):
        """
        Create a `QToolTip` for the point hovered by the mouse
        """
        if self.scatterplot_item is None:
            return False
        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        point_data = [p.data() for p in self.scatterplot_item.pointsAt(act_pos)]
        text = self.master.get_tooltip(point_data)
        if text:
            QToolTip.showText(event.screenPos(), text, widget=self.plot_widget)
            return True
        else:
            return False
