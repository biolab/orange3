from collections import Counter, defaultdict
import sys
import itertools
import warnings
from xml.sax.saxutils import escape
from math import log2, log10, floor, ceil

import numpy as np
from scipy import sparse as sp

from AnyQt.QtCore import Qt, QRectF, QPointF, QSize
from AnyQt.QtGui import (
    QStaticText, QColor, QPen, QBrush, QPainterPath, QTransform, QPainter
)
from AnyQt.QtWidgets import (
    QApplication, QToolTip, QGraphicsTextItem, QGraphicsRectItem
)

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.LegendItem import LegendItem, ItemSample
from pyqtgraph.graphicsItems.TextItem import TextItem

from Orange.statistics.util import bincount
from Orange.util import OrangeDeprecationWarning
from Orange.widgets import gui
from Orange.widgets.utils import classdensity
from Orange.widgets.utils.colorpalette import (
    ColorPaletteGenerator, ContinuousPaletteGenerator, DefaultRGBColors
)
from Orange.widgets.utils.plot import OWPalette, OWPlotGUI
from Orange.widgets.visualize.utils.plotutils import (
    HelpEventDelegate as EventDelegate,
    InteractiveViewBox as ViewBox
)
from Orange.widgets.visualize.owscatterplotgraph_obsolete import (
    OWScatterPlotGraph as OWScatterPlotGraphObs
)
from Orange.widgets.settings import Setting, ContextSetting
from Orange.widgets.widget import OWWidget, Msg


SELECTION_WIDTH = 5
MAX = 11  # maximum number of colors or shapes (including Other)
MAX_POINTS_IN_TOOLTIP = 5


class PaletteItemSample(ItemSample):
    """A color strip to insert into legends for discretized continuous values"""

    def __init__(self, palette, scale):
        """
        :param palette: palette used for showing continuous values
        :type palette: ContinuousPaletteGenerator
        :param scale: an instance of DiscretizedScale that defines the
                      conversion of values into bins
        :type scale: DiscretizedScale
        """
        super().__init__(None)
        self.palette = palette
        self.scale = scale
        cuts = ["{0:.{1}f}".format(scale.offset + i * scale.width, scale.decimals)
                for i in range(scale.bins + 1)]
        self.labels = [QStaticText("{} - {}".format(fr, to))
                       for fr, to in zip(cuts, cuts[1:])]
        for label in self.labels:
            label.prepare()
        self.text_width = max(label.size().width() for label in self.labels)

    def boundingRect(self):
        return QRectF(0, 0, 40 + self.text_width, 20 + self.scale.bins * 15)

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        scale = self.scale
        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        for i, label in enumerate(self.labels):
            color = QColor(*self.palette.getRGB((i + 0.5) / scale.bins))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(0, i * 15, 15, 15)
            p.setPen(QPen(Qt.black))
            p.drawStaticText(20, i * 15 + 1, label)


class LegendItem(LegendItem):
    def __init__(self, size=None, offset=None, pen=None, brush=None):
        super().__init__(size, offset)

        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setHorizontalSpacing(15)
        self.layout.setColumnAlignment(1, Qt.AlignLeft | Qt.AlignVCenter)

        if pen is None:
            pen = QPen(QColor(196, 197, 193, 200), 1)
            pen.setCosmetic(True)
        self.__pen = pen

        if brush is None:
            brush = QBrush(QColor(232, 232, 232, 100))
        self.__brush = brush

    def storeAnchor(self):
        """
        Return the current relative anchor position (relative to the parent)
        """
        anchor = legend_anchor_pos(self)
        if anchor is None:
            anchor = ((1, 0), (1, 0))
        return anchor

    def restoreAnchor(self, anchors):
        """
        Restore (parent) relative position from stored anchors.

        The restored position is within the parent bounds.
        """
        anchor, parentanchor = anchors
        self.anchor(*bound_anchor_pos(anchor, parentanchor))

    def setPen(self, pen):
        """Set the legend frame pen."""
        pen = QPen(pen)
        if pen != self.__pen:
            self.prepareGeometryChange()
            self.__pen = pen
            self.updateGeometry()

    def pen(self):
        """Pen used to draw the legend frame."""
        return QPen(self.__pen)

    def setBrush(self, brush):
        """Set background brush"""
        brush = QBrush(brush)
        if brush != self.__brush:
            self.__brush = brush
            self.update()

    def brush(self):
        """Background brush."""
        return QBrush(self._brush)

    def paint(self, painter, option, widget=None):
        painter.setPen(self.__pen)
        painter.setBrush(self.__brush)
        rect = self.contentsRect()
        painter.drawRoundedRect(rect, 2, 2)

    def addItem(self, item, name):
        super().addItem(item, name)
        # Fix-up the label alignment
        _, label = self.items[-1]
        label.setText(name, justify="left")

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


ANCHORS = {
    Qt.TopLeftCorner: (0, 0),
    Qt.TopRightCorner: (1, 0),
    Qt.BottomLeftCorner: (0, 1),
    Qt.BottomRightCorner: (1, 1)
}


def corner_anchor(corner):
    """Return the relative corner coordinates for Qt.Corner
    """
    return ANCHORS[corner]


def legend_anchor_pos(legend):
    """
    Return the legend's anchor positions relative to it's parent (if defined).

    Return `None` if legend does not have a parent or the parent's size
    is empty.

    .. seealso:: LegendItem.anchor, rect_anchor_pos

    """
    parent = legend.parentItem()
    if parent is None or parent.size().isEmpty():
        return None

    rect = legend.geometry()  # in parent coordinates.
    parent_rect = QRectF(QPointF(0, 0), parent.size())

    # Find the closest corner of rect to parent rect
    c1, _, *parentPos = rect_anchor_pos(rect, parent_rect)
    return corner_anchor(c1), tuple(parentPos)


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


def rect_anchor_pos(rect, parent_rect):
    """
    Find the 'best' anchor corners of rect within parent_rect.

    Return a tuple of (rect_corner, parent_corner, rx, ry),
    where rect/parent_corners are Qt.Corners which are closest and
    rx, ry are the relative positions of the rect_corner within
    parent_rect. If the parent_rect is empty return `None`.

    """
    if parent_rect.isEmpty():
        return None

    # Find the closest corner of rect to parent rect
    corners = (Qt.TopLeftCorner, Qt.TopRightCorner,
               Qt.BottomRightCorner, Qt.BottomLeftCorner)

    def rect_corner(rect, corner):
        if corner == Qt.TopLeftCorner:
            return rect.topLeft()
        elif corner == Qt.TopRightCorner:
            return rect.topRight()
        elif corner == Qt.BottomLeftCorner:
            return rect.bottomLeft()
        elif corner == Qt.BottomRightCorner:
            return rect.bottomRight()
        else:
            assert False

    def corner_dist(c1, c2):
        d = (rect_corner(rect, c1) - rect_corner(parent_rect, c2))
        return d.x() ** 2 + d.y() ** 2

    if parent_rect.contains(rect):
        closest = min(corners,
                      key=lambda corner: corner_dist(corner, corner))
        p = rect_corner(rect, closest)

        return (closest, closest,
                (p.x() - parent_rect.left()) / parent_rect.width(),
                (p.y() - parent_rect.top()) / parent_rect.height())
    else:

        c1, c2 = min(itertools.product(corners, corners),
                     key=lambda pair: corner_dist(*pair))

        p = rect_corner(rect, c1)

        return (c1, c2,
                (p.x() - parent_rect.left()) / parent_rect.width(),
                (p.y() - parent_rect.top()) / parent_rect.height())


class DiscretizedScale:
    """
    Compute suitable bins for continuous value from its minimal and
    maximal value.

    The width of the bin is a power of 10 (including negative powers).
    The minimal value is rounded up and the maximal is rounded down. If this
    gives less than 3 bins, the width is divided by four; if it gives
    less than 6, it is halved.

    .. attribute:: offset
        The start of the first bin.

    .. attribute:: width
        The width of the bins

    .. attribute:: bins
        The number of bins

    .. attribute:: decimals
        The number of decimals used for printing out the boundaries
    """
    def __init__(self, min_v, max_v):
        """
        :param min_v: Minimal value
        :type min_v: float
        :param max_v: Maximal value
        :type max_v: float
        """
        super().__init__()
        dif = max_v - min_v if max_v != min_v else 1
        if np.isnan(dif):
            min_v = 0
            dif = decimals = 1
        else:
            decimals = -floor(log10(dif))
        resolution = 10 ** -decimals
        bins = ceil(dif / resolution)
        if bins < 6:
            decimals += 1
            if bins < 3:
                resolution /= 4
            else:
                resolution /= 2
            bins = ceil(dif / resolution)
        self.offset = resolution * floor(min_v // resolution)
        self.bins = bins
        self.decimals = max(decimals, 0)
        self.width = resolution

    def compute_bins(self, a):
        """
        Compute bin number(s) for the given value(s).

        :param a: value(s)
        :type a: a number or numpy.ndarray
        """
        a = (a - self.offset) / self.width
        if isinstance(a, np.ndarray):
            a.clip(0, self.bins - 1)
        else:
            a = min(self.bins - 1, max(0, a))
        return a


class InteractiveViewBox(ViewBox):
    def __init__(self, graph, enable_menu=False):
        super().__init__(graph, enable_menu)
        warnings.warn("InteractiveViewBox class has been deprecated since "
                      "3.17. Use Orange.widgets.visualize.utils.plotutils."
                      "InteractiveViewBox instead.", OrangeDeprecationWarning)


class OWScatterPlotGraph(OWScatterPlotGraphObs):
    def __init__(self, scatter_widget, parent=None, _="None", view_box=InteractiveViewBox):
        super().__init__(scatter_widget, parent=parent, _=_, view_box=view_box)
        warnings.warn("OWScatterPlotGraph class has been deprecated since "
                      "3.17. Use OWScatterPlotBase instead.",
                      OrangeDeprecationWarning)


class ScatterPlotItem(pg.ScatterPlotItem):
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        super().paint(painter, option, widget)


def _define_symbols():
    """
    Add symbol ? to ScatterPlotItemSymbols,
    reflect the triangle to point upwards
    """
    symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols
    path = QPainterPath()
    path.addEllipse(QRectF(-0.35, -0.35, 0.7, 0.7))
    path.moveTo(-0.5, 0.5)
    path.lineTo(0.5, -0.5)
    path.moveTo(-0.5, -0.5)
    path.lineTo(0.5, 0.5)
    symbols["?"] = path

    tr = QTransform()
    tr.rotate(180)
    symbols['t'] = tr.map(symbols['t'])


_define_symbols()


def _make_pen(color, width):
    p = QPen(color, width)
    p.setCosmetic(True)
    return p


class OWScatterPlotBase(gui.OWComponent):
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
    - `get_color_labels`, `get_shape_labels`, which are return lists of
       strings used for the color and shape legend;
    - `get_tooltip`, which gives a tooltip for a single data point
    - (optional) `impute_sizes`, `impute_shapes` get final coordinates and
      shapes, and replace nans;
    - `get_subset_mask` returns a bool array indicating whether a
      data point is in the subset or not (e.g. in the 'Data Subset' signal
      in the Scatter plot and similar widgets);
    - `set_palette` sets the plot's palette appropriate for visualizing the
      current color data;
    - `is_continuous_color` decides the type of the color legend;
    - `combined_legend` tells whether the color and shape legend should be
      combined into one (usually because they represent the same data).

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
    `get_color_data` as well as the widget's methods needed to contruct the
    legend.

    The view also provides a method `reset`, which should be called only
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
            self.make_legend()

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
    """
    label_only_selected = Setting(False)
    point_width = Setting(10)
    alpha_value = Setting(128)
    show_grid = Setting(False)
    show_legend = Setting(True)
    class_density = Setting(False)
    jitter_size = Setting(0)

    resolution = 256

    CurveSymbols = np.array("o x t + d s t2 t3 p h star ?".split())
    MinShapeSize = 6
    DarkerValue = 120
    UnknownColor = (168, 50, 168)

    def __init__(self, scatter_widget, parent=None, view_box=ViewBox):
        super().__init__(scatter_widget)

        self.subset_is_shown = False

        self.view_box = view_box(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, parent=parent,
                                         background="w")
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.getPlotItem().buttonsHidden = True
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QSize(500, 500)
        scene = self.plot_widget.scene()
        self._create_drag_tooltip(scene)

        self.replot = self.plot_widget.replot
        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None

        self.labels = []

        self.master = scatter_widget

        self.selection = None  # np.ndarray
        self.n_points = 0

        self.gui = OWPlotGUI(self)
        self.palette = None

        self.legend = self.color_legend = None
        self.__legend_anchor = (1, 0), (1, 0)
        self.__color_legend_anchor = (1, 1), (1, 1)

        self.scale = None  # DiscretizedScale

        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.update_grid()

        self._tooltip_delegate = EventDelegate(self.help_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)

    def _create_drag_tooltip(self, scene):
        tip_parts = [
            (Qt.ShiftModifier, "Shift: Add group"),
            (Qt.ShiftModifier + Qt.ControlModifier,
             "Shift-{}: Append to group".
             format("Cmd" if sys.platform == "darwin" else "Ctrl")),
            (Qt.AltModifier, "Alt: Remove")
        ]
        all_parts = ", ".join(part for _, part in tip_parts)
        self.tiptexts = {
            int(modifier): all_parts.replace(part, "<b>{}</b>".format(part))
            for modifier, part in tip_parts
        }
        self.tiptexts[0] = all_parts

        self.tip_textitem = text = QGraphicsTextItem()
        # Set to the longest text
        text.setHtml(self.tiptexts[Qt.ShiftModifier + Qt.ControlModifier])
        text.setPos(4, 2)
        r = text.boundingRect()
        rect = QGraphicsRectItem(0, 0, r.width() + 8, r.height() + 4)
        rect.setBrush(QColor(224, 224, 224, 212))
        rect.setPen(QPen(Qt.NoPen))
        self.update_tooltip(Qt.NoModifier)

        scene.drag_tooltip = scene.createItemGroup([rect, text])
        scene.drag_tooltip.hide()

    def update_tooltip(self, modifiers):
        modifiers &= Qt.ShiftModifier + Qt.ControlModifier + Qt.AltModifier
        text = self.tiptexts.get(int(modifiers), self.tiptexts[0])
        self.tip_textitem.setHtml(text)

    def clear(self):
        """
        Remove all graphical elements from the plot

        Calls the pyqtgraph's plot widget's clear, removes the legend(s) and
        resets the(ir) anchors, sets all handles to `None`, removes labels and
        selections.

        This method should generally not be called by the widget. If the data
        is gone (*e.g.* upon receiving `None` as an input data signal), this
        should be handler by calling `reset_graph`, which will in turn call
        `clear`.

        Derived classes should override this method if they add more graphical
        elements. For instance, the regression line in the scatterplot adds
        `self.reg_line_item = None` (the line in the plot is already removed
        in this method).
        """
        self.remove_legend()
        self.plot_widget.clear()

        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.labels = []
        self.selection = None

    def reset_graph(self):
        """
        Reset the graph to new data (or no data)

        The method must be called when the plot receives new data, in
        particular when the number of points change. If only their properties
        - like coordinates or shapes - change, an update method
        (`update_coordinates`, `update_shapes`...) should be called instead.

        The method must also be called when the data is gone.

        The method calls `clear`, followed by calls of all update methods.
        """
        self.clear()
        self.update_coordinates()
        self.update_point_props()

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
    def _reset_view(self, x_data, y_data):
        """
        Set the range of the view box

        Args:
            x_data (np.ndarray): x coordinates
            y_data (np.ndarray) y coordinates
        """
        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        self.view_box.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.view_box.init_history()
        self.view_box.tag_history()

    def get_coordinates(self):
        """
        Prepare coordinates of the points in the plot

        The method is called by `update_coordinates`. It gets the coordinates
        from the widget, jitters them and return them.

        The method also stores the number of points.

        Returns:
            (tuple): a pair of numpy arrays containing coordinates,
                or `(None, None)`.
        """
        x, y = self.master.get_coordinates_data()
        if x is None:
            self.n_points = 0
            return None, None
        x, y = self.jitter_coordinates(x, y)
        self.n_points = len(x) if x is not None else 0
        return x, y

    def jitter_coordinates(self, x, y):
        if self.jitter_size == 0:
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

    @classmethod
    def _update_plot_coordinates(cls, plot, x, y):
        """
        Change the coordinates of points while keeping other properites

        Note. Pyqtgraph does not offer a method for this: setting coordinates
        invalidates other data. We therefore retrieve the data to set it
        together with the coordinates. Pyqtgraph also does not offer a
        (documented) method for retrieving the data, yet using
        `plot.data[prop]` looks reasonably safe. The alternative, calling
        update for every property would essentially reset the graph, which
        can be time consuming.
        """
        data = dict(x=x, y=y)
        for prop in ('pen', 'brush', 'size', 'symbol', 'data'):
            data[prop] = plot.data[prop]
        plot.setData(**data)

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
        if x is None or not len(x):
            return
        if self.scatterplot_item is None:
            kwargs = {"x": x, "y": y, "data": np.arange(self.n_points)}
            self.scatterplot_item = ScatterPlotItem(**kwargs)
            self.scatterplot_item.sigClicked.connect(self.select_by_click)
            self.scatterplot_item_sel = ScatterPlotItem(**kwargs)
            self.plot_widget.addItem(self.scatterplot_item_sel)
            self.plot_widget.addItem(self.scatterplot_item)
            self.update_point_props()
        else:
            self._update_plot_coordinates(self.scatterplot_item, x, y)
            self._update_plot_coordinates(self.scatterplot_item_sel, x, y)

        self.update_label_coords(x, y)
        self.update_density()  # Todo: doesn't work: try MDS with density on
        self._reset_view(x, y)

    # Sizes
    def get_sizes(self):
        """
        Prepare data for sizes of points in the plot

        The method is called by `update_sizes`. It gets the sizes
        from the widget and performs the necessary scaling and sizing.

        Returns:
            (np.ndarray): sizes
        """
        size_column = self.master.get_size_data()
        if size_column is None:
            return np.ones(self.n_points) * self.point_width
        size_column = size_column.copy()
        size_column -= np.min(size_column)
        mx = np.max(size_column)
        if mx > 0:
            size_column /= mx
        return self.MinShapeSize + self.point_width * size_column

    def update_sizes(self):
        """
        Trigger an update of point sizes

        The method calls `self.get_sizes`, which in turn calls the widget's
        `get_size_data`. The result are properly scaled and then passed
        back to widget for imputing (`master.impute_sizes`).
        """
        if self.scatterplot_item:
            size_data = self.get_sizes()
            self.master.impute_sizes(size_data)
            self.scatterplot_item.setSize(size_data)
            self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)

    update_point_size = update_sizes  # backward compatibility (needed?!)
    update_size = update_sizes

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
        self.master.set_palette()
        c_data = self.master.get_color_data()
        subset = self.master.get_subset_mask()
        self.subset_is_shown = subset is not None
        if c_data is None:  # same color
            return self._get_same_colors(subset)
        elif self.master.is_continuous_color():
            return self._get_continuous_colors(c_data, subset)
        else:
            return self._get_discrete_colors(c_data, subset)

    def _get_same_colors(self, subset):
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
        color = self.plot_widget.palette().color(OWPalette.Data)
        pen = [_make_pen(color, 1.5) for _ in range(self.n_points)]
        if subset is not None:
            brush = [(QBrush(QColor(128, 128, 128, 0)),
                      QBrush(QColor(128, 128, 128, 255)))[s]
                     for s in subset]
        else:
            color = QColor(128, 128, 128, self.alpha_value)
            brush = [QBrush(color) for _ in range(self.n_points)]
        return pen, brush

    def _get_continuous_colors(self, c_data, subset):
        """
        Return the pens and colors whose color represent an index into
        a continuous palette. The same color is used for pen and brush,
        except the former is darker. If the data has a subset, the brush
        is transparent for points that are not in the subset.
        """
        self.scale = DiscretizedScale(np.nanmin(c_data), np.nanmax(c_data))
        c_data -= self.scale.offset
        c_data /= self.scale.width
        c_data = np.floor(c_data) + 0.5
        c_data /= self.scale.bins
        c_data = np.clip(c_data, 0, 1)
        pen = self.palette.getRGB(c_data)
        brush = np.hstack(
            [pen, np.full((len(pen), 1), self.alpha_value, dtype=int)])
        pen *= 100
        pen //= self.DarkerValue
        pen = [_make_pen(QColor(*col), 1.5) for col in pen.tolist()]

        if subset is not None:
            brush[:, 3] = 0
            brush[subset, 3] = 255
        brush = np.array([QBrush(QColor(*col)) for col in brush.tolist()])
        return pen, brush

    def _get_discrete_colors(self, c_data, subset):
        """
        Return the pens and colors whose color represent an index into
        a discrete palette. The same color is used for pen and brush,
        except the former is darker. If the data has a subset, the brush
        is transparent for points that are not in the subset.
        """
        n_colors = self.palette.number_of_colors
        c_data = c_data.copy()
        c_data[np.isnan(c_data)] = n_colors
        c_data = c_data.astype(int)
        colors = np.r_[self.palette.getRGB(np.arange(n_colors)),
                       [[128, 128, 128]]]
        pens = np.array(
            [_make_pen(QColor(*col).darker(self.DarkerValue), 1.5)
             for col in colors])
        pen = pens[c_data]
        alpha = self.alpha_value if subset is None else 255
        brushes = np.array([
            [QBrush(QColor(0, 0, 0, 0)),
             QBrush(QColor(col[0], col[1], col[2], alpha))]
            for col in colors])
        brush = brushes[c_data]

        if subset is not None:
            brush = np.where(subset, brush[:, 1], brush[:, 0])
        else:
            brush = brush[:, 1]
        return pen, brush

    def update_colors(self):
        """
        Trigger an update of point sizes

        The method calls `self.get_colors`, which in turn calls the widget's
        `get_color_data` to get the indices in the pallette. `get_colors`
        returns a list of pens and brushes to which this method uses to
        update the colors. Finally, the method triggets the update of the
        legend and the density plot.
        """
        if self.scatterplot_item is None:
            return
        pen_data, brush_data = self.get_colors()
        self.scatterplot_item.setPen(pen_data, update=False, mask=None)
        self.scatterplot_item.setBrush(brush_data, mask=None)
        self.make_legend()
        self.update_density()

    update_alpha_value = update_colors

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

    def update_density(self):
        """
        Remove the existing density plot (if there is one) and replace it
        with a new one (if enabled).

        The method gets the colors from the pens of the currently plotted
        points.
        """
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
        if self.scatterplot_item is not None \
                and self.master.can_draw_density() and self.class_density:
            [min_x, max_x], [min_y, max_y] = self.view_box.viewRange()
            x_data, y_data = self.scatterplot_item.getData()
            rgb_data = [pen.color().getRgb()[:3]
                        for pen in self.scatterplot_item.data['pen']]
            self.density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution,
                x_data, y_data, rgb_data)
            self.plot_widget.addItem(self.density_img)
        else:
            self.density_img = None

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
            pen = [nopen] * self.n_points
        else:
            sels = np.max(self.selection)
            if sels == 1:
                pen = np.where(
                    self.selection,
                    _make_pen(QColor(255, 190, 0, 255), SELECTION_WIDTH + 1),
                    nopen)
            else:
                palette = ColorPaletteGenerator(number_of_colors=sels + 1)
                pen = np.choose(
                    self.selection,
                    [nopen] + [_make_pen(palette[i], SELECTION_WIDTH + 1)
                               for i in range(sels)])
        return pen, [QBrush(QColor(255, 255, 255, 0))] * self.n_points

    # Labels
    def get_labels(self):
        """
        Prepare data for labels for points

        The method returns the results of the widget's `get_label_data`

        Returns:
            (labels): a sequence of labels
        """
        return self.master.get_label_data()

    def update_labels(self):
        """
        Trigger an updaet of labels

        The method calls `get_labels` which in turn calls the widget's
        `get_label_data`. The obtained labels are shown if the corresponding
        points are selected or if `label_only_selected` is `false`.
        """
        if self.label_only_selected and self.selection is None:
            label_data = None
        else:
            label_data = self.get_labels()
        if label_data is None:
            for label in self.labels:
                label.setText("")
            return
        if not self.labels:
            self._create_labels()
        black = pg.mkColor(0, 0, 0)
        if self.label_only_selected:
            for label, text, selected \
                    in zip(self.labels, label_data, self.selection):
                label.setText(text if selected else "", black)
        else:
            for label, text in zip(self.labels, label_data):
                label.setText(text, black)

    def _create_labels(self):
        """
        Create a `TextItem` for each point and store them in `self.labels`
        """
        if not self.scatterplot_item:
            return
        for x, y in zip(*self.scatterplot_item.getData()):
            ti = TextItem()
            self.plot_widget.addItem(ti)
            ti.setPos(x, y)
            self.labels.append(ti)

    def update_label_coords(self, x, y):
        """
        Update the coordinates of labels

        The method is currently called exclusively be `update_coordinates`

        Args:
            x (np.ndarray): x coordinates
            y (np.ndarray): y coordinates
        """
        if self.label_only_selected:
            if self.selection is not None:
                for label, selected, xp, yp in zip(
                        self.labels, self.selection, x, y):
                    if selected:
                        label.setPos(xp, yp)
        else:
            for label, xp, yp in zip(self.labels, x, y):
                label.setPos(xp, yp)

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
        # Data has to be copied so the imputation can change it in-place
        # TODO: Try avoiding this when we move imputation to the widget
        if shape_data is not None:
            shape_data = np.copy(shape_data)
        self.master.impute_shapes(shape_data, len(self.CurveSymbols) - 1)
        if isinstance(shape_data, np.ndarray):
            shape_data = shape_data.astype(int)
        else:
            shape_data = np.zeros(self.n_points, dtype=int)
        return self.CurveSymbols[shape_data]

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
        self.make_legend()

    def update_grid(self):
        """Show or hide the grid"""
        self.plot_widget.showGrid(x=self.show_grid, y=self.show_grid)

    def update_legend(self):
        """Show or hide the legend"""
        if self.legend:
            self.legend.setVisible(self.show_legend)

    def create_legend(self):
        """Create a legend"""
        self.legend = LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.restoreAnchor(self.__legend_anchor)

    def remove_legend(self):
        """Remove the legend and reset its position"""
        if self.legend:
            anchor = legend_anchor_pos(self.legend)
            if anchor is not None:
                self.__legend_anchor = anchor
            self.legend.setParent(None)
            self.legend = None
        if self.color_legend:
            anchor = legend_anchor_pos(self.color_legend)
            if anchor is not None:
                self.__color_legend_anchor = anchor
            self.color_legend.setParent(None)
            self.color_legend = None

    def make_legend(self):
        """Create the color and shape legends"""
        self.remove_legend()
        if not self.legend:
            self.create_legend()
        self._make_color_legend()
        self._make_shape_legend()
        self.update_legend()

    def _make_color_legend(self):
        """
        Adds items representing the colors to the legend

        - If the legend is continuous (which is checked by calling the
          widget's `is_continuous_color`), the legend is a colored strip.
        - Otherwise, if the same attribute is used for shape and color
          (which is checked by the widget's method `combined_legend`),
          this method returns a legend with different shapes in shown
          in the corresponding color.
        - Otherwise, a normal legend for colors is created.
        """
        if self.master.is_continuous_color():
            if not self.scale:
                return
            legend = self.color_legend = LegendItem()
            legend.setParentItem(self.plot_widget.getViewBox())
            legend.restoreAnchor(self.__color_legend_anchor)

            label = PaletteItemSample(self.palette, self.scale)
            legend.addItem(label, "")
            legend.setGeometry(label.boundingRect())
        else:
            labels = self.master.get_color_labels()
            if not labels or not self.palette:
                return
            use_shape = self.master.combined_legend()
            for i, value in enumerate(labels):
                color = QColor(*self.palette.getRGB(i))
                pen = _make_pen(color.darker(self.DarkerValue), 1.5)
                color.setAlpha(255 if self.subset_is_shown else self.alpha_value)
                brush = QBrush(color)
                self.legend.addItem(
                    ScatterPlotItem(
                        pen=pen, brush=brush, size=10,
                        symbol=self.CurveSymbols[i] if use_shape else "o"),
                    escape(value))

    def _make_shape_legend(self):
        """
        Adds items representing the shapes to the legend

        If the color and shape legends are combined (checked by the widget's
        method `combined_legends`), this method does nothing.
        """
        if self.master.combined_legend():
            return
        labels = self.master.get_shape_labels()
        if labels is None:
            return
        color = QColor(0, 0, 0)
        color.setAlpha(self.alpha_value)
        for i, value in enumerate(labels):
            self.legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10,
                                symbol=self.CurveSymbols[i]), escape(value))

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

    def select_by_click(self, _, points):
        if self.scatterplot_item is not None:
            self.select(points)

    def select_by_rectangle(self, value_rect):
        if self.scatterplot_item is not None:
            points = [point
                      for point in self.scatterplot_item.points()
                      if value_rect.contains(QPointF(point.pos()))]
            self.select(points)

    def select_by_index(self, indices):
        if self.scatterplot_item is not None:
            points = [point for point in self.scatterplot_item.points()
                      if point.data() in indices]
            self.select(points)

    def unselect_all(self):
        self.selection = None
        self.select([])
        self.update_selection_colors()
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def select(self, points):
        # noinspection PyArgumentList
        if self.n_points == 0:
            return
        if self.selection is None:
            self.selection = np.zeros(self.n_points, dtype=np.uint8)
        indices = [p.data() for p in points]
        keys = QApplication.keyboardModifiers()
        # Remove from selection
        if keys & Qt.AltModifier:
            self.selection[indices] = 0
        # Append to the last group
        elif keys & Qt.ShiftModifier and keys & Qt.ControlModifier:
            self.selection[indices] = np.max(self.selection)
        # Create a new group
        elif keys & Qt.ShiftModifier:
            self.selection[indices] = np.max(self.selection) + 1
        # No modifiers: new selection
        else:
            self.selection = np.zeros(self.n_points, dtype=np.uint8)
            self.selection[indices] = 1
        self.update_selection_colors()
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

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

    def box_zoom_select(self, parent):
        g = self.gui
        box_zoom_select = gui.vBox(parent, "Zoom/Select")
        zoom_select_toolbar = g.zoom_select_toolbar(
            box_zoom_select, nomargin=True,
            buttons=[g.StateButtonsBegin, g.SimpleSelect, g.Pan, g.Zoom,
                     g.StateButtonsEnd, g.ZoomReset]
        )
        buttons = zoom_select_toolbar.buttons
        buttons[g.Zoom].clicked.connect(self.zoom_button_clicked)
        buttons[g.Pan].clicked.connect(self.pan_button_clicked)
        buttons[g.SimpleSelect].clicked.connect(self.select_button_clicked)
        buttons[g.ZoomReset].clicked.connect(self.reset_button_clicked)
        return box_zoom_select


class HelpEventDelegate(EventDelegate):
    def __init__(self, delegate, parent=None):
        super().__init__(delegate, parent)
        warnings.warn("HelpEventDelegate class has been deprecated since 3.17."
                      " Use Orange.widgets.visualize.utils.plotutils."
                      "HelpEventDelegate instead.", OrangeDeprecationWarning)


class OWProjectionWidget(OWWidget):
    """
    Base widget for widgets that use attribute data to set the colors, labels,
    shapes and sizes of points.

    The widgets defines settings `attr_color`, `attr_label`, `attr_shape`
    and `attr_size`, but leaves defining the gui to the derived widgets.
    These are expected to have controls that manipulate these settings,
    and the controls are expected to use attribute models.

    The widgets also defines attributes `data` and `valid_data` and expects
    the derived widgets to use them to store an instances of `data.Table`
    and a bool `np.ndarray` with indicators of valid (that is, shown)
    data points.
    """
    attr_color = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_label = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_shape = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_size = ContextSetting(None, required=ContextSetting.OPTIONAL)

    class Information(OWWidget.Information):
        missing_size = Msg(
            "Points with undefined '{}' are shown in smaller size")
        missing_shape = Msg(
            "Points with undefined '{}' are shown as crossed circles")

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = None

        self.set_palette()

    def init_attr_values(self):
        """
        Set the models for `attr_color`, `attr_shape`, `attr_size` and
        `attr_label`. All values are set to `None`, except `attr_color`
        which is set to the class variable if it exists.
        """
        data = self.data
        domain = data.domain if data and len(data) else None
        for attr in ("attr_color", "attr_shape", "attr_size", "attr_label"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        if domain is not None:
            self.attr_color = domain.class_var

    def get_coordinates_data(self):
        """A get coordinated method that returns no coordinates.

        Derived classes must override this method.
        """
        return None, None

    def get_subset_mask(self):
        """
        Return the bool array indicating the points in the subset

        The base method does nothing and would usually be overridden by
        a method that returns indicators from the subset signal.

        Do not confuse the subset with selection.

        Returns:
            (np.ndarray or `None`): a bool array of indicators
        """
        return None

    @staticmethod
    def __get_overlap_groups(x, y):
        coord_to_id = defaultdict(list)
        for i, xy in enumerate(zip(x, y)):
            coord_to_id[xy].append(i)
        return coord_to_id

    def get_column(self, attr, filter_valid=True,
                   merge_infrequent=False, return_labels=False):
        """
        Retrieve the data from the given column in the data table

        The method:
        - densifies sparse data,
        - converts arrays with dtype object to floats if the attribute is
          actually primitive,
        - filters out invalid data (if `filter_valid` is `True`),
        - merges infrequent (discrete) values into a single value
          (if `merge_infrequent` is `True`).

        Tha latter feature is used for shapes and labels, where only a
        set number (`MAX`) of different values is shown, and others are
        merged into category 'Other'. In this case, the method may return
        either the data (e.g. color indices, shape indices) or the list
        of retained values, followed by `['Other']`.

        Args:
            attr (:obj:~Orange.data.Variable): the column to extract
            filter_valid (bool): filter out invalid data (default: `True`)
            merge_infrequent (bool): merge infrequent values (default: `False`)
            return_labels (bool): return a list of labels instead of data
                (default: `False`)

        Returns:
            (np.ndarray): (valid) data from the column, or a list of labels
        """
        if attr is None:
            return None
        all_data = self.data.get_column_view(attr)[0]
        if sp.issparse(all_data):
            all_data = all_data.toDense()  # TODO -- just guessing; fix this!
        elif all_data.dtype == object and attr.is_primitive():
            all_data = all_data.astype(float)
        if filter_valid and self.valid_data is not None:
            all_data = all_data[self.valid_data]
        if not merge_infrequent or attr.is_continuous \
                or len(attr.values) <= MAX:
            return attr.values if return_labels else all_data
        dist = bincount(all_data, max_val=len(attr.values) - 1)
        infrequent = np.zeros(len(attr.values), dtype=bool)
        infrequent[np.argsort(dist[0])[:-(MAX-1)]] = True
        # If discrete variable has more than maximium allowed values,
        # less used values are joined as "Other"
        if return_labels:
            return [value for value, infreq in zip(attr.values, infrequent)
                    if not infreq] + ["Other"]
        else:
            result = all_data.copy()
            freq_vals = [i for i, f in enumerate(infrequent) if not f]
            for i, f in enumerate(infrequent):
                result[all_data == i] = MAX - 1 if f else freq_vals.index(i)
            return result

    # Sizes
    def get_size_data(self):
        """Return the column corresponding to `attr_size`"""
        if self.attr_size == OWPlotGUI.SizeByOverlap:
            x, y = self.get_coordinates_data()
            coord_to_id = self.__get_overlap_groups(x, y)
            overlaps = [len(coord_to_id[xy]) for xy in zip(x, y)]
            return [1 + log2(o) for o in overlaps]
        return self.get_column(self.attr_size)

    def impute_sizes(self, size_data):
        """
        Default imputation for size data

        Missing values are replaced by `MinShapeSize - 2`. Imputation is
        done in place.

        Args:
            size_data (np.ndarray): scaled points sizes
        """
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = self.graph.MinShapeSize - 2
            self.Information.missing_size(self.attr_size)
        else:
            self.Information.missing_size.clear()

    def sizes_changed(self):
        self.graph.update_sizes()
        self.graph.update_colors()

    # Colors
    def get_color_data(self):
        """Return the column corresponding to color data"""
        colors = self.get_column(self.attr_color, merge_infrequent=True)
        if self.attr_size == OWPlotGUI.SizeByOverlap:
            # color overlapping points by most frequent color
            x, y = self.get_coordinates_data()
            coord_to_id = self.__get_overlap_groups(x, y)
            majority_colors = np.empty(len(x))
            for i, xy in enumerate(zip(x, y)):
                cnt = Counter(colors[j] for j in coord_to_id[xy])
                majority_colors[i] = cnt.most_common(1)[0][0]
            return majority_colors
        return colors

    def get_color_labels(self):
        """
        Return labels for the color legend

        Returns:
            (list of str): labels
        """
        return self.get_column(self.attr_color, merge_infrequent=True,
                               return_labels=True)


    def is_continuous_color(self):
        """
        Tells whether the color is continuous

        Returns:
            (bool):
        """
        return self.attr_color is not None and self.attr_color.is_continuous

    def set_palette(self):
        """
        Set the graph palette suitable for the current `attr_color`

        This method is invoked by the plot's `get_data` and must be overridden
        if the widget offers coloring that is not based on attribute values.
        """
        if self.attr_color is None:
            self.graph.palette = None
            return
        colors = self.attr_color.colors
        if self.attr_color.is_discrete:
            self.graph.palette = ColorPaletteGenerator(
                number_of_colors=min(len(colors), MAX),
                rgb_colors=colors if len(colors) <= MAX
                else DefaultRGBColors)
        else:
            self.graph.palette = ContinuousPaletteGenerator(*colors)

    def can_draw_density(self):
        """
        Tells whether the current data and settings are suitable for drawing
        densities

        Returns:
            (bool):
        """
        return self.data is not None and \
               self.data.domain is not None and \
               len(self.data) > 1 and \
               self.attr_color is not None

    def colors_changed(self):
        self.graph.update_colors()

    # Labels
    def get_label_data(self, formatter=None):
        """Return the column corresponding to label data"""
        if self.attr_label:
            label_data = self.get_column(self.attr_label)
            return map(formatter or self.attr_label.str_val, label_data)

    def labels_changed(self):
        self.graph.update_labels()

    # Shapes
    def get_shape_data(self):
        """
        Return labels for the shape legend

        Returns:
            (list of str): labels
        """
        return self.get_column(self.attr_shape, merge_infrequent=True)

    def get_shape_labels(self):
        return self.get_column(self.attr_shape, merge_infrequent=True,
                               return_labels=True)

    def impute_shapes(self, shape_data, default_symbol):
        """
        Default imputation for shape data

        Missing values are replaced by `default_symbol`. Imputation is
        done in place.

        Args:
            shape_data (np.ndarray): scaled points sizes
            default_symbol (str): a string representing the symbol
        """
        if shape_data is None:
            return 0
        nans = np.isnan(shape_data)
        if np.any(nans):
            shape_data[nans] = default_symbol
            self.Information.missing_shape(self.attr_shape)
        else:
            self.Information.missing_shape.clear()
        return shape_data

    def shapes_changed(self):
        self.graph.update_shapes()

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        def show_part(point_data, singular, plural, max_shown, vars):
            cols = [escape('{} = {}'.format(var.name, point_data[var]))
                    for var in vars[:max_shown + 2]
                    if vars == domain.class_vars
                    or var not in skip_attrs][:max_shown]
            if not cols:
                return ""
            n_vars = len(vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
                + "<br/>".join(cols)

        domain = self.data.domain
        parts = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        point_data = self.data[point_id]
        return "<br/>".join(show_part(point_data, *columns)
                            for columns in parts)

    def get_tooltip(self, point_ids):
        """
        Return the tooltip string for the given points

        The method is called by the plot on mouse hover

        Args:
            point_ids (list): indices into `data`

        Returns:
            (str):
        """
        text = "<hr/>".join(self._point_tooltip(point_id)
                            for point_id in point_ids[:MAX_POINTS_IN_TOOLTIP])
        if len(point_ids) > MAX_POINTS_IN_TOOLTIP:
            text = "{} instances<hr/>{}<hr/>...".format(len(point_ids), text)
        return text

    def keyPressEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    # Legend
    def combined_legend(self):
        """Tells whether the shape and color legends are combined into one"""
        return self.attr_shape == self.attr_color

    def sizeHint(self):
        return QSize(1132, 708)
