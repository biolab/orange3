import sys
import itertools
import warnings
from xml.sax.saxutils import escape
from math import log10, floor, ceil

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSize, QTimer
from AnyQt.QtGui import (
    QStaticText, QColor, QPen, QBrush, QPainterPath, QTransform, QPainter
)
from AnyQt.QtWidgets import (
    QApplication, QToolTip, QGraphicsTextItem, QGraphicsRectItem
)

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.LegendItem import (
    LegendItem as PgLegendItem, ItemSample
)
from pyqtgraph.graphicsItems.TextItem import TextItem

from Orange.util import OrangeDeprecationWarning
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import classdensity
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.plot import OWPalette
from Orange.widgets.visualize.owscatterplotgraph_obsolete import (
    OWScatterPlotGraph as OWScatterPlotGraphObs
)
from Orange.widgets.visualize.utils.plotutils import (
    HelpEventDelegate as EventDelegate,
    InteractiveViewBox as ViewBox
)


SELECTION_WIDTH = 5


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


class LegendItem(PgLegendItem):
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

    def restoreAnchor(self, anchors):
        """
        Restore (parent) relative position from stored anchors.

        The restored position is within the parent bounds.
        """
        anchor, parentanchor = anchors
        self.anchor(*bound_anchor_pos(anchor, parentanchor))

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

    COLOR_NOT_SUBSET = (128, 128, 128, 0)
    COLOR_SUBSET = (128, 128, 128, 255)
    COLOR_DEFAULT = (128, 128, 128, 0)

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

        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.labels = []

        self.master = scatter_widget
        self._create_drag_tooltip(self.plot_widget.scene())

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

        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.update_grid_visibility()

        self._tooltip_delegate = EventDelegate(self.help_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.view_box.sigTransformChanged.connect(self.update_density)

        self.timer = None

    def _create_legend(self, anchor):
        legend = LegendItem()
        legend.setParentItem(self.plot_widget.getViewBox())
        legend.restoreAnchor(anchor)
        return legend

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
        self.update_tooltip()

        scene.drag_tooltip = scene.createItemGroup([rect, text])
        scene.drag_tooltip.hide()

    def update_tooltip(self, modifiers=Qt.NoModifier):
        modifiers &= Qt.ShiftModifier + Qt.ControlModifier + Qt.AltModifier
        text = self.tiptexts.get(int(modifiers), self.tiptexts[0])
        self.tip_textitem.setHtml(text + self._get_jittering_tooltip())

    def _get_jittering_tooltip(self):
        warn_jittered = ""
        if self.jitter_size:
            warn_jittered = \
                '<br/><br/>' \
                '<span style="background-color: red; color: white; ' \
                'font-weight: 500;">' \
                '&nbsp;Warning: Selection is applied to unjittered data&nbsp;' \
                '</span>'
        return warn_jittered

    def update_jittering(self):
        self.update_tooltip()
        x, y = self.get_coordinates()
        if x is None or not len(x) or self.scatterplot_item is None:
            return
        self._update_plot_coordinates(self.scatterplot_item, x, y)
        self._update_plot_coordinates(self.scatterplot_item_sel, x, y)
        self._update_label_coords(x, y)

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

    def _update_plot_coordinates(self, plot, x, y):
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
        for prop in ('pen', 'brush', 'size', 'symbol', 'data',
                     'sourceRect', 'targetRect'):
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
            self._update_plot_coordinates(self.scatterplot_item, x, y)
            self._update_plot_coordinates(self.scatterplot_item_sel, x, y)

        self._update_label_coords(x, y)
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
        return self.MinShapeSize + (5 + self.point_width) * size_column

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
                # 0.5 - np.cos(np.arange(0.17, 1, 0.17) * np.pi) / 2
                factors = [0.07, 0.26, 0.52, 0.77, 0.95, 1]

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

            if np.sum(current_size_data) / self.n_valid != -1 and np.sum(diff):
                # If encountered any strange behaviour when updating sizes,
                # implement it with threads
                self.timer = QTimer(self.scatterplot_item, interval=50)
                self.timer.timeout.connect(Timeout())
                self.timer.start()
            else:
                self.scatterplot_item.setSize(size_data)
                self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)

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
        self.palette = self.master.get_palette()
        c_data = self.master.get_color_data()
        c_data = self._filter_visible(c_data)
        subset = self.master.get_subset_mask()
        subset = self._filter_visible(subset)
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
        pen = [_make_pen(color, 1.5) for _ in range(self.n_shown)]
        if subset is not None:
            brush = np.where(
                subset,
                *(QBrush(QColor(*col))
                  for col in (self.COLOR_SUBSET, self.COLOR_NOT_SUBSET)))
        else:
            color = QColor(*self.COLOR_DEFAULT)
            color.setAlpha(self.alpha_value)
            brush = [QBrush(color) for _ in range(self.n_shown)]
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
        update the colors. Finally, the method triggers the update of the
        legend and the density plot.
        """
        if self.scatterplot_item is not None:
            pen_data, brush_data = self.get_colors()
            self.scatterplot_item.setPen(pen_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
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
            rgb_data = [
                pen.color().getRgb()[:3] if pen is not None else (255, 255, 255)
                for pen in self.scatterplot_item.data['pen']]
            if len(set(rgb_data)) <= 1:
                return
            [min_x, max_x], [min_y, max_y] = self.view_box.viewRange()
            x_data, y_data = self.scatterplot_item.getData()
            self.density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution,
                x_data, y_data, rgb_data)
            self.plot_widget.addItem(self.density_img)

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
                    _make_pen(QColor(255, 190, 0, 255), SELECTION_WIDTH + 1),
                    nopen)
            else:
                palette = ColorPaletteGenerator(number_of_colors=sels + 1)
                pen = np.choose(
                    self._filter_visible(self.selection),
                    [nopen] + [_make_pen(palette[i], SELECTION_WIDTH + 1)
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
        Trigger an updaet of labels

        The method calls `get_labels` which in turn calls the widget's
        `get_label_data`. The obtained labels are shown if the corresponding
        points are selected or if `label_only_selected` is `false`.
        """
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []
        if self.scatterplot_item is None \
                or self.label_only_selected and self.selection is None:
            return
        labels = self.get_labels()
        if labels is None:
            return
        black = pg.mkColor(0, 0, 0)
        x, y = self.scatterplot_item.getData()
        if self.label_only_selected:
            selected = np.nonzero(self._filter_visible(self.selection))
            labels = labels[selected]
            x = x[selected]
            y = y[selected]
        for label, xp, yp in zip(labels, x, y):
            ti = TextItem(label, black)
            ti.setPos(xp, yp)
            self.plot_widget.addItem(ti)
            self.labels.append(ti)

    def _update_label_coords(self, x, y):
        """Update label coordinates"""
        if self.label_only_selected:
            selected = np.nonzero(self._filter_visible(self.selection))
            x = x[selected]
            y = y[selected]
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
        color_labels = None if cont_color else self.master.get_color_labels()
        if shape_labels == color_labels and shape_labels is not None:
            self._update_combined_legend(shape_labels)
        else:
            self._update_shape_legend(shape_labels)
            if cont_color:
                self._update_continuous_color_legend()
            else:
                self._update_color_legend(color_labels)
        self.update_legend_visibility()

    def _update_shape_legend(self, labels):
        self.shape_legend.clear()
        if labels is None or self.scatterplot_item is None:
            return
        color = QColor(0, 0, 0)
        color.setAlpha(self.alpha_value)
        for label, symbol in zip(labels, self.CurveSymbols):
            self.shape_legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, symbol=symbol),
                escape(label))

    def _update_continuous_color_legend(self):
        self.color_legend.clear()
        if self.scale is None or self.scatterplot_item is None:
            return
        label = PaletteItemSample(self.palette, self.scale)
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
        for i, (label, symbol) in enumerate(zip(labels, symbols)):
            color = QColor(*self.palette.getRGB(i))
            pen = _make_pen(color.darker(self.DarkerValue), 1.5)
            color.setAlpha(255 if self.subset_is_shown else self.alpha_value)
            brush = QBrush(color)
            legend.addItem(
                ScatterPlotItem(pen=pen, brush=brush, size=10, symbol=symbol),
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

    def select_by_click(self, _, points):
        if self.scatterplot_item is not None:
            self.select(points)

    def select_by_rectangle(self, value_rect):
        if self.scatterplot_item is not None:
            x0, y0 = value_rect.topLeft().x(), value_rect.topLeft().y()
            x1, y1 = value_rect.bottomRight().x(), value_rect.bottomRight().y()
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
        if keys & Qt.AltModifier:
            self.selection_remove(indices)
        elif keys & Qt.ShiftModifier and keys & Qt.ControlModifier:
            self.selection_append(indices)
        elif keys & Qt.ShiftModifier:
            self.selection_new_group(indices)
        else:
            self.selection_select(indices)

    def selection_select(self, indices):
        self.selection = np.zeros(self.n_valid, dtype=np.uint8)
        self.selection[indices] = 1
        self._update_after_selection()

    def selection_append(self, indices):
        self.selection[indices] = np.max(self.selection)
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


class HelpEventDelegate(EventDelegate):
    def __init__(self, delegate, parent=None):
        super().__init__(delegate, parent)
        warnings.warn("HelpEventDelegate class has been deprecated since 3.17."
                      " Use Orange.widgets.visualize.utils.plotutils."
                      "HelpEventDelegate instead.", OrangeDeprecationWarning)
