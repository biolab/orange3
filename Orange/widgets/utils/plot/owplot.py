'''

#################
Plot (``owplot``)
#################

.. autoclass:: OrangeWidgets.plot.OWPlot

'''

from PyQt4 import QtCore, QtGui

LeftLegend = 0
RightLegend = 1
BottomLegend = 2
TopLegend = 3
ExternalLegend = 4

UNUSED_ATTRIBUTES_STR = 'unused attributes'

from .owaxis import *
from .owcurve import *
from .owlegend import *
from .owplotgui import OWPlotGUI
from .owtools import *

from ..colorpalette import ColorPaletteGenerator

from PyQt4.QtGui import (
    QPen, QBrush, QColor,
    QGraphicsView,  QGraphicsScene, QPainter, QTransform, QPolygonF,
    QGraphicsRectItem)

from PyQt4.QtCore import QPointF, QPropertyAnimation, pyqtProperty, SIGNAL, Qt, QEvent

## Color values copied from orngView.SchemaView for consistency
SelectionPen = QPen(QBrush(QColor(51, 153, 255, 192)),
                    1, Qt.SolidLine, Qt.RoundCap)
SelectionBrush = QBrush(QColor(168, 202, 236, 192))

#from OWDlgs import OWChooseImageSizeDlg
#from OWColorPalette import *      # color palletes, ...
#from Orange.utils import deprecated_members, deprecated_attribute

from . import orangeqt

def n_min(*args):
    lst = args[0] if len(args) == 1 else args
    a = [i for i in lst if i is not None]
    return min(a) if a else None

def n_max(*args):
    lst = args[0] if len(args) == 1 else args
    a = [i for i in lst if i is not None]
    return max(a) if a else None

name_map = {
    "saveToFileDirect": "save_to_file_direct",
    "saveToFile" : "save_to_file",
    "addCurve" : "add_curve",
    "addMarker" : "add_marker",
    "updateLayout" : "update_layout",
    "activateZooming" : "activate_zooming",
    "activateSelection" : "activate_selection",
    "activateRectangleSelection" : "activate_rectangle_selection",
    "activatePolygonSelection" : "activate_polygon_selection",
    "activatePanning" : "activate_panning",
    "getSelectedPoints" : "get_selected_points",
    "setAxisScale" : "set_axis_scale",
    "setAxisLabels" : "set_axis_labels",
    "setAxisAutoScale" : "set_axis_autoscale",
    "setTickLength" : "set_axis_tick_length",
    "updateCurves" : "update_curves",
    "itemList" : "plot_items",
    "setShowMainTitle" : "set_show_main_title",
    "setMainTitle" : "set_main_title",
    "invTransform" : "inv_transform",
    "setAxisTitle" : "set_axis_title",
    "setShowAxisTitle" : "set_show_axis_title"
}

#@deprecated_members(name_map, wrap_methods=list(name_map.keys()))
class OWPlot(orangeqt.Plot):
    """
    The base class for all plots in Orange. It uses the Qt Graphics View Framework
    to draw elements on a graph.

    **Plot layout**

        .. attribute:: show_legend

            A boolean controlling whether the legend is displayed or not

        .. attribute:: show_main_title

            Controls whether or not the main plot title is displayed

        .. attribute:: main_title

            The plot title, usually show on top of the plot

        .. automethod:: set_main_title

        .. automethod:: set_show_main_title

        .. attribute:: axis_margin

            How much space (in pixels) should be left on each side for the axis, its label and its title.

        .. attribute:: title_margin

            How much space (in pixels) should be left at the top of the plot for the title, if the title is shown.

            .. seealso:: attribute :attr:`show_main_title`

        .. attribute:: plot_margin

            How much space (in pixels) should be left at each side of the plot as whitespace.


    **Coordinate transformation**

        There are several coordinate systems used by OWPlot:

        * `widget` coordinates.

          This is the coordinate system of the position returned by :meth:`.QEvent.pos()`.
          No calculations or positions is done with this coordinates, they must first be converted
          to scene coordinates with :meth:`mapToScene`.

        * `data` coordinates.

          The value used internally in Orange to specify the values of attributes.
          For example, this can be age in years, the number of legs, or any other numeric value.

        * `plot` coordinates.

          These coordinates specify where the plot items are placed on the graph, but doesn't account for zoom.
          They can be retrieved for a particular plot item with :meth:`.PlotItem.pos()`.

        * `scene` or `zoom` coordinates.

          Like plot coordinates, except that they take the :attr:`zoom_transform` into account. They represent the
          actual position of an item on the scene.

          These are the coordinates returned by :meth:`.PlotItem.scenePos()` and :meth:`mapToScene`.

          For example, they can be used to determine what is under the cursor.

        In most cases, you will use data coordinates for interacting with the actual data, and scene coordinates for
        interacting with the plot items. The other two sets are mostly used for converting.

        .. automethod:: map_to_graph

        .. automethod:: map_from_graph

        .. automethod:: transform

        .. automethod:: inv_transform

        .. method:: nearest_point(pos)

            Returns the point nearest to ``pos``, or ``None`` if no point is close enough.

            :param pos: The position in scene coordinates
            :type pos: QPointF

            :rtype: :obj:`.OWPoint`

        .. method:: point_at(pos)

            If there is a point with data coordinates equal to ``pos``, if is returned.
            Otherwise, this function returns None.

            :param pos: The position in data coordinates
            :type pos: tuple of float float

            :rtype: :obj:`.OWPoint`


    **Data curves**
        The preferred method for showing a series of data points is :meth:`set_main_curve_data`.
        It allows you to specify point positions, colors, labels, sizes and shapes.

        .. automethod:: set_main_curve_data

        .. automethod:: add_curve

        .. automethod:: add_custom_curve

        .. automethod:: add_marker

        .. method:: add_item(item)

            Adds any PlotItem ``item`` to this plot.
            Calling this function directly is useful for adding a :obj:`.Marker` or another object that does not have to appear in the legend.
            For data curves, consider using :meth:`add_custom_curve` instead.

        .. method:: plot_items()

            Returns the list of all plot items added to this graph with :meth:`add_item` or :meth:`.PlotItem.attach`.

    **Axes**

        .. automethod:: add_axis

        .. automethod:: add_custom_axis

        .. automethod:: set_axis_enabled

        .. automethod:: set_axis_labels

        .. automethod:: set_axis_scale

    **Settings**

	.. attribute:: gui

            An :obj:`.OWPlotGUI` object associated with this graph

    **Point Selection and Marking**

        There are four possible selection behaviors used for selecting or marking points in OWPlot.
        They are used in :meth:`select_points` and :meth:`mark_points` and are the same for both operations.

        .. data:: AddSelection

            The points are added to the selection, without affected the currently selected points

        .. data:: RemoveSelection

            The points are removed from the selection, without affected the currently selected points

        .. data:: ToggleSelection

            The points' selection state is toggled

        .. data:: ReplaceSelection

            The current selection is replaced with the new one

        .. note:: There are exacly the same functions for point selection and marking.
                For simplicity, they are only documented once.

        .. method:: select_points(area, behavior)
        .. method:: mark_points(area, behavior)

            Selects or marks all points inside the ``area``

            :param area: The newly selected/marked area
            :type area: QRectF or QPolygonF

            :param behavior: :data:`AddSelection`, :data:`RemoveSelection`, :data:`ToggleSelection` or :data:`ReplaceSelection`
            :type behavior: int

        .. method:: unselect_all_points()
        .. method:: unmark_all_points()

            Unselects or unmarks all the points in the plot

        .. method:: selected_points()
        .. method:: marked_points()

            Returns a list of all selected or marked points

            :rtype: list of OWPoint

        .. method:: selected_points(xData, yData)

            For each of the point specified by ``xData`` and ``yData``, the point's selection state is returned.

            :param xData: The list of x coordinates
            :type xData: list of float

            :param yData: The list of y coordinates
            :type yData: list of float

            :rtype: list of int

    **Color schemes**

        By default, OWPlot uses the application's system palette for drawing everything
        except data curves and points. This way, it maintains consistency with other application
        with regards to the user interface.

        If data is plotted with no color specified, it will use a system color as well,
        so that a good contrast with the background in guaranteed.

        OWPlot uses the :meth:`.OWidget.palette` to determine its color scheme, so it can be
        changed using :meth:`.QWidget.setPalette`. There are also two predefined color schemes:
        ``OWPalette.Dark`` and ``OWPalette.Light``, which provides a dark and a light scheme
        respectively.

        .. attribute:: theme_name

            A string attribute with three possible values:
            ==============  ===========================
            Value           Meaning
            --------------  ---------------------------
            "default"       The system palette is used
            "dark"          The dark theme is used
            "light"         The light theme is used
            ==============  ===========================

            To apply the settings, first set this attribute's value, and then call :meth:`update_theme`

        .. automethod:: update_theme

        On the other hand, curves with a specified color will use colors from Orange's palette,
        which can be configured within Orange. Each plot contains two separate palettes:
        one for continuous attributes, and one for discrete ones. Both are created by
        :obj:`.OWColorPalette.ColorPaletteGenerator`

        .. attribute:: continuous_palette

            The palette used when point color represents a continuous attribute

        .. attribute:: discrete_palette

            The palette used when point color represents a discrete attribute

    """

    point_settings = ["point_width", "alpha_value"]
    plot_settings = ["show_legend", "show_grid"]
    appearance_settings = ["antialias_plot", "animate_plot", "animate_points", "disable_animations_threshold", "auto_adjust_performance"]

    def settings_list(self, graph_name, settings):
        return [graph_name + '.' + setting for setting in settings]

    def __init__(self, parent = None,  name = "None",  show_legend = 1, axes = [xBottom, yLeft], widget = None ):
        """
            Creates a new graph

            If your visualization uses axes other than ``xBottom`` and ``yLeft``, specify them in the
            ``axes`` parameter. To use non-cartesian axes, set ``axes`` to an empty list
            and add custom axes with :meth:`add_axis` or :meth:`add_custom_axis`
        """
        orangeqt.Plot.__init__(self, parent)
        self.widget = widget
        self.parent_name = name
        self.show_legend = show_legend
        self.title_item = None

        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)

        self._legend = OWLegend(self, self.scene())
        self._legend.setZValue(LegendZValue)
        self._legend_margin = QRectF(0, 0, 100, 0)
        self._legend_moved = False
        self.axes = dict()

        self.axis_margin = 50
        self.y_axis_extra_margin = 30
        self.title_margin = 40
        self.graph_margin = 10

        self.mainTitle = None
        self.showMainTitle = False
        self.XaxisTitle = None
        self.YLaxisTitle = None
        self.YRaxisTitle = None

        # Method aliases, because there are some methods with different names but same functions
        self.setCanvasBackground = self.setCanvasColor
        self.map_from_widget = self.mapToScene

        # OWScatterPlot needs these:
        self.point_width = 5
        self.show_filled_symbols = True
        self.alpha_value = 255
        self.show_grid = True

        self.curveSymbols = list(range(13))
        self.tips = TooltipManager(self)
        self.setMouseTracking(True)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.PanGesture)

        self.state = NOTHING
        self._pressed_mouse_button = Qt.NoButton
        self._pressed_point = None
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.sendSelectionOnUpdate = False
        self.auto_send_selection_callback = None

        self.data_range = {}
        self.map_transform = QTransform()
        self.graph_area = QRectF()

        ## Performance optimization
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.scene().setItemIndexMethod(QGraphicsScene.NoIndex)

        self.animate_plot = True
        self.animate_points = True
        self.antialias_plot = True
        self.antialias_points = True
        self.antialias_lines = True

        self.auto_adjust_performance = True
        self.disable_animations_threshold = 5000
     #   self.setInteractive(False)

        self.warn_unused_attributes = False

        self._bounds_cache = {}
        self._transform_cache = {}
        self.block_update = False

        self.use_animations = True
        self._animations = []

        ## Mouse event handlers
        self.mousePressEventHandler = None
        self.mouseMoveEventHandler = None
        self.mouseReleaseEventHandler = None
        self.mouseStaticClickHandler = self.mouseStaticClick
        self.static_click = False

        self._marker_items = []
        self.grid_curve = PlotGrid(self)

        self._zoom_rect = None
        self._zoom_transform = QTransform()
        self.zoom_stack = []
        self.old_legend_margin = None
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        ## Add specified axes:

        for key in axes:
            if key in [yLeft, xTop]:
                self.add_axis(key, title_above=1)
            else:
                self.add_axis(key)

        self.continuous_palette = ColorPaletteGenerator(numberOfColors = -1)
        self.discrete_palette = ColorPaletteGenerator()

        self.gui = OWPlotGUI(self)
        """
            An :obj:`.OWPlotGUI` object associated with this plot
    	"""
        self.activate_zooming()
        self.selection_behavior = self.AddSelection

        self.main_curve = None

        self.replot()

#    selectionCurveList = deprecated_attribute("selectionCurveList", "selection_items")
#    autoSendSelectionCallback = deprecated_attribute("autoSendSelectionCallback", "auto_send_selection_callback")
#    showLegend = deprecated_attribute("showLegend", "show_legend")
#    pointWidth = deprecated_attribute("pointWidth", "point_width")
#    alphaValue = deprecated_attribute("alphaValue", "alpha_value")
#    useAntialiasing = deprecated_attribute("useAntialiasing", "use_antialiasing")
#    showFilledSymbols = deprecated_attribute("showFilledSymbols", "show_filled_symbols")
#    mainTitle = deprecated_attribute("mainTitle", "main_title")
#    showMainTitle = deprecated_attribute("showMainTitle", "show_main_title")
#    gridCurve = deprecated_attribute("gridCurve", "grid_curve")
#    contPalette = deprecated_attribute("contPalette", "continuous_palette")
#    discPalette = deprecated_attribute("discPalette", "discrete_palette")

    def scrollContentsBy(self, dx, dy):
        # This is overriden here to prevent scrolling with mouse and keyboard
        # Instead of moving the contents, we simply do nothing
        pass

    def graph_area_rect(self):
        return self.graph_area

    def map_to_graph(self, point, axes = None, zoom = False):
        '''
            Maps ``point``, which can be ether a tuple of (x,y), a QPoint or a QPointF, from data coordinates
            to plot coordinates.

            :param point: The point in data coordinates
            :type point: tuple or QPointF

            :param axes: The pair of axes along which to transform the point.
                         If none are specified, (xBottom, yLeft) will be used.
            :type axes: tuple of float float

            :param zoom: if ``True``, the current :attr:`zoom_transform` will be considered in the transformation, and the result will be in scene coordinates instead.
            :type zoom: int

            :return: The transformed point in scene coordinates
            :type: tuple of float float
        '''
        if type(point) == tuple:
            (x, y) = point
            point = QPointF(x, y)
        if axes:
            x_id, y_id = axes
            point = point * self.transform_for_axes(x_id, y_id)
        else:
            point = point * self.map_transform
        if zoom:
            point = point * self._zoom_transform
        return (point.x(), point.y())

    def map_from_graph(self, point, axes = None, zoom = False):
        '''
            Maps ``point``, which can be ether a tuple of (x,y), a QPoint or a QPointF, from plot coordinates
            to data coordinates.

            :param point: The point in data coordinates
            :type point: tuple or QPointF

            :param axes: The pair of axes along which to transform the point. If none are specified, (xBottom, yLeft) will be used.
            :type axes: tuple of float float

            :param zoom: if ``True``, the current :attr:`zoom_transform` will be considered in the transformation, and the ``point`` should be in scene coordinates instead.
            :type zoom: int

            :returns: The transformed point in data coordinates
            :rtype: tuple of float float
        '''
        if type(point) == tuple:
            (x, y) = point
            point = QPointF(x,y)
        if zoom:
            t, ok = self._zoom_transform.inverted()
            point = point * t
        if axes:
            x_id, y_id = axes
            t, ok = self.transform_for_axes(x_id, y_id).inverted()
        else:
            t, ok = self.map_transform.inverted()
        ret = point * t
        return (ret.x(), ret.y())

    def save_to_file(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons, parent=self)
        sizeDlg.exec_()

    def save_to_file_direct(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)

    def activate_zooming(self):
        '''
            Activates the zooming mode, where the user can zoom in and out with a single mouse click
            or by dragging the mouse to form a rectangular area
        '''
        self.state = ZOOMING

    def activate_rectangle_selection(self):
        '''
            Activates the rectangle selection mode, where the user can select points in a rectangular area
            by dragging the mouse over them
        '''
        self.state = SELECT_RECTANGLE

    def activate_selection(self):
        '''
            Activates the point selection mode, where the user can select points by clicking on them
        '''
        self.state = SELECT

    def activate_polygon_selection(self):
        '''
            Activates the polygon selection mode, where the user can select points by drawing a polygon around them
        '''
        self.state = SELECT_POLYGON

    def activate_panning(self):
        '''
            Activates the panning mode, where the user can move the zoom projection by dragging the mouse
        '''
        self.state = PANNING

    def set_show_main_title(self, b):
        '''
            Shows the main title if ``b`` is ``True``, and hides it otherwise.
        '''
        self.showMainTitle = b
        self.replot()

    def set_main_title(self, t):
        '''
            Sets the main title to ``t``
        '''
        self.mainTitle = t
        self.replot()

    def setShowXaxisTitle(self, b = -1):
        if b == -1 and hasattr(self, 'showXaxisTitle'):
            b = self.showXaxisTitle
        self.setShowAxisTitle(xBottom, b)

    def setXaxisTitle(self, title):
        self.setAxisTitle(xBottom, title)

    def setShowYLaxisTitle(self, b = -1):
        if b == -1 and hasattr(self, 'showYLaxisTitle'):
            b = self.showYLaxisTitle
        self.setShowAxisTitle(yLeft, b)

    def setYLaxisTitle(self, title):
        self.setAxisTitle(yLeft, title)

    def setShowYRaxisTitle(self, b = -1):
        if b == -1 and hasattr(self, 'showYRaxisTitle'):
            b = self.showYRaxisTitle
        self.setShowAxisTitle(yRight, b)

    def setYRaxisTitle(self, title):
        self.setAxisTitle(yRight, title)

    def enableGridXB(self, b):
        self.grid_curve.set_x_enabled(b)
        self.replot()

    def enableGridYL(self, b):
        self.grid_curve.set_y_enabled(b)
        self.replot()

    def setGridColor(self, c):
        self.grid_curve.set_pen(QPen(c))
        self.replot()

    def setCanvasColor(self, c):
        p = self.palette()
        p.setColor(OWPalette.Canvas, c)
        self.set_palette(p)

    def setData(self, data):
        self.clear()
        self.replot()

    def setXlabels(self, labels):
        if xBottom in self.axes:
            self.set_axis_labels(xBottom, labels)
        elif xTop in self.axes:
            self.set_axis_labels(xTop, labels)

    def set_axis_autoscale(self, axis_id):
        if axis_id in self.axes:
            self.axes[axis_id].auto_scale = True
        elif axis_id in self.data_range:
            del self.data_range[axis_id]

    def set_axis_labels(self, axis_id, labels):
        '''
            Sets the labels of axis ``axis_id`` to ``labels``. This is used for axes displaying a discrete data type.

            :param labels: The ID of the axis to change
            :type labels: int

            :param labels: The list of labels to be displayed along the axis
            :type labels: A list of strings

            .. note:: This changes the axis scale and removes any previous scale set with :meth:`set_axis_scale`.
        '''
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        self.axes[axis_id].set_labels(labels)

    def set_axis_scale(self, axis_id, min, max, step_size=0):
        '''
            Sets the scale of axis ``axis_id`` to show an interval between ``min`` and ``max``.
            If ``step`` is specified and non-zero, it determines the steps between label on the axis.
            Otherwise, they are calculated automatically.

            .. note:: This changes the axis scale and removes any previous labels set with :meth:`set_axis_labels`.
        '''
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        if axis_id in self.axes:
            self.axes[axis_id].set_scale(min, max, step_size)
        else:
            self.data_range[axis_id] = (min, max)

    def set_axis_title(self, axis_id, title):
        if axis_id in self.axes:
            self.axes[axis_id].set_title(title)

    def set_show_axis_title(self, axis_id, b):
        if axis_id in self.axes:
            if b == -1:
                b = not self.axes[axis_id].show_title
            self.axes[axis_id].set_show_title(b)
            self.replot()

    def set_axis_tick_length(self, axis_id, minor, medium, major):
        if axis_id in self.axes:
            self.axes[axis_id].set_tick_legth(minor, medium, major)

    def setYLlabels(self, labels):
        self.set_axis_labels(yLeft, labels)

    def setYRlabels(self, labels):
        self.set_axis_labels(yRight, labels)

    def add_custom_curve(self, curve, enableLegend = False):
        '''
            Adds a custom PlotItem ``curve`` to the plot.
            If ``enableLegend`` is ``True``, a curve symbol defined by
            :meth:`.OWCurve.point_item` and the ``curve``'s name
            :obj:`.OWCurve.name` is added to the legend.

            This function recalculates axis bounds and replots the plot if needed.

            :param curve: The curve to add
            :type curve: :obj:`.OWCurve`
        '''
        self.add_item(curve)
        if enableLegend:
            self.legend().add_curve(curve)
        for key in [curve.axes()]:
            if key in self._bounds_cache:
                del self._bounds_cache[key]
        self._transform_cache = {}
        if hasattr(curve, 'tooltip'):
            curve.setToolTip(curve.tooltip)
        x,y = curve.axes()
        if curve.is_auto_scale() and (self.is_axis_auto_scale(x) or self.is_axis_auto_scale(y)):
            self.set_dirty()
            self.replot()
        else:
            curve.set_graph_transform(self.transform_for_axes(x,y))
            curve.update_properties()
        return curve

    def add_curve(self, name, brushColor = None, penColor = None, size = 5, style = Qt.NoPen,
                 symbol = OWPoint.Ellipse, enableLegend = False, xData = [], yData = [], showFilledSymbols = None,
                 lineWidth = 1, pen = None, autoScale = 0, antiAlias = None, penAlpha = 255, brushAlpha = 255,
                 x_axis_key = xBottom, y_axis_key = yLeft):
        '''
            Creates a new :obj:`.OWCurve` with the specified parameters and adds it to the graph.
            If ``enableLegend`` is ``True``, a curve symbol is added to the legend.
        '''
        c = OWCurve(xData, yData, x_axis_key, y_axis_key, tooltip=name)
        c.set_zoom_transform(self._zoom_transform)
        c.name = name
        c.set_style(style)

        if not brushColor:
            brushColor = self.color(OWPalette.Data)
        if not penColor:
            penColor = self.color(OWPalette.Data)

        c.set_color(penColor)

        if pen:
            p = pen
        else:
            p = QPen()
            p.setColor(penColor)
            p.setWidth(lineWidth)
        c.set_pen(p)

        c.set_brush(brushColor)

        c.set_symbol(symbol)
        c.set_point_size(size)
        c.set_data(xData,  yData)

        c.set_auto_scale(autoScale)

        return self.add_custom_curve(c, enableLegend)

    def set_main_curve_data(self, x_data, y_data, color_data, label_data, size_data, shape_data, marked_data = [], valid_data = [], x_axis_key=xBottom, y_axis_key=yLeft):
        """
            Creates a single curve that can have points of different colors, shapes and sizes.
            This is the preferred method for visualization that show a series of different points.

            :param x_data: The list of X coordinates of the points
            :type x_data: list of float

            :param y_data: The list of Y coordinates of the points
            :type y_data: list of float

            :param color_data: The list of point colors
            :type color_data: list of QColor

            :param label_data: The list of point labels
            :type label_data: list of str

            :param size_data: The list of point sizes
            :type size_data: list of int

            :param shape_data: The list of point symbols
            :type shape_data: list of int

            The number of points in the curve will be equal to min(len(x_data), len(y_data)).
            The other four list can be empty, in which case a default value will be used.
            If they contain only one element, its value will be used for all points.

            .. note:: This function does not add items to the legend automatically.
                      You will have to add them yourself with :meth:`.OWLegend.add_item`.

            .. seealso:: :obj:`.OWMultiCurve`, :obj:`.OWPoint`
        """
        if not self.main_curve:
            self.main_curve = OWMultiCurve([], [])
            self.add_item(self.main_curve)

        self.update_performance(len(x_data))

        if len(valid_data):
            import numpy
            x_data = numpy.compress(valid_data, x_data)
            y_data = numpy.compress(valid_data, y_data)
            if len(color_data) > 1:
                color_data = numpy.compress(valid_data, color_data)
            if len(size_data) > 1:
                size_data = numpy.compress(valid_data, size_data)
            if len(shape_data) > 1:
                shape_data = numpy.compress(valid_data, shape_data)
            if len(label_data) > 1:
                label_data = numpy.compress(valid_data, label_data)
            if len(marked_data) > 1:
                marked_data = numpy.compress(valid_data, marked_data).tolist()

        c = self.main_curve
        c.set_data(x_data, y_data)
        c.set_axes(x_axis_key, y_axis_key)
        c.set_point_colors(color_data)
        c.set_point_labels(label_data)
        c.set_point_sizes(size_data)
        c.set_point_symbols(shape_data)
        if len(marked_data):
            c.set_points_marked(marked_data)
            self.emit(SIGNAL('marked_points_changed()'))
        c.name = 'Main Curve'

        self.replot()

    def remove_curve(self, item):
        '''
            Removes ``item`` from the plot
        '''
        self.remove_item(item)
        self.legend().remove_curve(item)

    def plot_data(self, xData, yData, colors, labels, shapes, sizes):
        pass

    def add_axis(self, axis_id, title = '', title_above = False, title_location = AxisMiddle, line = None, arrows = 0, zoomable = False):
        '''
            Creates an :obj:`OrangeWidgets.plot.OWAxis` with the specified ``axis_id`` and ``title``.
        '''
        a = OWAxis(axis_id, title, title_above, title_location, line, arrows, self)
        self.scene().addItem(a)
        a.zoomable = zoomable
        a.update_callback = self.replot
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        self.axes[axis_id] = a
        if not axis_id in CartesianAxes:
            self.setShowAxisTitle(axis_id, True)
        return a

    def remove_all_axes(self, user_only = True):
        '''
            Removes all axes from the plot
        '''
        ids = []
        for id,item in self.axes.items():
            if not user_only or id >= UserAxis:
                ids.append(id)
                self.scene().removeItem(item)
        for id in ids:
            del self.axes[id]

    def add_custom_axis(self, axis_id, axis):
        '''
            Adds a custom ``axis`` with id ``axis_id`` to the plot
        '''
        self.axes[axis_id] = axis
        self.replot()

    def add_marker(self, name, x, y, alignment = -1, bold = 0, color = None, brushColor = None, size=None, antiAlias = None,
                    x_axis_key = xBottom, y_axis_key = yLeft):
        m = Marker(name, x, y, alignment, bold, color, brushColor)
        self._marker_items.append((m, x, y, x_axis_key, y_axis_key))
        self.add_custom_curve(m)

        return m

    def removeAllSelections(self):
        ## TODO
        pass

    def clear(self):
        '''
            Clears the plot, removing all curves, markers and tooltips.
            Axes and the grid are not removed
        '''
        for i in self.plot_items():
            if i is not self.grid_curve:
                self.remove_item(i)
        self.main_curve = None
        self._bounds_cache = {}
        self._transform_cache = {}
        self.clear_markers()
        self.tips.removeAll()
        self.legend().clear()
        self.old_legend_margin = None
        self.update_grid()

    def clear_markers(self):
        '''
            Removes all markers added with :meth:`add_marker` from the plot
        '''
        for item,x,y,x_axis,y_axis in self._marker_items:
            item.detach()
        self._marker_items = []

    def update_layout(self):
        '''
            Updates the plot layout.

            This function recalculates the position of titles, axes, the legend and the main plot area.
            It does not update the curve or the other plot items.
        '''
        if not self.isVisible():
            # No point in updating the graph if it's still hidden
            return
        graph_rect = QRectF(self.contentsRect())
        self.centerOn(graph_rect.center())
        m = self.graph_margin
        graph_rect.adjust(m, m, -m, -m)

        if self.showMainTitle and self.mainTitle:
            if self.title_item:
                self.scene().remove_item(self.title_item)
                del self.title_item
            self.title_item = QGraphicsTextItem(self.mainTitle, scene=self.scene())
            title_size = self.title_item.boundingRect().size()
            ## TODO: Check if the title is too big
            self.title_item.setPos( graph_rect.width()/2 - title_size.width()/2, self.title_margin/2 - title_size.height()/2 )
            graph_rect.setTop(graph_rect.top() + self.title_margin)

        if self.show_legend:
            self._legend_outside_area = QRectF(graph_rect)
            self._legend.max_size = self._legend_outside_area.size()
            r = self._legend_margin
            graph_rect.adjust(r.left(), r.top(), -r.right(), -r.bottom())

        self._legend.update_items()

        axis_rects = dict()
        base_margin = min(self.axis_margin,  graph_rect.height()/4, graph_rect.height()/4)
        if xBottom in self.axes and self.axes[xBottom].isVisible():
            margin = base_margin
            if self.axes[xBottom].should_be_expanded():
                margin += min(20, graph_rect.height()/8, graph_rect.width() / 8)
            bottom_rect = QRectF(graph_rect)
            bottom_rect.setTop( bottom_rect.bottom() - margin)
            axis_rects[xBottom] = bottom_rect
            graph_rect.setBottom( graph_rect.bottom() - margin)
        if xTop in self.axes and self.axes[xTop].isVisible():
            margin = base_margin
            if self.axes[xTop].should_be_expanded():
                margin += min(20, graph_rect.height()/8, graph_rect.width() / 8)
            top_rect = QRectF(graph_rect)
            top_rect.setBottom(top_rect.top() + margin)
            axis_rects[xTop] = top_rect
            graph_rect.setTop(graph_rect.top() + margin)
        if yLeft in self.axes and self.axes[yLeft].isVisible():
            margin = base_margin
            if self.axes[yLeft].should_be_expanded():
                margin += min(20, graph_rect.height()/8, graph_rect.width() / 8)
            left_rect = QRectF(graph_rect)
            left = graph_rect.left() + margin + self.y_axis_extra_margin
            left_rect.setRight(left)
            graph_rect.setLeft(left)
            axis_rects[yLeft] = left_rect
            if xBottom in axis_rects:
                axis_rects[xBottom].setLeft(left)
            if xTop in axis_rects:
                axis_rects[xTop].setLeft(left)
        if yRight in self.axes and self.axes[yRight].isVisible():
            margin = base_margin
            if self.axes[yRight].should_be_expanded():
                margin += min(20, graph_rect.height()/8, graph_rect.width() / 8)
            right_rect = QRectF(graph_rect)
            right = graph_rect.right() - margin - self.y_axis_extra_margin
            right_rect.setLeft(right)
            graph_rect.setRight(right)
            axis_rects[yRight] = right_rect
            if xBottom in axis_rects:
                axis_rects[xBottom].setRight(right)
            if xTop in axis_rects:
                axis_rects[xTop].setRight(right)

        if self.graph_area != graph_rect:
            self.graph_area = QRectF(graph_rect)
            self.set_graph_rect(self.graph_area)
            self._transform_cache = {}

            if self._zoom_rect:
                data_zoom_rect = self.map_transform.inverted()[0].mapRect(self._zoom_rect)
                self.map_transform = self.transform_for_axes()
                self.set_zoom_rect(self.map_transform.mapRect(data_zoom_rect))

        self.map_transform = self.transform_for_axes()

        for c in self.plot_items():
            x,y = c.axes()
            c.set_graph_transform(self.transform_for_axes(x,y))
            c.update_properties()

    def update_zoom(self):
        '''
            Updates the zoom transformation of the plot items.
        '''
        zt = self.zoom_transform()
        self._zoom_transform = zt
        self.set_zoom_transform(zt)

        self.update_axes(zoom_only=True)
        self.viewport().update()

    def update_axes(self, zoom_only=False):
        """
            Updates the axes.

            If ``zoom_only`` is ``True``, only the positions of the axes and their labels are recalculated.
            Otherwise, all their labels are updated.
        """
        if self.warn_unused_attributes and not zoom_only:
            self._legend.remove_category(UNUSED_ATTRIBUTES_STR)

        for id, item in self.axes.items():
            if item.scale is None and item.labels is None:
                item.auto_range = self.bounds_for_axis(id)

            if id in XAxes:
                (x,y) = (id, yLeft)
            elif id in YAxes:
                (x,y) = (xBottom, id)
            else:
                (x,y) = (xBottom, yLeft)

            if id in CartesianAxes:
                ## This class only sets the lines for these four axes, widgets are responsible for the rest
                if x in self.axes and y in self.axes:
                    item.data_line = self.axis_line(self.data_rect_for_axes(x,y), id)
            if id in CartesianAxes:
                item.graph_line = self.axis_line(self.graph_area, id, invert_y = True)
            elif item.data_line:
                t = self.transform_for_axes(x, y)
                item.graph_line = t.map(item.data_line)

            if item.graph_line and item.zoomable:
                item.graph_line = self._zoom_transform.map(item.graph_line)

            if not zoom_only:
                if item.graph_line:
                    item.show()
                else:
                    item.hide()
                    if self.warn_unused_attributes:
                        self._legend.add_item(UNUSED_ATTRIBUTES_STR, item.title, None)
            item.zoom_transform = self._zoom_transform
            item.update(zoom_only)

    def replot(self):
        '''
            Replot the entire graph.

            This functions redraws everything on the graph, so it can be very slow
        '''
        #self.setBackgroundBrush(self.color(OWPalette.Canvas))
        self._bounds_cache = {}
        self._transform_cache = {}
        self.set_clean()
        self.update_antialiasing()
        self.update_legend()
        self.update_layout()
        self.update_zoom()
        self.update_axes()
        self.update_grid()
        self.update_filled_symbols()
        self.setSceneRect(QRectF(self.contentsRect()))
        self.viewport().update()

    def update_legend(self):
        if self.show_legend and not self._legend_moved:
            ## If the legend hasn't been moved it, we set it outside, in the top right corner
            m = self.graph_margin
            r = QRectF(self.contentsRect())
            r.adjust(m, m, -m, -m)
            self._legend.max_size = r.size()
            self._legend.update_items()
            w = self._legend.boundingRect().width()
            self._legend_margin = QRectF(0, 0, w, 0)
            self._legend.set_floating(False)
            self._legend.set_orientation(Qt.Vertical)
            self._legend.setPos(QRectF(self.contentsRect()).topRight() + QPointF(-w, 0))


        if (self._legend.isVisible() == self.show_legend):
            return

        self._legend.setVisible(self.show_legend)
        if self.show_legend:
            if self.old_legend_margin is not None:
                self.animate(self, 'legend_margin', self.old_legend_margin, duration = 100)
            else:
                r = self.legend_rect()
                self.ensure_inside(r, self.contentsRect())
                self._legend.setPos(r.topLeft())
                self.notify_legend_moved(r.topLeft())
        else:
            self.old_legend_margin = self.legend_margin
            self.animate(self, 'legend_margin', QRectF(), duration=100)

    def update_filled_symbols(self):
        ## TODO: Implement this in Curve.cpp
        pass

    def update_grid(self):
        self.grid_curve.set_x_enabled(self.show_grid)
        self.grid_curve.set_y_enabled(self.show_grid)
        self.grid_curve.update_properties()

    def legend(self):
        '''
            Returns the plot's legend, which is a :obj:`OrangeWidgets.plot.OWLegend`
        '''
        return self._legend

    def legend_rect(self):
        if self.show_legend:
            return self._legend.mapRectToScene(self._legend.boundingRect())
        else:
            return QRectF()

    def isLegendEvent(self, event, function):
        if self.show_legend and self.legend_rect().contains(self.mapToScene(event.pos())):
            function(self, event)
            return True
        else:
            return False

    def mouse_action(self, event):
        b = event.buttons() | event.button()
        m = event.modifiers()
        if b == Qt.LeftButton | Qt.RightButton:
            b = Qt.MidButton
        if m & Qt.AltModifier and b == Qt.LeftButton:
            m = m & ~Qt.AltModifier
            b = Qt.MidButton

        if b == Qt.LeftButton and not m:
            return self.state

        if b == Qt.RightButton and not m and self.state == SELECT:
            return SELECT_RIGHTCLICK

        if b == Qt.MidButton:
            return PANNING

        if b in [Qt.LeftButton, Qt.RightButton] and (self.state == ZOOMING or m == Qt.ControlModifier):
            return ZOOMING

        if b == Qt.LeftButton and m == Qt.ShiftModifier:
            return SELECT

    ## Event handling

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        else:
            return orangeqt.Plot.event(self, event)

    def gestureEvent(self, event):
        for gesture in event.gestures():
            if gesture.state() == Qt.GestureStarted:
                self.current_gesture_scale = 1.
                event.accept(gesture)
                continue
            elif gesture.gestureType() == Qt.PinchGesture:
                old_animate_plot = self.animate_plot
                self.animate_plot = False
                self.zoom(gesture.centerPoint(), gesture.scaleFactor()/self.current_gesture_scale )
                self.current_gesture_scale = gesture.scaleFactor()
                self.animate_plot = old_animate_plot
            elif gesture.gestureType() == Qt.PanGesture:
                self.pan(gesture.delta())
        return True

    def resizeEvent(self, event):
        self.replot()
        s = event.size() - event.oldSize()
        if self.legend_margin.right() > 0:
            self._legend.setPos(self._legend.pos() + QPointF(s.width(), 0))
        if self.legend_margin.bottom() > 0:
            self._legend.setPos(self._legend.pos() + QPointF(0, s.height()))

    def showEvent(self, event):
        self.replot()

    def mousePressEvent(self, event):
        self.static_click = True
        self._pressed_mouse_button = event.button()
        self._pressed_mouse_pos = event.pos()

        if self.mousePressEventHandler and self.mousePressEventHandler(event):
            event.accept()
            return

        if self.isLegendEvent(event, QGraphicsView.mousePressEvent):
            return

        point = self.mapToScene(event.pos())
        a = self.mouse_action(event)

        if a == SELECT and hasattr(self, 'move_selected_points'):
            self._pressed_point = self.nearest_point(point)
            self._pressed_point_coor = None
            if self._pressed_point is not None:
                self._pressed_point_coor = self._pressed_point.coordinates()

        if a == PANNING:
            self._last_pan_pos = point
            event.accept()
        else:
            orangeqt.Plot.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if event.buttons() and (self._pressed_mouse_pos - event.pos()).manhattanLength() > qApp.startDragDistance():
            self.static_click = False

        if self.mouseMoveEventHandler and self.mouseMoveEventHandler(event):
            event.accept()
            return

        if self.isLegendEvent(event, QGraphicsView.mouseMoveEvent):
            return

        point = self.mapToScene(event.pos())
        if not self._pressed_mouse_button:
            if self.receivers(SIGNAL('point_hovered(Point*)')) > 0:
                self.emit(SIGNAL('point_hovered(Point*)'), self.nearest_point(point))

        ## We implement a workaround here, because sometimes mouseMoveEvents are not fast enough
        ## so the moving legend gets left behind while dragging, and it's left in a pressed state
        if self._legend.mouse_down:
            QGraphicsView.mouseMoveEvent(self, event)
            return

        a = self.mouse_action(event)

        if a == SELECT and self._pressed_point is not None and self._pressed_point.is_selected() and hasattr(self, 'move_selected_points'):
            animate_points = self.animate_points
            self.animate_points = False
            x1, y1 = self._pressed_point_coor
            x2, y2 = self.map_from_graph(point, zoom=True)
            self.move_selected_points((x2 - x1, y2 - y1))
            self.replot()
            if self._pressed_point is not None:
                self._pressed_point_coor = self._pressed_point.coordinates()

            self.animate_points = animate_points

        elif a in [SELECT, ZOOMING] and self.graph_area.contains(point):
            if not self._current_rs_item:
                self._selection_start_point = self.mapToScene(self._pressed_mouse_pos)
                self._current_rs_item = QGraphicsRectItem(scene=self.scene())
                self._current_rs_item.setPen(SelectionPen)
                self._current_rs_item.setBrush(SelectionBrush)
                self._current_rs_item.setZValue(SelectionZValue)
            self._current_rs_item.setRect(QRectF(self._selection_start_point, point).normalized())
        elif a == PANNING:
            if not self._last_pan_pos:
                self._last_pan_pos = self.mapToScene(self._pressed_mouse_pos)
            self.pan(point - self._last_pan_pos)
            self._last_pan_pos = point
        else:
            x, y = self.map_from_graph(point, zoom=True)
            text, x, y = self.tips.maybeTip(x, y)
            if type(text) == int:
                text = self.buildTooltip(text)
            if text and x is not None and y is not None:
                tp = self.mapFromScene(QPointF(x,y) * self.map_transform * self._zoom_transform)
                self.showTip(tp.x(), tp.y(), text)
            else:
                orangeqt.Plot.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        self._pressed_mouse_button = Qt.NoButton

        if self.mouseReleaseEventHandler and self.mouseReleaseEventHandler(event):
            event.accept()
            return
        if self.static_click and self.mouseStaticClickHandler and self.mouseStaticClickHandler(event):
            event.accept()
            return

        if self.isLegendEvent(event, QGraphicsView.mouseReleaseEvent):
            return

        a = self.mouse_action(event)
        if a == SELECT and self._pressed_point is not None:
            self._pressed_point = None
        if a in [ZOOMING, SELECT] and self._current_rs_item:
            rect = self._current_rs_item.rect()
            if a == ZOOMING:
                self.zoom_to_rect(self._zoom_transform.inverted()[0].mapRect(rect))
            else:
                self.add_selection(rect)
            self.scene().removeItem(self._current_rs_item)
            self._current_rs_item = None
            return
        orangeqt.Plot.mouseReleaseEvent(self, event)

    def mouseStaticClick(self, event):
        point = self.mapToScene(event.pos())
        if point not in self.graph_area:
            return False

        a = self.mouse_action(event)
        b = event.buttons() | event.button()

        if a == ZOOMING:
            if event.button() == Qt.LeftButton:
                self.zoom_in(point)
            elif event.button() == Qt.RightButton:
                self.zoom_back()
            else:
                return False
            return True
        elif a == SELECT and b == Qt.LeftButton:
            point_item = self.nearest_point(point)
            b = self.selection_behavior

            if b == self.ReplaceSelection:
                self.unselect_all_points()
                b = self.AddSelection

            if point_item:
                point_item.set_selected(b == self.AddSelection or (b == self.ToggleSelection and not point_item.is_selected()))
            self.emit(SIGNAL('selection_changed()'))
        elif a == SELECT and b == Qt.RightButton:
            point_item = self.nearest_point(point)
            if point_item:
                self.emit(SIGNAL('point_rightclicked(Point*)'), self.nearest_point(point))
            else:
                self.unselect_all_points()
        else:
            return False

    def wheelEvent(self, event):
        point = self.mapToScene(event.pos())
        d = event.delta() / 120.0
        self.zoom(point, pow(2,d))

    @staticmethod
    def transform_from_rects(r1, r2):
        """
            Returns a QTransform that maps from rectangle ``r1`` to ``r2``.
        """
        if r1 is None or r2 is None:
            return QTransform()
        if r1.width() == 0 or r1.height() == 0 or r2.width() == 0 or r2.height() == 0:
            return QTransform()
        tr1 = QTransform().translate(-r1.left(), -r1.top())
        ts = QTransform().scale(r2.width()/r1.width(), r2.height()/r1.height())
        tr2 = QTransform().translate(r2.left(), r2.top())
        return tr1 * ts * tr2

    def transform_for_zoom(self, factor, point, rect):
        if factor == 1:
            return QTransform()

        dp = point

        t = QTransform()
        t.translate(dp.x(), dp.y())
        t.scale(factor, factor)
        t.translate(-dp.x(), -dp.y())
        return t

    def rect_for_zoom(self, point, old_rect, scale = 2):
        r = QRectF()
        r.setWidth(old_rect.width() / scale)
        r.setHeight(old_rect.height() / scale)
        r.moveCenter(point)

        self.ensure_inside(r, self.graph_area)

        return r

    def set_state(self, state):
        self.state = state
        if state != SELECT_RECTANGLE:
            self._current_rs_item = None
        if state != SELECT_POLYGON:
            self._current_ps_item = None

    def get_selected_points(self, xData, yData, validData):
        if self.main_curve:
            selected = []
            points = self.main_curve.points()
            i = 0
            for d in validData:
                if d:
                    selected.append(points[i].is_selected())
                    i += 1
                else:
                    selected.append(False)
        else:
            selected = self.selected_points(xData, yData)
        unselected = [not i for i in selected]
        return selected, unselected

    def add_selection(self, reg):
        """
            Selects all points in the region ``reg`` using the current :attr: `selection_behavior`.
        """
        self.select_points(reg, self.selection_behavior)
        self.viewport().update()
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def points_equal(self, p1, p2):
        if type(p1) == tuple:
            (x, y) = p1
            p1 = QPointF(x, y)
        if type(p2) == tuple:
            (x, y) = p2
            p2 = QPointF(x, y)
        return (QPointF(p1)-QPointF(p2)).manhattanLength() < self.polygon_close_treshold

    def data_rect_for_axes(self, x_axis = xBottom, y_axis = yLeft):
        """
            Calculates the bounding rectangle in data coordinates for the axes ``x_axis`` and ``y_axis``.
        """
        if x_axis in self.axes and y_axis in self.axes:
            x_min, x_max = self.bounds_for_axis(x_axis, try_auto_scale=True)
            y_min, y_max = self.bounds_for_axis(y_axis, try_auto_scale=True)
            if (x_min or x_max) and (y_min or y_max):
                r = QRectF(x_min, y_min, x_max-x_min, y_max-y_min)
                return r
        r = orangeqt.Plot.data_rect_for_axes(self, x_axis, y_axis)
        for id, axis in self.axes.items():
            if id not in CartesianAxes and axis.data_line:
                r |= QRectF(axis.data_line.p1(), axis.data_line.p2())
        ## We leave a 5% margin on each side so the graph doesn't look overcrowded
        ## TODO: Perhaps change this from a fixed percentage to always round to a round number
        dx = r.width() / 20.0
        dy = r.height() / 20.0
        r.adjust(-dx, -dy, dx, dy)
        return r

    def transform_for_axes(self, x_axis = xBottom, y_axis = yLeft):
        """
            Returns the graph transform that maps from data to scene coordinates using axes ``x_axis`` and ``y_axis``.
        """
        if not (x_axis, y_axis) in self._transform_cache:
            # We must flip the graph area, becase Qt coordinates start from top left, while graph coordinates start from bottom left
            a = QRectF(self.graph_area)
            t = a.top()
            a.setTop(a.bottom())
            a.setBottom(t)
            self._transform_cache[(x_axis, y_axis)] = self.transform_from_rects(self.data_rect_for_axes(x_axis, y_axis), a)
        return self._transform_cache[(x_axis, y_axis)]

    def transform(self, axis_id, value):
        """
            Transforms the ``value`` from data to plot coordinates along the axis ``axis_id``.

            This function always ignores zoom. If you need to account for zooming, use :meth:`map_to_graph`.
        """
        if axis_id in XAxes:
            size = self.graph_area.width()
            margin = self.graph_area.left()
        else:
            size = self.graph_area.height()
            margin = self.graph_area.top()
        m, M = self.bounds_for_axis(axis_id)
        if m is None or M is None or M == m:
            return 0
        else:
            return margin + (value-m)/(M-m) * size

    def inv_transform(self, axis_id, value):
        """
            Transforms the ``value`` from plot to data coordinates along the axis ``axis_id``.

            This function always ignores zoom. If you need to account for zooming, use :meth:`map_from_graph`.
        """
        if axis_id in XAxes:
            size = self.graph_area.width()
            margin = self.graph_area.left()
        else:
            size = self.graph_area.height()
            margin = self.graph_area.top()
        m, M = self.bounds_for_axis(axis_id)
        if m is not None and M is not None:
            return m + (value-margin)/size * (M-m)
        else:
            return 0

    def bounds_for_axis(self, axis_id, try_auto_scale=True):
        if axis_id in self.axes and not self.axes[axis_id].auto_scale:
            return self.axes[axis_id].bounds()
        if try_auto_scale:
            lower, upper = orangeqt.Plot.bounds_for_axis(self, axis_id)
            if lower != upper:
                lower = lower - (upper-lower)/20.0
                upper = upper + (upper-lower)/20.0
            return lower, upper
        else:
            return None, None

    def enableYRaxis(self, enable=1):
        self.set_axis_enabled(yRight, enable)

    def enableLRaxis(self, enable=1):
        self.set_axis_enabled(yLeft, enable)

    def enableXaxis(self, enable=1):
        self.set_axis_enabled(xBottom, enable)

    def set_axis_enabled(self, axis, enable):
        if axis not in self.axes:
            self.add_axis(axis)
        self.axes[axis].setVisible(enable)
        self.replot()

    @staticmethod
    def axis_coordinate(point, axis_id):
        if axis_id in XAxes:
            return point.x()
        elif axis_id in YAxes:
            return point.y()
        else:
            return None

    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleTooltipText(self, example, indices = None, maxIndices = 20):
        if indices and type(indices[0]) == str:
            indices = [self.attributeNameIndex[i] for i in indices]
        if not indices:
            indices = list(range(len(self.dataDomain.attributes)))

        # don't show the class value twice
        if example.domain.classVar:
            classIndex = self.attributeNameIndex[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)

        text = "<b>Attributes:</b><br>"
        for index in indices[:maxIndices]:
            attr = self.attributeNames[index]
            if attr not in example.domain:  text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            elif example[attr].isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            else:                           text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(attr), Qt.escape(str(example[attr])))
        if len(indices) > maxIndices:
            text += "&nbsp;"*4 + " ... <br>"

        if example.domain.classVar:
            text = text[:-4]
            text += "<hr><b>Class:</b><br>"
            if example.getclass().isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(example.domain.classVar.name))
            else:                              text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain.classVar.name), Qt.escape(str(example.getclass())))

        if len(example.domain.getmetas()) != 0:
            text = text[:-4]
            text += "<hr><b>Meta attributes:</b><br>"
            # show values of meta attributes
            for key in example.domain.getmetas():
                try: text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain[key].name), Qt.escape(str(example[key])))
                except: pass
        return text[:-4]        # remove the last <br>

    # show a tooltip at x,y with text. if the mouse will move for more than 2 pixels it will be removed
    def showTip(self, x, y, text):
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self, QRect(x-3,y-3,6,6))

    def notify_legend_moved(self, pos):
        self._legend_moved = True
        l = self.legend_rect()
        g = getattr(self, '_legend_outside_area', QRectF())
        p = QPointF()
        rect = QRectF()
        offset = 20
        if pos.x() > g.right() - offset:
            self._legend.set_orientation(Qt.Vertical)
            rect.setRight(self._legend.boundingRect().width())
            p = g.topRight() - self._legend.boundingRect().topRight()
        elif pos.x() < g.left() + offset:
            self._legend.set_orientation(Qt.Vertical)
            rect.setLeft(self._legend.boundingRect().width())
            p = g.topLeft()
        elif pos.y() < g.top() + offset:
            self._legend.set_orientation(Qt.Horizontal)
            rect.setTop(self._legend.boundingRect().height())
            p = g.topLeft()
        elif pos.y() > g.bottom() - offset:
            self._legend.set_orientation(Qt.Horizontal)
            rect.setBottom(self._legend.boundingRect().height())
            p = g.bottomLeft() - self._legend.boundingRect().bottomLeft()

        if p.isNull():
            self._legend.set_floating(True, pos)
        else:
            self._legend.set_floating(False, p)

        if rect != self._legend_margin:
            orientation = Qt.Horizontal if rect.top() or rect.bottom() else Qt.Vertical
            self._legend.set_orientation(orientation)
            self.animate(self, 'legend_margin', rect, duration=100)

    def get_legend_margin(self):
        return self._legend_margin

    def set_legend_margin(self, value):
        self._legend_margin = value
        self.update_layout()
        self.update_axes()

    legend_margin = pyqtProperty(QRectF, get_legend_margin, set_legend_margin)

    def update_curves(self):
        if self.main_curve:
            self.main_curve.set_alpha_value(self.alpha_value)
        else:
            for c in self.plot_items():
                if isinstance(c, orangeqt.Curve) and not getattr(c, 'ignore_alpha', False):
                    au = c.auto_update()
                    c.set_auto_update(False)
                    c.set_point_size(self.point_width)
                    color = c.color()
                    color.setAlpha(self.alpha_value)
                    c.set_color(color)
                    c.set_auto_update(au)
                    c.update_properties()
        self.viewport().update()

    update_point_size = update_curves
    update_alpha_value = update_curves

    def update_antialiasing(self, use_antialiasing=None):
        if use_antialiasing is not None:
            self.antialias_plot = use_antialiasing

        self.setRenderHint(QPainter.Antialiasing, self.antialias_plot)

    def update_animations(self, use_animations=None):
        if use_animations is not None:
            self.animate_plot = use_animations
            self.animate_points = use_animations

    def update_performance(self, num_points = None):
        if self.auto_adjust_performance:
            if not num_points:
                if self.main_curve:
                    num_points = len(self.main_curve.points())
                else:
                    num_points = sum( len(c.points()) for c in self.curves )
            if num_points > self.disable_animations_threshold:
                self.disabled_animate_points = self.animate_points
                self.animate_points = False

                self.disabled_animate_plot = self.animate_plot
                self.animate_plot = False

                self.disabled_antialias_lines = self.animate_points
                self.antialias_lines = False

            elif hasattr(self, 'disabled_animate_points'):
                self.animate_points = self.disabled_animate_points
                del self.disabled_animate_points

                self.animate_plot = self.disabled_animate_plot
                del self.disabled_animate_plot

                self.antialias_lines = self.disabled_antialias_lines
                del self.disabled_antialias_lines

    def animate(self, target, prop_name, end_val, duration = None, start_val = None):
        for a in self._animations:
            if a.state() == QPropertyAnimation.Stopped:
                self._animations.remove(a)
        if self.animate_plot:
            a = QPropertyAnimation(target, prop_name)
            a.setEndValue(end_val)
            if start_val is not None:
                a.setStartValue(start_val)
            if duration:
                a.setDuration(duration)
            self._animations.append(a)
            a.start(QPropertyAnimation.KeepWhenStopped)
        else:
            target.setProperty(prop_name, end_val)

    def clear_selection(self):
        self.unselect_all_points()

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def pan(self, delta):
        if type(delta) == tuple:
            x, y = delta
        else:
            x, y = delta.x(), delta.y()
        t = self.zoom_transform()
        x = x / t.m11()
        y = y / t.m22()
        r = QRectF(self.zoom_rect)
        r.translate(-QPointF(x,y))
        self.ensure_inside(r, self.graph_area)
        self.zoom_rect = r

    def zoom_to_rect(self, rect):
        print(len(self.zoom_stack))
        self.ensure_inside(rect, self.graph_area)

        # add to zoom_stack if zoom_rect is larger
        if self.zoom_rect.width() > rect.width() or self.zoom_rect.height() > rect.height():
            self.zoom_stack.append(self.zoom_rect)

        self.animate(self, 'zoom_rect', rect, start_val = self.get_zoom_rect())

    def zoom_back(self):
        print(len(self.zoom_stack))
        if self.zoom_stack:
            rect = self.zoom_stack.pop()
            self.animate(self, 'zoom_rect', rect, start_val = self.get_zoom_rect())

    def reset_zoom(self):
        self._zoom_rect = None
        self.update_zoom()

    def zoom_transform(self):
        return self.transform_from_rects(self.zoom_rect, self.graph_area)

    def zoom_in(self, point):
        self.zoom(point, scale = 2)

    def zoom_out(self, point):
        self.zoom(point, scale = 0.5)

    def zoom(self, point, scale):
        print(len(self.zoom_stack))
        t, ok = self._zoom_transform.inverted()
        point = point * t
        r = QRectF(self.zoom_rect)
        i = 1.0/scale
        r.setTopLeft(point*(1-i) + r.topLeft()*i)
        r.setBottomRight(point*(1-i) + r.bottomRight()*i)

        self.ensure_inside(r, self.graph_area)

        # remove smaller zoom rects from stack
        while len(self.zoom_stack) > 0 and r.width() >= self.zoom_stack[-1].width() and r.height() >= self.zoom_stack[-1].height():
            self.zoom_stack.pop()

        self.zoom_to_rect(r)

    def get_zoom_rect(self):
        if self._zoom_rect:
            return self._zoom_rect
        else:
            return self.graph_area

    def set_zoom_rect(self, rect):
        self._zoom_rect = rect
        self._zoom_transform = self.transform_from_rects(rect, self.graph_area)
        self.update_zoom()

    zoom_rect = pyqtProperty(QRectF, get_zoom_rect, set_zoom_rect)

    @staticmethod
    def ensure_inside(small_rect, big_rect):
        if small_rect.width() > big_rect.width():
            small_rect.setWidth(big_rect.width())
        if small_rect.height() > big_rect.height():
            small_rect.setHeight(big_rect.height())

        if small_rect.right() > big_rect.right():
            small_rect.moveRight(big_rect.right())
        elif small_rect.left() < big_rect.left():
            small_rect.moveLeft(big_rect.left())

        if small_rect.bottom() > big_rect.bottom():
            small_rect.moveBottom(big_rect.bottom())
        elif small_rect.top() < big_rect.top():
            small_rect.moveTop(big_rect.top())

    def shuffle_points(self):
        if self.main_curve:
            self.main_curve.shuffle_points()

    def set_progress(self, done, total):
        if not self.widget:
            return

        if done == total:
            self.widget.progressBarFinished()
        else:
            self.widget.progressBarSet(100.0 * done / total)

    def start_progress(self):
        if self.widget:
            self.widget.progressBarInit()

    def end_progress(self):
        if self.widget:
            self.widget.progressBarFinished()

    def is_axis_auto_scale(self, axis_id):
        if axis_id not in self.axes:
            return axis_id not in self.data_range
        return self.axes[axis_id].auto_scale

    def axis_line(self, rect, id, invert_y = False):
        if invert_y:
            r = QRectF(rect)
            r.setTop(rect.bottom())
            r.setBottom(rect.top())
            rect = r
        if id == xBottom:
            line = QLineF(rect.topLeft(), rect.topRight())
        elif id == xTop:
            line = QLineF(rect.bottomLeft(), rect.bottomRight())
        elif id == yLeft:
            line = QLineF(rect.topLeft(), rect.bottomLeft())
        elif id == yRight:
            line = QLineF(rect.topRight(), rect.bottomRight())
        else:
            line = None
        return line

    def color(self, role, group = None):
        if group:
            return self.palette().color(group, role)
        else:
            return self.palette().color(role)

    def set_palette(self, p):
        '''
            Sets the plot palette to ``p``.

            :param p: The new color palette
            :type p: :obj:`.QPalette`
        '''
        self.setPalette(p)
        self.replot()

    def update_theme(self):
        '''
            Updates the current color theme, depending on the value of :attr:`theme_name`.
        '''
        if self.theme_name.lower() == 'default':
            self.set_palette(OWPalette.System)
        elif self.theme_name.lower() == 'light':
            self.set_palette(OWPalette.Light)
        elif self.theme_name.lower() == 'dark':
            self.set_palette(OWPalette.Dark)
