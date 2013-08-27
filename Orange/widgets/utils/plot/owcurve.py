'''
##############################
Curve (``owcurve``)
##############################

.. class:: OWPlotItem

    This class represents a base for any item than can be added to a plot.

    .. method:: attach(plot)

        Attaches this item to ``plot``. The Plot takes the ownership of this item.

        :param plot: the plot to which to add this item
        :type plot: :obj:`.OWPlot`

        :seealso: :meth:`.OWPlot.add_item`.

    .. method:: detach()

        Removes this item from its plot. The item's ownership is returned to Python.

    .. method:: plot()

        :returns: The plot this item is attached to. If the item is not attached to any plot, ``None`` is returned.
        :rtype: :obj:`.OWPlot`

    .. method:: data_rect()

        Returns the bounding rectangle of this item in data coordinates. This method is used in autoscale calculations.

    .. method:: set_data_rect(rect)

        :param rect: The new bounding rectangle in data coordinates
        :type rect: :obj:`.QRectF`

    .. method:: set_graph_transform(transform)

        Sets the graph transform (the transformation that maps from data to plot coordinates) for this item.

    .. method:: graph_transform()

        :returns: The current graph transformation.
        :rtype: QTransform

    .. method:: set_zoom_transform(transform)

        Sets the zoom transform (the transformation that maps from plot to scene coordinates) for this item.

    .. method:: zoom_transform()

        :returns: The current zoom transformation.
        :rtype: QTransform

    .. method:: set_axes(x_axis, y_axis)

        Sets the pair of axes used for positioning this item.

    .. method:: axes()

        :returns: The item's pair of axes
        :rtype: tuple of int int

    .. method:: update_properties()

        Called by the plot, this function is supposed to updates the item's internal state to match its settings.

        The default implementation does nothing and shold be reimplemented by subclasses.

    .. method:: register_points()

        If this item constains any points (of type :obj:`.OWPoint`), add them to the plot in this function.

        The default implementation does nothing.

    .. method:: set_in_background(background)

        If ``background`` is ``True``, the item is moved to be background of this plot, behind other items and axes.
        Otherwise, it's brought to the front, in front of axes.

        The default in ``False``, so that items apper in front of axes.

    .. method:: is_in_background()

        Returns if item is in the background, set with :meth:`set_in_background`.

    **Subclassing**

        Often you will want to create a custom curve class that inherits from OWCurve.
        For this purpose, OWPlotItem provides two virtual methods: :meth:`paint` and :meth:`update_properties`.

        * ``update_properties()`` is called whenever a curve or the plot is changed and needs to be updated.
          In this method, child items and other members should be recalculated and updated.
          The OWCurve even provides a number of methods for asynchronous (threaded) updating.

        * `paint()` is called whenever the item needs to be painted on the scene.
          This method is called more often, so it's advisable to avoid long operation in the method.

        Most provided plot items, including :obj:`OWCurve`, :obj:`OWMultiCurve` and utility curves in :mod:`.owtools`
        only reimplement the first method, because they are optimized for performance with large data sets.

.. autoclass:: OWCurve

.. class:: OWMultiCurve

    A multi-curve is a curve in which each point can have its own properties.
    The point coordinates can be set by calling :meth:`.OWCurve.set_data`, just like in a normal curve.

    In addition, OWMultiCurve provides methods for setting properties for individual points.
    Each of this methods take a list as a parameter. If the list has less elements that the curve's data,
    the first element is used for all points.

    .. method:: set_point_colors(lst)

        :param lst: The list of colors to assign to points
        :type lst: list of QColor

        .. seealso:: :meth:`.OWPoint.set_color`

    .. method:: set_point_sizes(lst)

        :param lst: The list of sizes to assign to points
        :type lst: list of int

        .. seealso:: :meth:`.OWPoint.set_size`

    .. method:: set_point_symbols(lst)

        :param lst: The list of symbols to assign to points
        :type lst: list of int

        .. seealso:: :meth:`.OWPoint.set_symbol`

'''

from .owconstants import *
import orangeqt

OWPlotItem = orangeqt.PlotItem

#@deprecated_members({
#    "setYAxis" : "set_y_axis",
#    "setData" : "set_data"
#})

class OWCurve(orangeqt.Curve):
    """
        This class represents a curve on a plot.
        It is essentially a plot item with a series of data points or a continuous line.

        :param xData: list of x coordinates
        :type xData: list of float

        :param yData: list of y coordinates
        :type yData: list of float

        :param x_axis_key: The x axis of this curve
        :type x_axis_key: int

        :param y_axis_key: The y axis of this curve
        :type y_axis_key: int

        :param tooltip: The curve's tooltip
        :type tooltip: str

        .. note::

            All the points or line segments in an OWCurve have the same properties.
            Different points in one curve are supported by the :obj:`.OWMultiCurve` class.


        .. method:: point_item(x, y, size=0, parent=None)

            Returns a single point with this curve's properties.
            It is useful for representing the curve, for example in the legend.

            :param x: The x coordinate of the point.
            :type x: float

            :param y: The y coordinate of the point.
            :type y: float

            :param size: If nonzero, this argument determines the size of the resulting point.
                         Otherwise, the point is created with the curve's :meth:`OWCurve.point_size`
            :type size: int

            :param parent: An optional parent for the returned item.
            :type parent: :obj:`.QGraphicsItem`

        .. attribute:: name

            The name of the curve, used in the legend or in tooltips.

        .. method:: set_data(x_data, y_data)

            Sets the curve's data to a list of coordinates specified by ``x_data`` and ``y_data``.

        .. method:: data()

            :returns: The curve's data as a list of data points.
            :rtype: list of tuple of float float

        .. method:: set_style(style)

            Sets the curve's style to ``style``.

            The following values are recognized by OWCurve:

            ===================  ===============================================
            Value                Result
            ===================  ===============================================
            OWCurve.Points       Only points are shown, no lines
            OWCurve.Lines        A continuous line is shown, no points
            OWCurve.LinesPoints  Both points and lines between them are shown
            OWCurve.Dots         A dotted line is shown, no points
            OWCurve.NoCurve      Deprecated, same as ``OWCurve.Points``
            ===================  ===============================================

            Curve subclasses can use this value for different drawing modes.
            Values up to OWCurve.UserCurve are reserved, so use only higher numbers, like the following example::

                class MyCurve(OWCurve):
                    PonyStyle = OWCurve.UserCurve + 42

                    def draw_ponies()
                        # Draw type-specific things here

                    def update_properties(self):
                        if self.style() == PonyStyle:
                            self.draw_ponies()
                        else:
                            OWCurve.update_properties(self)

        .. method:: style()

            :return: The curve's style, set with :meth:`set_style`
            :rtype: int

        .. method:: cancel_all_updates()

            Cancel all pending threaded updates and block until they are finished.
            This is usually called before starting a new round of updates.

        .. method:: update_number_of_items()

            Resizes the point list so that it matches the number of data points in :meth:`data`

        .. method:: update_point_coordinates()

            Sets the coordinates of each point to match :meth:`data`.

        .. method:: update_point_positions()

            Sets the scene positions of the points to match their data coordinates.
    """
    NoCurve = orangeqt.Curve.Points

    def __init__(self, xData=[], yData=[], x_axis_key=xBottom, y_axis_key=yLeft, tooltip=None):
        orangeqt.Curve.__init__(self, xData, yData)
        self.set_axes(x_axis_key, y_axis_key)
        if tooltip:
            self.setToolTip(tooltip)
        self.name = ''

OWMultiCurve = orangeqt.MultiCurve


