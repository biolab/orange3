"""
#####################
Point (``owpoint``)
#####################

.. class:: OWPoint

    Represents a point on the plot, usually a part of a curve, or in a legend item.

    The point is identified with its symbol, color, size, label, and state, where the state can be ether unselected (default),
    marker, or selected. All these attributes can be changed and retrieved after the point is constructed.
    For example, color can be set with :meth:`set_color`, while the current color is returned by :meth:`color`.
    There are similarily named function for the other attributes.

    .. method:: __init__(symbol, color, size)

        :param symbol: The point symbol.
        :type symbol: int

        :param color: The point color.
        :type color: QColor

        :param size: The point size.
        :type size: int

    .. method:: set_color(color)

        Sets the point's color to ``color``

    .. method:: color()

        :returns: the point's color

    .. method:: set_size(size)

        Sets the point's size to ``size``

    .. method:: size()

        :returns: the point's size

    .. method:: set_symbol(symbol)

        Sets the point's symbol to ``symbol``

    .. method:: symbol()

        :returns: the point's symbol

    .. method:: set_selected(selected)

        Sets the point's selected state to ``selected``

    .. method:: is_selected()

        :returns: ``True`` if the point is selected, ``False`` otherwise.

    .. method:: set_marked(marked)

        Sets the point's marked state to ``marked``

    .. method:: is_marked()

        :returns: ``True`` if the point is marked, ``False`` otherwise.

    .. method:: set_label(label)

        Sets the point's label to ``label``.
        The label is displayed under the symbol.

    .. method:: label()

        :returns: The point`s label, set with :meth:`set_label`
        :rtype: str
"""

from PyQt4.QtGui import QGraphicsPathItem, QPen, QBrush
from PyQt4.QtCore import Qt, QPointF

from . import orangeqt

OWPoint = orangeqt.Point
