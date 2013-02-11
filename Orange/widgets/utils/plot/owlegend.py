"""
##########################
Plot legend (``owlegend``)
##########################

.. autoclass:: OWLegendItem

.. autoclass:: OWLegendTitle

.. autoclass:: OWLegend
    :members:

"""

from PyQt4.QtGui import QGraphicsTextItem, QGraphicsRectItem, QGraphicsObject, QPen, QLinearGradient
from PyQt4.QtCore import QPointF, QRectF, Qt, QPropertyAnimation, QSizeF

from .owpoint import *
from .owcurve import OWCurve
from .owtools import move_item_xy
from .owpalette import OWPalette

PointColor = 1
PointSize = 2
PointSymbol = 4

class OWLegendItem(QGraphicsObject):
    """
        Represents a legend item with a title and a point symbol.

        :param name: The text to display
        :type name: str

        :param point: The point symbol
        :type point: :obj:`.OWPoint`

        :param parent: The parent item, passed to QGraphicsItem
        :type parent: :obj:`QGraphicsItem`

        .. seealso:: :meth:`.OWLegend.add_item`, :meth:`.OWLegend.add_curve`.
    """
    def __init__(self, name, point, parent):
        QGraphicsObject.__init__(self, parent)
        self.text_item = QGraphicsTextItem(name, self)
        if point:
            s = point.size()
            height = max(2*s, self.text_item.boundingRect().height())
        else:
            height = self.text_item.boundingRect().height()
        p = 0.5 * height
        self.text_item.setPos(height, 0)
        self.point_item = point
        if point:
            self.point_item.setParentItem(self)
            self.point_item.setPos(p, p)
        self._rect = QRectF(0, 0, height + self.text_item.boundingRect().width(), height )
        self.rect_item = QGraphicsRectItem(self._rect, self)
        self.rect_item.setPen(QPen(Qt.NoPen))
        self.rect_item.stackBefore(self.text_item)
        if self.point_item:
            self.rect_item.stackBefore(self.point_item)

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget):
        pass

class OWLegendTitle(QGraphicsObject):
    """
        A legend item that shows ``text`` with a bold font and no symbol.
    """
    def __init__(self, text, parent):
        QGraphicsObject.__init__(self, parent)
        self.text_item = QGraphicsTextItem(text + ':', self)
        f = self.text_item.font()
        f.setBold(True)
        self.text_item.setFont(f)
        self.rect_item = QGraphicsRectItem(self.text_item.boundingRect(), self)
        self.rect_item.setPen(QPen(Qt.NoPen))
        self.rect_item.stackBefore(self.text_item)

    def boundingRect(self):
        return self.text_item.boundingRect()

    def paint(self, painter, option, widget):
        pass

class OWLegendGradient(QGraphicsObject):

    gradient_width = 20

    def __init__(self, palette, values, parent):
        QGraphicsObject.__init__(self, parent)
        self.parent = parent
        self.palette = palette
        self.values = values
        self.legend = parent
        self.label_items = [QGraphicsTextItem(text, self) for text in values]
        for i in self.label_items:
            i.setTextWidth(50)

        self.rect = QRectF()

        self.gradient_item = QGraphicsRectItem(self)
        self.gradient = QLinearGradient()
        self.gradient.setStops([(v*0.1, self.palette[v*0.1]) for v in range(11) ])
        self.orientation = Qt.Horizontal
        self.set_orientation(Qt.Vertical)

    def set_orientation(self, orientation):
        if self.orientation == orientation:
            return

        self.orientation = orientation

        if self.orientation == Qt.Vertical:
            height = max([item.boundingRect().height() for item in self.label_items])
            total_height = height * max(5, len(self.label_items))
            interval = (total_height - self.label_items[-1].boundingRect().height()) / (len(self.label_items) -1)
            self.gradient_item.setRect(10, 0, self.gradient_width, total_height)
            self.gradient.setStart(10, 0)
            self.gradient.setFinalStop(10, total_height)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            y = 0
            x = 30
            for item in self.label_items:
                move_item_xy(item, x, y, self.parent.graph.animate_plot)
                y += interval
            self.rect = QRectF(10, 0, self.gradient_width + max([item.boundingRect().width() for item in self.label_items]), self.label_items[0].boundingRect().height() * max(5, len(self.label_items)))
        else:
            width = 50
            height = max([item.boundingRect().height() for item in self.label_items])
            total_width = width * max(5, len(self.label_items))
            interval = (total_width - self.label_items[-1].boundingRect().width()) / (len(self.label_items) -1)

            self.gradient_item.setRect(0, 0, total_width, self.gradient_width)
            self.gradient.setStart(0, 0)
            self.gradient.setFinalStop(total_width, 0)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            x = 0
            y = 30
            for item in self.label_items:
                move_item_xy(item, x, y, self.parent.graph.animate_plot)
                x += interval
            self.rect = QRectF(0, 0, total_width, self.gradient_width + height)

    def boundingRect(self):
        return getattr(self, 'rect', QRectF())

    def paint(self, painter, option, widget):
        pass


class OWLegend(QGraphicsObject):
    """
        A legend for :obj:`.OWPlot`.

        Its items are arranged into a hierarchy by `category`. This is useful when points differ in more than one attribute.
        In such a case, there can be one category for point color and one for point shape. Usually the category name
        will be the name of the attribute, while the item's title will be the value.

        Arbitrary categories can be created, for an example see :meth:`.OWPlot.update_axes`, which creates a special category
        for unused axes.
        decimals
        .. image:: files/legend-categories.png

        In the image above, `type` and `milk` are categories with 7 and 2 possible values, respectively.

    """
    def __init__(self, graph, scene):
        QGraphicsObject.__init__(self)
        if scene:
            scene.addItem(self)
        self.graph = graph
        self.curves = []
        self.items = {}
        self.attributes = []
        self.point_attrs = {}
        self.point_vals = {}
        self.default_values = {
                               PointColor : Qt.black,
                               PointSize : 8,
                               PointSymbol : OWPoint.Ellipse
                               }
        self.box_rect = QRectF()
        self.setFiltersChildEvents(True)
        self.setFlag(self.ItemHasNoContents, True)
        self.mouse_down = False
        self._orientation = Qt.Vertical
        self.max_size = QSizeF()
        self._floating = True
        self._floating_animation = None
        self._mouse_down_pos = QPointF()

    def clear(self):
        """
            Removes all items from the legend
        """
        for lst in self.items.values():
            for i in lst:
                i.setParentItem(None)
                if self.scene():
                    self.scene().removeItem(i)
        self.items = {}
        self.update_items()


    def add_curve(self, curve):
        """
            Adds a legend item with the same point symbol and name as ``curve``.

            If the curve's name contains the equal sign (=), it is split at that sign. The first part of the curve
            is a used as the category, and the second part as the value.
        """
        i = curve.name.find('=')
        if i == -1:
            cat = ''
            name = curve.name
        else:
            cat = curve.name[:i]
            name = curve.name[i+1:]
        self.add_item(cat, name, curve.point_item(0, 0, 0))

    def add_item(self, category, value, point):
        """
            Adds an item with title ``value`` and point symbol ``point`` to the specified ``category``.
        """
        if category not in self.items:
            self.items[category] = [OWLegendTitle(category, self)]
        self.items[category].append(OWLegendItem(str(value), point, self))
        self.update_items()

    def add_color_gradient(self, title, values):
        if len(values) < 2:
            # No point in showing a gradient with less that two values
            return
        if title in self.items:
            self.remove_category(title)
        item = OWLegendGradient(self.graph.contPalette, [str(v) for v in values], self)
        self.items[title] = [OWLegendTitle(title, self), item]
        self.update_items()

    def remove_category(self, category):
        """
            Removes ``category`` and all items that belong to it.
        """
        if category not in self.items:
            return
        if self.scene():
            for item in self.items[category]:
                self.scene().removeItem(item)
        del self.items[category]

    def update_items(self):
        """
            Updates the legend, repositioning the items according to the legend's orientation.
        """
        self.box_rect = QRectF()
        x = y = 0

        for lst in self.items.values():
            for item in lst:
                if hasattr(item, 'text_item'):
                    item.text_item.setDefaultTextColor(self.graph.color(OWPalette.Text))
                if hasattr(item, 'rect_item'):
                    item.rect_item.setBrush(self.graph.color(OWPalette.Canvas))
                if hasattr(item, 'set_orientation'):
                    item.set_orientation(self._orientation)

        if self._orientation == Qt.Vertical:
            for lst in self.items.values():
                for item in lst:
                    if self.max_size.height() and y and y + item.boundingRect().height() > self.max_size.height():
                        y = 0
                        x = x + item.boundingRect().width()
                    self.box_rect = self.box_rect | item.boundingRect().translated(x, y)
                    move_item_xy(item, x, y, self.graph.animate_plot)
                    y = y + item.boundingRect().height()
        elif self._orientation == Qt.Horizontal:
            for lst in self.items.values():
                max_h = max(item.boundingRect().height() for item in lst)
                for item in lst:
                    if self.max_size.width() and x and x + item.boundingRect().width() > self.max_size.width():
                        x = 0
                        y = y + max_h
                    self.box_rect = self.box_rect | item.boundingRect().translated(x, y)
                    move_item_xy(item, x, y, self.graph.animate_plot)
                    x = x + item.boundingRect().width()
                if lst:
                    x = 0
                    y = y + max_h

    def mouseMoveEvent(self, event):
        self.graph.notify_legend_moved(event.scenePos())
        if self._floating:
            p = event.scenePos() - self._mouse_down_pos
            if self._floating_animation and self._floating_animation.state() == QPropertyAnimation.Running:
                self.set_pos_animated(p)
            else:
                self.setPos(p)
        event.accept()

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        self.mouse_down = True
        self._mouse_down_pos = event.scenePos() - self.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.unsetCursor()
        self.mouse_down = False
        self._mouse_down_pos = QPointF()
        event.accept()

    def boundingRect(self):
        return self.box_rect

    def paint(self, painter, option, widget=None):
        pass

    def set_orientation(self, orientation):
        """
            Sets the legend's orientation to ``orientation``.
        """
        self._orientation = orientation
        self.update_items()

    def orientation(self):
        return self._orientation

    def set_pos_animated(self, pos):
        if (self.pos() - pos).manhattanLength() < 6 or not self.graph.animate_plot:
            self.setPos(pos)
        else:
            t = 250
            if self._floating_animation and self._floating_animation.state() == QPropertyAnimation.Running:
                t = t - self._floating_animation.currentTime()
                self._floating_animation.stop()
            self._floating_animation = QPropertyAnimation(self, 'pos')
            self._floating_animation.setEndValue(pos)
            self._floating_animation.setDuration(t)
            self._floating_animation.start(QPropertyAnimation.KeepWhenStopped)

    def set_floating(self, floating, pos=None):
        """
            If floating is ``True``, the legend can be dragged with the mouse.
            Otherwise, it's fixed in its position.

            If ``pos`` is specified, the legend is moved there.
        """
        if floating == self._floating:
            return
        self._floating = floating
        if pos:
            if floating:
                self.set_pos_animated(pos - self._mouse_down_pos)
            else:
                self.set_pos_animated(pos)

