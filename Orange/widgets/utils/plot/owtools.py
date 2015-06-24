'''
##############################
Plot tools (``owtools``)
##############################

.. autofunction:: resize_plot_item_list

.. autofunction:: move_item

.. autofunction:: move_item_xy

.. autoclass:: TooltipManager
    :members:

.. autoclass:: PolygonCurve
    :members:
    :show-inheritance:

.. autoclass:: RectangleCurve
    :members:
    :show-inheritance:

.. autoclass:: CircleCurve
    :members:
    :show-inheritance:

.. autoclass:: UnconnectedLinesCurve
    :members:
    :show-inheritance:

.. autoclass:: Marker
    :members:
    :show-inheritance:

'''

from PyQt4.QtGui import (QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
    QPolygonF, QGraphicsPolygonItem, QGraphicsEllipseItem, QPen, QBrush,
    QGraphicsPixmapItem, QGraphicsPathItem, QPainterPath, qRgb, QImage, QPixmap)
from PyQt4.QtCore import Qt, QRectF, QPointF, QPropertyAnimation, qVersion
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.utils.scaling import get_variable_values_sorted

from .owcurve import *
from .owpalette import OWPalette
import orangeqt
import Orange

#from Orange.preprocess.scaling import get_variable_values_sorted
#from Orange import orangeom
#import ColorPalette

def resize_plot_item_list(lst, size, item_type, parent):
    """
        Efficiently resizes a list of QGraphicsItems (PlotItems, Curves, etc.).
        If the list is to be reduced, i.e. if len(lst) > size, then the extra items are first removed from the scene.
        If items have to be added to the scene, new items will be of type ``item_type`` and will have ``parent``
        as their parent item.

        The list is resized in place, this function returns nothing.

        :param lst: The list to be resized
        :type lst: list of QGraphicsItem

        :param size: The needed size of the list
        :type size: int

        :param item_type: The type of items that should be added if the list has to be increased
        :type item_type: type

        :param parent: Any new items will have this as their parent item
        :type parent: QGraphicsItem
    """
    n = len(lst)
    if n > size:
        for i in lst[size:]:
            i.setParentItem(None)
            if i.scene():
                i.scene().removeItem(i)
        del lst[size:]
    elif n < size:
        lst.extend(item_type(parent) for i in range(size - n))

def move_item(item, pos, animate = True, duration = None):
    '''
        Animates ``item`` to move to position ``pos``.
        If animations are turned off globally, the item is instead move immediately, without any animation.

        :param item: The item to move
        :type item: QGraphicsItem

        :param pos: The final position of the item
        :type pos: QPointF

        :param duration: The duration of the animation. If unspecified, Qt's default value of 250 miliseconds is used.
        :type duration: int
    '''
    if not duration:
        duration = 250
    orangeqt.PlotItem.move_item(item, pos, animate, duration)

def move_item_xy(item, x, y, animate = True, duration = None):
    '''
        Same as
        move_item(item, QPointF(x, y), duration)
    '''
    move_item(item, QPointF(x, y), animate, duration)

class TooltipManager:
    """
        A dynamic tool tip manager.

        :param plot: The plot used for transforming the coordinates
        :type plot: :obj:`.OWPlot`
    """
    def __init__(self, plot):
        self.graph = plot
        self.positions=[]
        self.texts=[]

    def addToolTip(self, x, y, text, customX = 0, customY = 0):
        """
            Adds a tool tip. If a tooltip with the same name already exists, it updates it instead of adding a new one.

            :param x: The x coordinate of the tip, in data coordinates.
            :type x: float

            :param y: The y coordinate of the tip, in data coordinates.
            :type y: float

            :param text: The text to show in the tip.
            :type text: str or int

            :param customX: The maximum horizontal distance in pixels from the point (x,y) at which to show the tooltip.
            :type customX: float

            :param customY: The maximum vertical distance in pixels from the point (x,y) at which to show the tooltip.
            :type customY: float

            If ``customX`` and ``customY`` are omitted, a default of 6 pixels is used.
        """
        self.positions.append((x,y, customX, customY))
        self.texts.append(text)

    #Decides whether to pop up a tool tip and which text to pop up
    def maybeTip(self, x, y):
        """
            Decides whether to pop up a tool tip and which text to show in it.

            :param x: the x coordinate of the mouse in data coordinates.
            :type x: float

            :param y: the y coordinate of the mouse in data coordinates.
            :type y: float

            :returns: A tuple consisting of the ``text``, ``x`` and ``y`` arguments to :meth:`addToolTip` of the
                      closest point.
            :rtype: tuple of (int or str), float, float
        """
        if len(self.positions) == 0: return ("", -1, -1)
        dists = [max(abs(x-position[0])- position[2],0) + max(abs(y-position[1])-position[3], 0) for position in self.positions]
        nearestIndex = dists.index(min(dists))

        intX = abs(self.graph.transform(xBottom, x) - self.graph.transform(xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.graph.transform(yLeft, y) - self.graph.transform(yLeft, self.positions[nearestIndex][1]))
        if self.positions[nearestIndex][2] == 0 and self.positions[nearestIndex][3] == 0:   # if we specified no custom range then assume 6 pixels
            if intX + intY < 6:  return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
            else:                return ("", None, None)
        else:
            if abs(self.positions[nearestIndex][0] - x) <= self.positions[nearestIndex][2] and abs(self.positions[nearestIndex][1] - y) <= self.positions[nearestIndex][3]:
                return (self.texts[nearestIndex], x, y)
            else:
                return ("", None, None)

    def removeAll(self):
        """
            Removes all tips
        """
        self.positions = []
        self.texts = []

class PolygonCurve(OWCurve):
    """
        A plot item that shows a filled or empty polygon.

        :param pen: The pen used to draw the polygon's outline
        :type pen: :obj:`.QPen`

        :param brush: The brush used to paint the polygon's inside
        :type brush: :obj:`.QBrush`

        :param xData: The list of x coordinates
        :type xData: list of float

        :param yData: The list of y coordinates
        :type yData: list of float

        :param tooltip: The tool tip shown when hovering over this curve
        :type tooltip: str
    """
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = [], yData = [], tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self._data_polygon = self.polygon_from_data(xData, yData)
        self._polygon_item = QGraphicsPolygonItem(self)
        self.set_pen(pen)
        self.set_brush(brush)

    def update_properties(self):
        self._polygon_item.setPolygon(self.graph_transform().map(self._data_polygon))
        self._polygon_item.setPen(self.pen())
        self._polygon_item.setBrush(self.brush())

    @staticmethod
    def polygon_from_data(xData, yData):
        """
            Creates a polygon from a list of x and y coordinates.

            :returns: A polygon with point corresponding to ``xData`` and ``yData``.
            :rtype: QPolygonF
        """
        if xData and yData:
            n = min(len(xData), len(yData))
            p = QPolygonF(n+1)
            for i in range(n):
                p[i] = QPointF(xData[i], yData[i])
            p[n] = QPointF(xData[0], yData[0])
            return p
        else:
            return QPolygonF()

    def set_data(self, xData, yData):
        self._data_polygon = self.polygon_from_data(xData, yData)
        OWCurve.set_data(self, xData, yData)

class RectangleCurve(OWCurve):
    """
        A plot item that shows a rectangle.

        This class accepts the same options as :obj:`.PolygonCurve`.
        The rectangle is calculated as the smallest rectangle that contains all points in ``xData`` and ``yData``.
    """
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None, tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self.set_pen(pen)
        self.set_brush(brush)
        self._item = QGraphicsRectItem(self)

    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())

class UnconnectedLinesCurve(orangeqt.UnconnectedLinesCurve):
    """
        A plot item that shows a series of unconnected straight lines.

        :param name: The name of this curve. :seealso: :attr:`.OWCurve.name`
        :type name: str

        :param pen: The pen used to draw the lines
        :type pen: QPen

        :param xData: The list of x coordinates
        :type xData: list of float

        :param yData: The list of y coordinates
        :type yData: list of float

        The data should contain an even number of elements. Lines are drawn between the `n`-th and
        `(n+1)`-th point for each even `n`.
    """
    def __init__(self, name, pen = QPen(Qt.black), xData = None, yData = None):
        orangeqt.UnconnectedLinesCurve.__init__(self)
        self.set_data(xData, yData)
        if pen:
            self.set_pen(pen)
        self.name = name

class CircleCurve(OWCurve):
    """
        Displays a circle on the plot

        :param pen: The pen used to draw the outline of the circle
        :type pen: QPen

        :param brush: The brush used to paint the inside of the circle
        :type brush: QBrush

        :param xCenter: The x coordinate of the circle's center
        :type xCenter: float

        :param yCenter: The y coordinate of the circle's center
        :type yCenter: float

        :param radius: The circle's radius
        :type radius: float
    """
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.NoBrush), xCenter = 0.0, yCenter = 0.0, radius = 1.0):
        OWCurve.__init__(self)
        self._item = QGraphicsEllipseItem(self)
        self.center = xCenter, yCenter
        self.radius = radius
        self._rect = QRectF(xCenter-radius, yCenter-radius, 2*radius, 2*radius)
        self.set_pen(pen)
        self.set_brush(brush)

    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())

    def data_rect(self):
        x, y = self.center
        r = self.radius
        return QRectF(x-r, y-r, 2*r, 2*r)

class Marker(orangeqt.PlotItem):
    """
        Displays a text marker on the plot.

        :param text: The text to display. It can be HTML-formatted
        :type tex: str

        :param x: The x coordinate of the marker's position
        :type x: float

        :param y: The y coordinate of the marker's position
        :type y: float

        :param align: The text alignment
        :type align:

        :param bold: If ``True``, the text will be show bold.
        :type bold: int

        :param color: The text color
        :type color: QColor

        :param brushColor: The color of the brush user to paint the background
        :type color: QColor

        :param size: Font size
        :type size: int

        Markers have the QGraphicsItem.ItemIgnoresTransformations flag set by default,
        so text remains the same size when zooming. There is no need to scale the manually.
    """
    def __init__(self, text, x, y, align, bold = 0, color = None, brushColor = None, size=None):
        orangeqt.PlotItem.__init__(self)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._item = QGraphicsTextItem(text, parent=self)
        self._data_point = QPointF(x,y)
        f = self._item.font()
        f.setBold(bold)
        if size:
            f.setPointSize(size)
        self._item.setFont(f)
        self._item.setPos(x, y)

    def update_properties(self):
        self._item.setPos(self.graph_transform().map(self._data_point))


class ProbabilitiesItem(orangeqt.PlotItem):
    """
        Displays class probabilities in the background

        :param classifier: The classifier for which the probabilities are calculated
        :type classifier: orange.P2NN

        :param granularity: The size of individual cells
        :type granularity: int

        :param scale: The data scale factor
        :type scale: float

        :param spacing: The space between cells
        :param spacing: int

        :param rect: The rectangle into which to draw the probabilities. If unspecified, the entire plot is used.
        :type rect: QRectF
    """
    def __init__(self, classifier, granularity, scale, spacing, rect=None):
        orangeqt.PlotItem.__init__(self)
        self.classifier = classifier
        self.rect = rect
        self.granularity = granularity
        self.scale = scale
        self.spacing = spacing
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.set_in_background(True)
        self.setZValue(ProbabilitiesZValue)

    def update_properties(self):
        ## Mostly copied from OWScatterPlotGraph
        if not self.plot():
            return

        if not self.rect:
            x,y = self.axes()
            self.rect = self.plot().data_rect_for_axes(x,y)
        s = self.graph_transform().mapRect(self.rect).size().toSize()
        if not s.isValid():
            return
        rx = s.width()
        ry = s.height()

        rx -= rx % self.granularity
        ry -= ry % self.granularity

        p = self.graph_transform().map(QPointF(0, 0)) - self.graph_transform().map(self.rect.topLeft())
        p = p.toPoint()

        ox = p.x()
        oy = -p.y()

        if self.classifier.classVar.is_continuous:
            imagebmp = orangeom.potentialsBitmap(self.classifier, rx, ry, ox, oy, self.granularity, self.scale)
            palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
        else:
            imagebmp, nShades = orangeom.potentialsBitmap(self.classifier, rx, ry, ox, oy, self.granularity, self.scale, self.spacing)
            palette = []
            sortedClasses = get_variable_values_sorted(self.classifier.domain.classVar)
            for cls in self.classifier.classVar.values:
                color = self.plot().discPalette.getRGB(sortedClasses.index(cls))
                towhite = [255-c for c in color]
                for s in range(nShades):
                    si = 1-float(s)/nShades
                    palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
            palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

        self.potentialsImage = QImage(imagebmp, rx, ry, QImage.Format_Indexed8)
        self.potentialsImage.setColorTable(ColorPaletteDlg.signedPalette(palette) if qVersion() < "4.5" else palette)
        self.potentialsImage.setNumColors(256)
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.potentialsImage))
        self.pixmap_item.setPos(self.graph_transform().map(self.rect.bottomLeft()))

    def data_rect(self):
        return self.rect if self.rect else QRectF()


#@deprecated_members({
#        'enableX' : 'set_x_enabled',
#        'enableY' : 'set_y_enabled',
#        'xEnabled' : 'is_x_enabled',
#        'yEnabled' : 'is_y_enabled',
#        'setPen' : 'set_pen'
#    })
class PlotGrid(orangeqt.PlotItem):
    """
        Draws a grid onto the plot

        :param plot: If specified, the grid will be attached to the ``plot``.
        :type plot: :obj:`.OWPlot`
    """
    def __init__(self, plot = None):
        orangeqt.PlotItem.__init__(self)
        self._x_enabled = True
        self._y_enabled = True
        self._path_item = QGraphicsPathItem(self)
        self.set_in_background(True)
        if plot:
            self.attach(plot)
            self._path_item.setPen(plot.color(OWPalette.Grid))

    def set_x_enabled(self, b):
        """
            Enables or disabled vertial grid lines
        """
        if b < 0:
            b = not self._x_enabled
        self._x_enabled = b
        self.update_properties()

    def is_x_enabled(self):
        """
            Returns whether vertical grid lines are enabled
        """
        return self._x_enabled

    def set_y_enabled(self, b):
        """
            Enables or disabled horizontal grid lines
        """
        if b < 0:
            b = not self._y_enabled
        self._y_enabled = b
        self.update_properties()

    def is_y_enabled(self):
        """
            Returns whether horizontal grid lines are enabled
        """
        return self._y_enabled

    def set_pen(self, pen):
        """
            Sets the pen used for drawing the grid lines
        """
        self._path_item.setPen(pen)

    def update_properties(self):
        p = self.plot()
        if p is None:
            return
        x_id, y_id = self.axes()
        rect = p.data_rect_for_axes(x_id, y_id)
        path = QPainterPath()
        if self._x_enabled and x_id in p.axes:
            for pos, label, size, _w in p.axes[x_id].ticks():
                path.moveTo(pos, rect.bottom())
                path.lineTo(pos, rect.top())
        if self._y_enabled and y_id in p.axes:
            for pos, label, size, _w in p.axes[y_id].ticks():
                path.moveTo(rect.left(), pos)
                path.lineTo(rect.right(), pos)
        self._path_item.setPath(self.graph_transform().map(path))

