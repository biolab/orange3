"""Legend classes to use with `QGraphicsScene` objects."""
import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLinearLayout, QGraphicsView, QApplication
)
from AnyQt.QtGui import QColor, QBrush, QPen, QLinearGradient, QFont
from AnyQt.QtCore import Qt, QPointF, QSizeF, QPoint, QSize, QRect

from Orange.widgets.utils.colorpalettes import ContinuousPalette


class Anchorable(QGraphicsWidget):
    """Anchorable base class.

    Subclassing the `Anchorable` class will anchor the given
    `QGraphicsWidget` to a position on the viewport. This does require you to
    use the `AnchorableGraphicsView` class, it is made to be composable, so
    that should not be a problem.

    Notes
    -----
    .. note:: Subclassing this class will not make your widget movable, you
        have to do that yourself. If you do make your widget movable, this will
        handle any further positioning when the widget is moved.

    """

    __corners = ['topLeft', 'topRight', 'bottomLeft', 'bottomRight']
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = __corners

    def __init__(self, parent=None, corner='bottomRight', offset=(10, 10)):
        super().__init__(parent)

        self.__corner_str = corner if corner in self.__corners else None
        # The flag indicates whether or not the item has been drawn on yet.
        # This is useful for determining the initial offset, due to the fact
        # that dimensions are available in the resize event, which can occur
        # multiple times.
        self.__has_been_drawn = False

        if isinstance(offset, tuple) or isinstance(offset, list):
            assert len(offset) == 2
            self.__offset = QPoint(*offset)
        elif isinstance(offset, QPoint):
            self.__offset = offset

    def moveEvent(self, event):
        super().moveEvent(event)
        # This check is needed because simply resizing the window will cause
        # the item to move and trigger a `moveEvent` therefore we need to check
        # that the movement was done intentionally by the user using the mouse
        if QApplication.mouseButtons() == Qt.LeftButton:
            self.recalculate_offset()

    def resizeEvent(self, event):
        # When the item is first shown, we need to update its position
        super().resizeEvent(event)
        if not self.__has_been_drawn:
            self.__offset = self.__calculate_actual_offset(self.__offset)
            self.update_pos()
            self.__has_been_drawn = True

    def showEvent(self, event):
        # When the item is first shown, we need to update its position
        super().showEvent(event)
        self.update_pos()

    def recalculate_offset(self):
        """This is called whenever the item is being moved and needs to
        recalculate its offset."""
        view = self.__get_view()
        # Get the view box and position of legend relative to the view,
        # not the scene
        pos = view.mapFromScene(self.pos())
        view_box = self.__usable_viewbox()

        self.__corner_str = self.__get_closest_corner()
        viewbox_corner = getattr(view_box, self.__corner_str)()

        self.__offset = viewbox_corner - pos

    def update_pos(self):
        """Update the widget position relative to the viewport.

        This is called whenever something happened with the view that caused
        this item to move from its anchored position, so we have to adjust the
        position to maintain the effect of being anchored."""
        view = self.__get_view()
        if self.__corner_str and view is not None:
            box = self.__usable_viewbox()
            corner = getattr(box, self.__corner_str)()
            new_pos = corner - self.__offset
            self.setPos(view.mapToScene(new_pos))

    def __calculate_actual_offset(self, offset):
        """Take the offset specified in the constructor and calculate the
        actual offset from the top left corner of the item so positioning can
        be done correctly."""
        off_x, off_y = offset.x(), offset.y()
        width = self.boundingRect().width()
        height = self.boundingRect().height()

        if self.__corner_str == self.TOP_LEFT:
            return QPoint(-off_x, -off_y)
        elif self.__corner_str == self.TOP_RIGHT:
            return QPoint(off_x + width, -off_y)
        elif self.__corner_str == self.BOTTOM_RIGHT:
            return QPoint(off_x + width, off_y + height)
        elif self.__corner_str == self.BOTTOM_LEFT:
            return QPoint(-off_x, off_y + height)

    def __get_closest_corner(self):
        view = self.__get_view()
        # Get the view box and position of legend relative to the view,
        # not the scene
        pos = view.mapFromScene(self.pos())
        legend_box = QRect(pos, self.size().toSize())
        view_box = QRect(QPoint(0, 0), view.size())

        def distance(pt1, pt2):
            # 2d euclidean distance
            return np.sqrt((pt1.x() - pt2.x()) ** 2 + (pt1.y() - pt2.y()) ** 2)

        distances = [
            (distance(getattr(view_box, corner)(),
                      getattr(legend_box, corner)()), corner)
            for corner in self.__corners
        ]
        _, corner = min(distances)
        return corner

    def __get_own_corner(self):
        view = self.__get_view()
        pos = view.mapFromScene(self.pos())
        legend_box = QRect(pos, self.size().toSize())
        return getattr(legend_box, self.__corner_str)()

    def __get_view(self):
        if self.scene() is not None:
            view, = self.scene().views()
            return view
        else:
            return None

    def __usable_viewbox(self):
        view = self.__get_view()

        if view.horizontalScrollBar().isVisible():
            height = view.horizontalScrollBar().size().height()
        else:
            height = 0

        if view.verticalScrollBar().isVisible():
            width = view.verticalScrollBar().size().width()
        else:
            width = 0

        size = view.size() - QSize(width, height)
        return QRect(QPoint(0, 0), size)


class AnchorableGraphicsView(QGraphicsView):
    """Subclass when wanting to use Anchorable items in your view."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Handle scroll bar hiding or showing
        self.horizontalScrollBar().valueChanged.connect(
            self.update_anchored_items)
        self.verticalScrollBar().valueChanged.connect(
            self.update_anchored_items)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_anchored_items()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.update_anchored_items()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.update_anchored_items()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.update_anchored_items()

    def update_anchored_items(self):
        """Update all the items that subclass the `Anchorable` class."""
        for item in self.__anchorable_items():
            item.update_pos()

    def __anchorable_items(self):
        return [i for i in self.scene().items() if isinstance(i, Anchorable)]


class ColorIndicator(QGraphicsWidget):
    """Base class for an item indicator.

    Usually the little square or circle in the legend in front of the text."""
    pass


class LegendItemSquare(ColorIndicator):
    """Legend square item.

    The legend square item is a small colored square image that can be plugged
    into the legend in front of the text object.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    color : QColor
        The color of the square.
    parent : QGraphicsItem

    See Also
    --------
    LegendItemCircle

    """

    SIZE = QSizeF(12, 12)
    _size_hint = SIZE

    def __init__(self, color, parent):
        super().__init__(parent)

        height, width = self.SIZE.height(), self.SIZE.width()
        self.__square = QGraphicsRectItem(0, 0, height, width)
        self.__square.setBrush(QBrush(color))
        self.__square.setPen(QPen(QColor(0, 0, 0, 0)))
        self.__square.setParentItem(self)
        self._size_hint = QSizeF(self.__square.boundingRect().size())

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self._size_hint


class LegendItemCircle(ColorIndicator):
    """Legend circle item.

    The legend circle item is a small colored circle image that can be plugged
    into the legend in front of the text object.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    color : QColor
        The color of the square.
    parent : QGraphicsItem

    See Also
    --------
    LegendItemSquare

    """

    SIZE = QSizeF(12, 12)
    _size_hint = SIZE

    def __init__(self, color, parent):
        super().__init__(parent)

        height, width = self.SIZE.height(), self.SIZE.width()
        self.__circle = QGraphicsEllipseItem(0, 0, height, width)
        self.__circle.setBrush(QBrush(color))
        self.__circle.setPen(QPen(QColor(0, 0, 0, 0)))
        self.__circle.setParentItem(self)
        self._size_hint = QSizeF(self.__circle.boundingRect().size())

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self._size_hint


class LegendItemTitle(QGraphicsWidget):
    """Legend item title - the text displayed in the legend.

    This should only really be used in conjunction with ˙LegendItem˙.

    Parameters
    ----------
    text : str
    parent : QGraphicsItem
    font : QFont
        This

    """
    _size_hint = QSizeF(100, 10)

    def __init__(self, text, parent, font):
        super().__init__(parent)

        self.__text = QGraphicsTextItem(text.title())
        self.__text.setParentItem(self)
        self.__text.setFont(font)
        self._size_hint = QSizeF(self.__text.boundingRect().size())

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self._size_hint


class LegendItem(QGraphicsLinearLayout):
    """Legend item - one entry in the legend.

    This represents one entry in the legend i.e. a color indicator and the text
    beside it.

    Parameters
    ----------
    color : QColor
        The color that the entry will represent.
    title : str
        The text that will be displayed for the color.
    parent : QGraphicsItem
    color_indicator_cls : ColorIndicator
        The type of `ColorIndicator` that will be used for the color.
    font : QFont, optional

    """

    def __init__(self, color, title, parent, color_indicator_cls, font=None):
        super().__init__()

        self.__parent = parent
        self.__color_indicator = color_indicator_cls(color, parent)
        self.__title_label = LegendItemTitle(title, parent, font=font)

        self.addItem(self.__color_indicator)
        self.addItem(self.__title_label)

        # Make sure items are aligned properly, since the color box and text
        # won't be the same height.
        self.setAlignment(self.__color_indicator, Qt.AlignCenter)
        self.setAlignment(self.__title_label, Qt.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(5)


class LegendGradient(QGraphicsWidget):
    """Gradient widget.

    A gradient square bar that can be used to display continuous values.

    Parameters
    ----------
    palette : iterable[QColor]
    parent : QGraphicsWidget
    orientation : Qt.Orientation

    Notes
    -----
    .. note:: While the gradient does support any number of colors, any more
        than 3 is not very readable. This should not be a problem, since Orange
        only implements 2 or 3 colors.

    """

    # Default sizes (assume gradient is vertical by default)
    GRADIENT_WIDTH = 20
    GRADIENT_HEIGHT = 150
    _size_hint = QSizeF(GRADIENT_WIDTH, GRADIENT_HEIGHT)

    def __init__(self, palette, parent, orientation):
        super().__init__(parent)

        self.__gradient = QLinearGradient()
        num_colors = len(palette)
        for idx, stop in enumerate(palette):
            self.__gradient.setColorAt(idx * (1. / (num_colors - 1)), stop)

        # We need to tell the gradient where it's start and stop points are
        self.__gradient.setStart(QPointF(0, 0))
        if orientation == Qt.Vertical:
            final_stop = QPointF(0, self.GRADIENT_HEIGHT)
        else:
            final_stop = QPointF(self.GRADIENT_HEIGHT, 0)
        self.__gradient.setFinalStop(final_stop)

        # Get the appropriate rectangle dimensions based on orientation
        if orientation == Qt.Vertical:
            width, height = self.GRADIENT_WIDTH, self.GRADIENT_HEIGHT
        elif orientation == Qt.Horizontal:
            width, height = self.GRADIENT_HEIGHT, self.GRADIENT_WIDTH

        self.__rect_item = QGraphicsRectItem(0, 0, width, height, self)
        self.__rect_item.setPen(QPen(QColor(0, 0, 0, 0)))
        self.__rect_item.setBrush(QBrush(self.__gradient))
        self._size_hint = QSizeF(self.__rect_item.boundingRect().size())

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self._size_hint


class ColorStripItem(QGraphicsWidget):
    def __init__(self, palette, parent, orientation):
        super().__init__(parent)
        self.__strip = palette.color_strip(150, 13, orientation)

    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.__strip)

    def sizeHint(self, *_):
        return QSizeF(self.__strip.width(), self.__strip.height())


class ContinuousLegendItem(QGraphicsLinearLayout):
    """Continuous legend item.

    Contains a gradient bar with the color ranges, as well as two labels - one
    on each side of the gradient bar.

    Parameters
    ----------
    palette : iterable[QColor]
    values : iterable[float...]
        The number of values must match the number of colors in passed in the
        color palette.
    parent : QGraphicsWidget
    font : QFont
    orientation : Qt.Orientation

    """

    def __init__(self, palette, values, parent, font=None,
                 orientation=Qt.Vertical):
        if orientation == Qt.Vertical:
            super().__init__(Qt.Horizontal)
        else:
            super().__init__(Qt.Vertical)

        self.__parent = parent
        self.__palette = palette
        self.__values = values

        if isinstance(palette, ContinuousPalette):
            self.__gradient = ColorStripItem(palette, parent, orientation)
        else:
            self.__gradient = LegendGradient(palette, parent, orientation)
        self.__labels_layout = QGraphicsLinearLayout(orientation)

        str_vals = self._format_values(values)

        self.__start_label = LegendItemTitle(str_vals[0], parent, font=font)
        self.__end_label = LegendItemTitle(str_vals[1], parent, font=font)
        self.__labels_layout.addItem(self.__start_label)
        self.__labels_layout.addStretch(1)
        self.__labels_layout.addItem(self.__end_label)

        # Gradient should be to the left, then labels on the right if vertical
        if orientation == Qt.Vertical:
            self.addItem(self.__gradient)
            self.addItem(self.__labels_layout)
        # Gradient should be on the bottom, labels on top if horizontal
        elif orientation == Qt.Horizontal:
            self.addItem(self.__labels_layout)
            self.addItem(self.__gradient)

    @staticmethod
    def _format_values(values):
        """Get the formatted values to output."""
        return ['{:.3f}'.format(v) for v in values]


class Legend(Anchorable):
    """Base legend class.

    This class provides common attributes for any legend subclasses:
      - Behaviour on `QGraphicsScene`
      - Appearance of legend

    Parameters
    ----------
    parent : QGraphicsItem, optional
    orientation : Qt.Orientation, optional
        The default orientation is vertical
    domain : Orange.data.domain.Domain, optional
        This field is left optional as in some cases, we may want to simply
        pass in a list that represents the legend.
    items : Iterable[QColor, str]
    bg_color : QColor, optional
    font : QFont, optional
    color_indicator_cls : ColorIndicator
        The color indicator class that will be used to render the indicators.

    See Also
    --------
    OWDiscreteLegend
    OWContinuousLegend
    OWContinuousLegend

    Notes
    -----
    .. warning:: If the domain parameter is supplied, the items parameter will
        be ignored.

    """

    def __init__(self, parent=None, orientation=Qt.Vertical, domain=None,
                 items=None, bg_color=QColor(232, 232, 232, 196),
                 font=None, color_indicator_cls=LegendItemSquare, **kwargs):
        super().__init__(parent, **kwargs)

        self._layout = None
        self.orientation = orientation
        self.bg_color = QBrush(bg_color)
        self.color_indicator_cls = color_indicator_cls

        # Set default font if none is given
        if font is None:
            self.font = QFont()
            self.font.setPointSize(10)
        else:
            self.font = font

        self.setFlags(QGraphicsWidget.ItemIsMovable |
                      QGraphicsItem.ItemIgnoresTransformations)

        self._setup_layout()
        if domain is not None:
            self.set_domain(domain)
        elif items is not None:
            self.set_items(items)

    def _clear_layout(self):
        self._layout = None
        for child in self.children():
            child.setParent(None)

    def _setup_layout(self):
        self._clear_layout()

        self._layout = QGraphicsLinearLayout(self.orientation)
        self._layout.setContentsMargins(10, 5, 10, 5)
        # If horizontal, there needs to be horizontal space between the items
        if self.orientation == Qt.Horizontal:
            self._layout.setSpacing(10)
        # If vertical spacing, vertical space is provided by child layouts
        else:
            self._layout.setSpacing(0)
        self.setLayout(self._layout)

    def set_domain(self, domain):
        """Handle receiving the domain object.

        Parameters
        ----------
        domain : Orange.data.domain.Domain

        Returns
        -------

        Raises
        ------
        AttributeError
            If the domain does not contain the correct type of class variable.

        """
        raise NotImplementedError()

    def set_items(self, values):
        """Handle receiving an array of items.

        Parameters
        ----------
        values : iterable[object, QColor]

        Returns
        -------

        """
        raise NotImplementedError()

    @staticmethod
    def _convert_to_color(obj):
        if isinstance(obj, QColor):
            return obj
        elif isinstance(obj, tuple) or isinstance(obj, list) \
                or isinstance(obj, np.ndarray):
            assert len(obj) in (3, 4)
            return QColor(*obj)
        else:
            return QColor(obj)

    def setVisible(self, is_visible):
        """Only display the legend if it contains any items."""
        return super().setVisible(is_visible and len(self._layout) > 0)

    def paint(self, painter, options, widget=None):
        painter.save()
        pen = QPen(QColor(196, 197, 193, 200), 1)
        brush = QBrush(QColor(self.bg_color))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(self.contentsRect())
        painter.restore()


class OWDiscreteLegend(Legend):
    """Discrete legend.

    See Also
    --------
    Legend
    OWContinuousLegend

    """

    def set_domain(self, domain):
        class_var = domain.class_var

        if not class_var.is_discrete:
            raise AttributeError('[OWDiscreteLegend] The class var provided '
                                 'was not discrete.')

        self.set_items(zip(class_var.values, list(class_var.colors)))

    def set_items(self, values):
        for class_name, color in values:
            legend_item = LegendItem(
                color=self._convert_to_color(color),
                title=class_name,
                parent=self,
                color_indicator_cls=self.color_indicator_cls,
                font=self.font
            )
            self._layout.addItem(legend_item)


class OWContinuousLegend(Legend):
    """Continuous legend.

    See Also
    --------
    Legend
    OWDiscreteLegend

    """

    def __init__(self, *args, **kwargs):
        # Variables used in the `set_` methods must be set before calling super
        self.__range = kwargs.get('range', ())

        super().__init__(*args, **kwargs)

        self._layout.setContentsMargins(10, 10, 10, 10)

    def set_domain(self, domain):
        class_var = domain.class_var

        if not class_var.is_continuous:
            raise AttributeError('[OWContinuousLegend] The class var provided '
                                 'was not continuous.')

        # The first and last values must represent the range, the rest should
        # be dummy variables, as they are not shown anywhere
        values = self.__range
        self.set_items((values, class_var.palette))

    def set_items(self, values):
        vals, palette = values
        if self.orientation == Qt.Vertical:
            vals = list(reversed(vals))
        self._layout.addItem(ContinuousLegendItem(
            palette=palette,
            values=vals,
            parent=self,
            font=self.font,
            orientation=self.orientation
        ))


class OWBinnedContinuousLegend(Legend):
    """Binned continuous legend in case you don't like gradients.

    This is not implemented yet, but in case it ever needs to be, the stub is
    available.

    See Also
    --------
    Legend
    OWDiscreteLegend
    OWContinuousLegend

    """

    def set_domain(self, domain):
        pass

    def set_items(self, values):
        pass
