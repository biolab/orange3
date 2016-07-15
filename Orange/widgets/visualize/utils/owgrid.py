"""Grid widget.

Positions items into a grid. This has been tested with widgets that have their
`boundingBox` and `sizeHint` methods properly defined.

"""
from itertools import zip_longest

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class GridItem(QtGui.QGraphicsWidget):
    """The base class for grid items, takes care of positioning in grid.

    Parameters
    ----------
    widget : QtGui.QGraphicsWidget
    parent : QtGui.QGraphicsWidget

    See Also
    --------
    OWGrid
    SelectableGridItem
    ZoomableGridItem

    """

    def __init__(self, widget, parent=None, **_):
        super().__init__(parent)
        # For some reason, the super constructor is not setting the parent
        self.setParent(parent)

        self.widget = widget
        if hasattr(self.widget, 'setParent'):
            self.widget.setParentItem(self)
            self.widget.setParent(self)

        # Move the child widget to (0, 0) so that bounding rects match up
        # This is needed because the bounding rect is caluclated with the size
        # hint from (0, 0), regardless of any method override
        rect = self.widget.boundingRect()
        self.widget.moveBy(-rect.topLeft().x(), -rect.topLeft().y())

    def boundingRect(self):
        return QtCore.QRectF(QtCore.QPointF(0, 0),
                             self.widget.boundingRectoundingRect().size())

    def sizeHint(self, size_hint, size_constraint=None, **kwargs):
        return self.widget.sizeHint(size_hint, size_constraint, **kwargs)


class SelectableGridItem(GridItem):
    """Makes a grid item selectable.

    Parameters
    ----------
    widget : QtGui.QGraphicsWidget
    parent : QtGui.QgraphicsWidget

    See Also
    --------
    OWGrid
    GridItem
    ZoomableGridItem

    """

    def __init__(self, widget, parent=None, **kwargs):
        super().__init__(widget, parent, **kwargs)

        self.setFlags(QtGui.QGraphicsWidget.ItemIsSelectable)

    def paint(self, painter, options, widget=None):
        super().paint(painter, options, widget)
        rect = self.boundingRect()
        painter.save()
        if self.isSelected():
            painter.setPen(QtGui.QPen(QtGui.QColor(125, 162, 206, 192)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(QtCore.QRectF(
                rect.topLeft(), self.geometry().size()), 3, 3)
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor('#ebebeb')))
            painter.drawRoundedRect(QtCore.QRectF(
                rect.topLeft(), self.geometry().size()), 3, 3)
        painter.restore()


class ZoomableGridItem(GridItem):
    """Makes a grid item "zoomable" through the `set_max_size` method.

    "Zoomable" here means there is a `Zoom` slider through which the grid items
    can be made larger and smaller in the grid.

    Notes
    -----
    .. note:: This grid item will override any bounding box or size hint
        defined in the class hierarchy with its own.
    .. note:: This makes the grid item square.

    Parameters
    ----------
    widget : QtGui.QGraphicsWidget
    parent : QtGui.QGraphicsWidget
    max_size : int
        The maximum size of the grid item.

    See Also
    --------
    OWGrid
    GridItem
    SelectableGridItem

    """

    def __init__(self, widget, parent=None, max_size=150, **kwargs):
        self._max_size = QtCore.QSizeF(max_size, max_size)
        # We store the offsets from the top left corner to move widget properly
        self.__offset_x = self.__offset_y = 0

        super().__init__(widget, parent, **kwargs)

        self._resize_widget()

    def set_max_size(self, max_size):
        self.widget.resetTransform()
        self._max_size = QtCore.QSizeF(max_size, max_size)
        self._resize_widget()

    def _resize_widget(self):
        w = self.widget
        own_hint = self.sizeHint(Qt.PreferredSize)

        scale_w = own_hint.width() / w.boundingRect().width()
        scale_h = own_hint.height() / w.boundingRect().height()
        scale = scale_w if scale_w < scale_h else scale_h

        # Move the widget back to origin then perfom transformations
        self.widget.moveBy(-self.__offset_x, -self.__offset_y)
        # Move the tranform origin to top left, so it stays in place when
        # scaling
        w.setTransformOriginPoint(w.boundingRect().topLeft())
        w.setScale(scale)
        # Then, move the scaled widget to the center of the bounding box
        own_rect = self.boundingRect()
        self.__offset_x = (own_rect.width() - w.boundingRect().width() *
                           scale) / 2
        self.__offset_y = (own_rect.height() - w.boundingRect().height() *
                           scale) / 2
        self.widget.moveBy(self.__offset_x, self.__offset_y)
        # Finally, tell the world you've changed
        self.updateGeometry()

    def boundingRect(self):
        return QtCore.QRectF(QtCore.QPointF(0, 0), self._max_size)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self._max_size


class OWGrid(QtGui.QGraphicsWidget):
    """Responsive grid layout widget.

    Manages grid items for various window sizes.

    Accepts grid items as items.

    Parameters
    ----------
    parent : QtGui.QGraphicsWidget

    Examples
    --------
    >>> grid = OWGrid()

    It's a good idea to define what you want your grid items to do. For this
    example, we will make them selectable and zoomable, so we define a class
    that inherits from both:
    >>> class MyGridItem(SelectableGridItem, ZoomableGridItem):
    >>>     pass

    We then take a list of items and wrap them into our new `MyGridItem`
    instances.
    >>> items = [QtGui.QGraphicsRectItem(0, 0, 10, 10)]
    >>> grid_items = [MyGridItem(i, grid) for i in items]

    We can then set the items to be displayed
    >>> grid.set_items(grid_items)

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QtGui.QSizePolicy.Maximum,
                           QtGui.QSizePolicy.Maximum)
        self.setContentsMargins(10, 10, 10, 10)

        self.__layout = QtGui.QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        self.setLayout(self.__layout)

    def set_items(self, items):
        for i, item in enumerate(items):
            # Place the items in some arbitrary order - they will be rearranged
            # before user sees this ordering
            self.__layout.addItem(item, i, 0)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.reflow(self.size().width())

    def reflow(self, width):
        """Recalculate the layout and reposition the elements so they fit.

        Parameters
        ----------
        width : int
            The maximum width of the grid.

        Returns
        -------

        """
        # When setting the geometry when opened, the layout doesn't yet exist
        if self.layout() is None:
            return

        grid = self.__layout

        left, right, *_ = self.getContentsMargins()
        width -= left + right

        # Get size hints with 32 as the minimum size for each cell
        widths = [max(64, h.width()) for h in self._hints(Qt.PreferredSize)]
        ncol = self._fit_n_cols(widths, grid.horizontalSpacing(), width)

        # The number of columns is already optimal
        if ncol == grid.columnCount():
            return

        # remove all items from the layout, then re-add them back in updated
        # positions
        items = self._items()

        for item in items:
            grid.removeItem(item)

        for i, item in enumerate(items):
            grid.addItem(item, i // ncol, i % ncol)
            grid.setAlignment(item, Qt.AlignCenter)

    def clear(self):
        for item in self._items():
            self.__layout.removeItem(item)
            item.setParent(None)

    @staticmethod
    def _fit_n_cols(widths, spacing, constraint):

        def sliced(seq, n_col):
            """Slice the widths into n lists that contain their respective
            widths. E.g. [5, 5, 5], 2 => [[5, 5], [5]]"""
            return [seq[i:i + n_col] for i in range(0, len(seq), n_col)]

        def flow_width(widths, spacing, ncol):
            w = sliced(widths, ncol)
            col_widths = map(max, zip_longest(*w, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths) + 1):
            width = flow_width(widths, spacing, ncol)
            if width <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best

    def _items(self):
        if not self.__layout:
            return []
        return [self.__layout.itemAt(i) for i in range(self.__layout.count())]

    def _hints(self, which):
        return [item.sizeHint(which) for item in self._items()]
