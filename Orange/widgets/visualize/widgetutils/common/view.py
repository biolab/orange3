from itertools import repeat

import numpy as np
from PyQt4 import QtGui
from PyQt4.QtCore import Qt


class ZoomableGraphicsView(QtGui.QGraphicsView):
    """Zoomable graphics view.

    Composable graphics view that adds zoom functionality.

    It also handles automatic resizing of content whenever the window is
    resized.

    Right click will reset the zoom to a factor where the entire scene is
    visible.

    Parameters
    ----------
    scene : QtGui.QGraphicsScene
    padding : int or tuple, optional
        Specify the padding around the drawn widgets. Can be an int, or tuple,
        the tuple can contain either 2 or 4 elements.

    Notes
    -----
      - This view will consume wheel scrolling and right mouse click events.

    """

    def __init__(self, scene, padding=(0, 0), **kwargs):
        self.zoom = 1
        self.scale_factor = 1 / 16
        # zoomout limit prevents the zoom factor to become negative, which
        # results in the canvas being flipped over the x axis
        self.__zoomout_limit_reached = False
        # Does the view need to recalculate the initial scale factor
        self.__needs_to_recalculate_initial = True
        self.__initial_zoom = -1

        self.__central_widget = None
        self.__set_padding(padding)

        super().__init__(scene, **kwargs)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.__needs_to_recalculate_initial = True

    def wheelEvent(self, ev):
        self.__handle_zoom(ev.delta())
        super().wheelEvent(ev)

    def mousePressEvent(self, ev):
        # right click resets the zoom factor
        if ev.button() == Qt.RightButton:
            self.reset_zoom()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Plus:
            self.__handle_zoom(1)
        elif ev.key() == Qt.Key_Minus:
            self.__handle_zoom(-1)

        super().keyPressEvent(ev)

    def __set_padding(self, padding):
        # Allow for multiple formats of padding for convenience
        if isinstance(padding, int):
            padding = list(repeat(padding, 4))
        elif isinstance(padding, list) or isinstance(padding, tuple):
            if len(padding) == 2:
                padding = (*padding, *padding)
        else:
            padding = 0, 0, 0, 0

        l, t, r, b = padding
        self.__padding = -l, -t, r, b

    def __handle_zoom(self, direction):
        """Handle zoom event, direction is positive if zooming in, otherwise
        negative."""
        if self.__zooming_in(direction):
            self.__reset_zoomout_limit()
        if self.__zoomout_limit_reached and self.__zooming_out(direction):
            return

        self.zoom += np.sign(direction) * self.scale_factor
        if self.zoom <= 0:
            self.__zoomout_limit_reached = True
            self.zoom += self.scale_factor
        else:
            self.setTransformationAnchor(self.AnchorUnderMouse)
            self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))

    @staticmethod
    def __zooming_out(direction):
        return direction < 0

    def __zooming_in(self, ev):
        return not self.__zooming_out(ev)

    def __reset_zoomout_limit(self):
        self.__zoomout_limit_reached = False

    def set_central_widget(self, widget):
        self.__central_widget = widget

    def central_widget_rect(self):
        """Get the bounding box of the central widget.

        If a central widget and padding are set, this method calculates the
        rect containing both of them. This is useful because if the padding was
        added directly onto the widget, the padding would be rescaled as well.

        If the central widget is not set, return the scene rect instead.

        Returns
        -------
        QtCore.QRectF

        """
        if self.__central_widget is None:
            return self.scene().itemsBoundingRect().adjusted(*self.__padding)
        return self.__central_widget.boundingRect().adjusted(*self.__padding)

    def recalculate_and_fit(self):
        """Recalculate the optimal zoom and fits the content into view.

        Should be called if the scene contents change, so that the optimal zoom
        can be recalculated.

        Returns
        -------

        """
        if self.__central_widget is not None:
            self.fitInView(self.central_widget_rect(), Qt.KeepAspectRatio)
        else:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

        self.__initial_zoom = self.matrix().m11()
        self.zoom = self.__initial_zoom

    def reset_zoom(self):
        """Reset the zoom to the optimal factor."""
        self.zoom = self.__initial_zoom
        self.__zoomout_limit_reached = False

        if self.__needs_to_recalculate_initial:
            self.recalculate_and_fit()
        else:
            self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))


class PannableGraphicsView(QtGui.QGraphicsView):
    """Pannable graphics view.

    Enables panning the graphics view.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

    def enterEvent(self, ev):
        self.viewport().setCursor(Qt.ArrowCursor)
        super().enterEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.viewport().setCursor(Qt.ArrowCursor)


class PreventDefaultWheelEvent(QtGui.QGraphicsView):
    def wheelEvent(self, ev):
        ev.accept()
