"""Common gui.OWComponent components."""
from AnyQt.QtCore import Qt

from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.plotutils import (
    MouseEventDelegate, VizInteractiveViewBox
)


class OWVizGraph(OWScatterPlotBase):
    """Class is used as a graph base class for OWFreeViz and OWRadviz."""
    DISTANCE_DIFF = 0.08

    def __init__(self, scatter_widget, parent, view_box=VizInteractiveViewBox):
        super().__init__(scatter_widget, parent, view_box)
        self._attributes = ()
        self._points = None
        self._point_items = None
        self._circle_item = None
        self._indicator_item = None
        self._tooltip_delegate = MouseEventDelegate(self.help_event,
                                                    self.show_indicator_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.view_box.sigResized.connect(self.update_density)

    def set_attributes(self, attributes):
        self._attributes = attributes

    def set_point(self, i, x, y):
        self._points[i][0] = x
        self._points[i][1] = y

    def set_points(self, points):
        self._points = points

    def get_points(self):
        return self._points

    def update_coordinates(self):
        super().update_coordinates()
        self.update_items()
        self.set_view_box_range()
        self.view_box.setAspectLocked(True, 1)

    def reset_button_clicked(self):
        self.set_view_box_range()

    def set_view_box_range(self):
        raise NotImplementedError

    def can_show_indicator(self, pos):
        raise NotImplementedError

    def show_indicator(self, point_i):
        self._update_indicator_item(point_i)

    def show_indicator_event(self, ev):
        scene = self.plot_widget.scene()
        if self.scatterplot_item is None or scene.drag_tooltip.isVisible():
            return False
        if self.view_box.mouse_state == 1:
            return True

        pos = self.scatterplot_item.mapFromScene(ev.scenePos())
        can_show, point_i = self.can_show_indicator(pos)
        if can_show:
            self._update_indicator_item(point_i)
            if self.view_box.mouse_state == 0:
                self.view_box.setCursor(Qt.OpenHandCursor)
        else:
            if self._indicator_item is not None:
                self.plot_widget.removeItem(self._indicator_item)
                self._indicator_item = None
            self.view_box.setCursor(Qt.ArrowCursor)
        return True

    def update_items(self):
        self._update_point_items()
        self._update_circle_item()

    def _update_point_items(self):
        self._remove_point_items()
        self._add_point_items()

    def _update_circle_item(self):
        self._remove_circle_item()
        self._add_circle_item()

    def _update_indicator_item(self, point_i):
        self._remove_indicator_item()
        self._add_indicator_item(point_i)

    def _remove_point_items(self):
        if self._point_items is not None:
            self.plot_widget.removeItem(self._point_items)
            self._point_items = None

    def _remove_circle_item(self):
        if self._circle_item is not None:
            self.plot_widget.removeItem(self._circle_item)
            self._circle_item = None

    def _remove_indicator_item(self):
        if self._indicator_item is not None:
            self.plot_widget.removeItem(self._indicator_item)
            self._indicator_item = None

    def _add_point_items(self):
        raise NotImplementedError

    def _add_circle_item(self):
        raise NotImplementedError

    def _add_indicator_item(self, point_i):
        raise NotImplementedError
