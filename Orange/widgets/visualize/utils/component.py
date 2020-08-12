"""Common gui.OWComponent components."""

from AnyQt.QtCore import Qt, QRectF
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import QGraphicsEllipseItem

import pyqtgraph as pg
from Orange.widgets.visualize.owscatterplotgraph import (
    OWScatterPlotBase, ScatterBaseParameterSetter
)
from Orange.widgets.visualize.utils.customizableplot import Updater
from Orange.widgets.visualize.utils.plotutils import (
    MouseEventDelegate, DraggableItemsViewBox
)

class AnchorParameterSetter(ScatterBaseParameterSetter):
    ANCHOR_LABEL = "Anchor"

    def __init__(self, master):
        super().__init__(master)
        self.anchor_font = QFont()

    def update_setters(self):
        super().update_setters()
        self.initial_settings[self.LABELS_BOX].update({
            self.ANCHOR_LABEL: self.FONT_SETTING
        })

        def update_anchors(**settings):
            self.anchor_font = Updater.change_font(self.anchor_font, settings)
            self.master.update_anchors()

        self._setters[self.LABELS_BOX][self.ANCHOR_LABEL] = update_anchors


class OWGraphWithAnchors(OWScatterPlotBase):
    """
    Graph for projections in which dimensions can be manually moved

    Class is used as a graph base class for OWFreeViz and OWRadviz."""
    DISTANCE_DIFF = 0.08

    def __init__(self, scatter_widget, parent, view_box=DraggableItemsViewBox):
        super().__init__(scatter_widget, parent, view_box)
        self.anchor_items = None
        self.circle_item = None
        self.indicator_item = None
        self._tooltip_delegate = MouseEventDelegate(self.help_event,
                                                    self.show_indicator_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.parameter_setter = AnchorParameterSetter(self)

    def clear(self):
        super().clear()
        self.anchor_items = None
        self.circle_item = None
        self.indicator_item = None

    def update_coordinates(self):
        super().update_coordinates()
        if self.scatterplot_item is not None:
            self.update_anchors()
            self.update_circle()
            self.set_view_box_range()
            self.view_box.setAspectLocked(True, 1)

    def update_anchors(self):
        raise NotImplementedError

    def update_circle(self):
        if self.scatterplot_item is not None and not self.circle_item:
            self.circle_item = QGraphicsEllipseItem()
            self.circle_item.setRect(QRectF(-1, -1, 2, 2))
            self.circle_item.setPen(pg.mkPen(QColor(0, 0, 0), width=2))
            self.plot_widget.addItem(self.circle_item)

    def reset_button_clicked(self):
        self.set_view_box_range()

    def set_view_box_range(self):
        raise NotImplementedError

    def closest_draggable_item(self, pos):
        return None

    def show_indicator(self, anchor_idx):
        self._update_indicator_item(anchor_idx)

    def show_indicator_event(self, ev):
        scene = self.plot_widget.scene()
        if self.scatterplot_item is None or scene.drag_tooltip.isVisible():
            return False
        if self.view_box.mouse_state == 1:
            return True

        pos = self.scatterplot_item.mapFromScene(ev.scenePos())
        anchor_idx = self.closest_draggable_item(pos)
        if anchor_idx is not None:
            self._update_indicator_item(anchor_idx)
            if self.view_box.mouse_state == 0:
                self.view_box.setCursor(Qt.OpenHandCursor)
        else:
            if self.indicator_item is not None:
                self.plot_widget.removeItem(self.indicator_item)
                self.indicator_item = None
            self.view_box.setCursor(Qt.ArrowCursor)
        return True

    def _update_indicator_item(self, anchor_idx):
        if self.indicator_item is not None:
            self.plot_widget.removeItem(self.indicator_item)
        self._add_indicator_item(anchor_idx)

    def _add_indicator_item(self, anchor_idx):
        pass
