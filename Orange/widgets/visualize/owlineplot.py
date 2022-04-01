from typing import List
from xml.sax.saxutils import escape

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QSize, QLineF, pyqtSignal as Signal
from AnyQt.QtGui import QPainter, QPen, QColor
from AnyQt.QtWidgets import QApplication, QGraphicsLineItem, QSizePolicy

import pyqtgraph as pg
from pyqtgraph.functions import mkPen
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from orangewidget.utils.listview import ListViewSearch
from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.data import Table, DiscreteVariable
from Orange.data.sql.table import SqlTable
from Orange.statistics.util import nanmean, nanmin, nanmax, nanstd
from Orange.widgets import gui, report
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler
)
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owdistributions import LegendItem
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.widget import OWWidget, Input, Output, Msg


def ccw(a, b, c):
    """
    Checks whether three points are listed in a counterclockwise order.
    """
    ax, ay = (a[:, 0], a[:, 1]) if a.ndim == 2 else (a[0], a[1])
    bx, by = (b[:, 0], b[:, 1]) if b.ndim == 2 else (b[0], b[1])
    cx, cy = (c[:, 0], c[:, 1]) if c.ndim == 2 else (c[0], c[1])
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def intersects(a, b, c, d):
    """
    Checks whether line segment a (given points a and b) intersects with line
    segment b (given points c and d).
    """
    return np.logical_and(ccw(a, c, d) != ccw(b, c, d),
                          ccw(a, b, c) != ccw(a, b, d))


def line_intersects_profiles(p1, p2, table):
    """
    Checks if a line intersects any line segments.

    Parameters
    ----------
    p1, p2 : ndarray
        Endpoints of the line, given x coordinate as p_[0]
        and y coordinate as p_[1].
    table : ndarray
        An array of shape m x n x p; where m is number of connected points
        for a individual profile (i. e. number of features), n is number
        of instances, p is number of coordinates (x and y).

    Returns
    -------
    result : ndarray
        Array of bools with shape of number of instances in the table.
    """
    res = np.zeros(len(table[0]), dtype=bool)
    for i in range(len(table) - 1):
        res = np.logical_or(res, intersects(p1, p2, table[i], table[i + 1]))
    return res


class LinePlotStyle:
    DEFAULT_COLOR = QColor(Qt.darkGray)
    SELECTION_LINE_COLOR = QColor(Qt.black)
    SELECTION_LINE_WIDTH = 2

    UNSELECTED_LINE_WIDTH = 1
    UNSELECTED_LINE_ALPHA = 100
    UNSELECTED_LINE_ALPHA_SEL = 50  # unselected lines, when selection exists

    SELECTED_LINE_WIDTH = 3
    SELECTED_LINE_ALPHA = 170

    RANGE_ALPHA = 25
    SELECTED_RANGE_ALPHA = 50

    MEAN_WIDTH = 6
    MEAN_DARK_FACTOR = 110


class BottomAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ticks = {}

    def set_ticks(self, ticks):
        self._ticks = dict(enumerate(ticks, 1)) if ticks else {}
        if not ticks:
            self.setTicks(None)

    def tickStrings(self, values, scale, _):
        return [self._ticks.get(v * scale, "") for v in values]


class LinePlotViewBox(ViewBox):
    selection_changed = Signal(np.ndarray)

    def __init__(self):
        super().__init__(enableMenu=False)
        self._profile_items = None
        self._can_select = True
        self._graph_state = SELECT

        self.setMouseMode(self.PanMode)

        pen = mkPen(LinePlotStyle.SELECTION_LINE_COLOR,
                    width=LinePlotStyle.SELECTION_LINE_WIDTH)
        self.selection_line = QGraphicsLineItem()
        self.selection_line.setPen(pen)
        self.selection_line.setZValue(1e9)
        self.addItem(self.selection_line, ignoreBounds=True)

    def update_selection_line(self, button_down_pos, current_pos):
        p1 = self.childGroup.mapFromParent(button_down_pos)
        p2 = self.childGroup.mapFromParent(current_pos)
        self.selection_line.setLine(QLineF(p1, p2))
        self.selection_line.resetTransform()
        self.selection_line.show()

    def set_graph_state(self, state):
        self._graph_state = state

    def enable_selection(self, enable):
        self._can_select = enable

    def get_selected(self, p1, p2):
        if self._profile_items is None:
            return np.array(False)
        return line_intersects_profiles(np.array([p1.x(), p1.y()]),
                                        np.array([p2.x(), p2.y()]),
                                        self._profile_items)

    def add_profiles(self, y):
        if sp.issparse(y):
            y = y.todense()
        self._profile_items = np.array(
            [np.vstack((np.full((1, y.shape[0]), i + 1), y[:, i].flatten())).T
             for i in range(y.shape[1])])

    def remove_profiles(self):
        self._profile_items = None

    def mouseDragEvent(self, ev, axis=None):
        if self._graph_state == SELECT and axis is None and self._can_select:
            ev.accept()
            if ev.button() == Qt.LeftButton:
                self.update_selection_line(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self.selection_line.hide()
                    p1 = self.childGroup.mapFromParent(
                        ev.buttonDownPos(ev.button()))
                    p2 = self.childGroup.mapFromParent(ev.pos())
                    self.selection_changed.emit(self.get_selected(p1, p2))
        elif self._graph_state == ZOOMING or self._graph_state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.autoRange()
            self.enableAutoRange()
        else:
            ev.accept()
            self.selection_changed.emit(np.array(False))

    def reset(self):
        self._profile_items = None
        self._can_select = True
        self._graph_state = SELECT


class ParameterSetter(CommonParameterSetter):
    MEAN_LABEL = "Mean"
    LINE_LABEL = "Lines"
    MISSING_LINE_LABEL = "Lines (missing value)"
    SEL_LINE_LABEL = "Selected lines"
    SEL_MISSING_LINE_LABEL = "Selected lines (missing value)"
    RANGE_LABEL = "Range"
    SEL_RANGE_LABEL = "Selected range"

    def __init__(self, master):
        super().__init__()
        self.master = master

    def update_setters(self):
        self.mean_settings = {
            Updater.WIDTH_LABEL: LinePlotStyle.MEAN_WIDTH,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
        }
        self.line_settings = {
            Updater.WIDTH_LABEL: LinePlotStyle.UNSELECTED_LINE_WIDTH,
            Updater.ALPHA_LABEL: LinePlotStyle.UNSELECTED_LINE_ALPHA,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
            Updater.ANTIALIAS_LABEL: True,
        }
        self.missing_line_settings = {
            Updater.WIDTH_LABEL: LinePlotStyle.UNSELECTED_LINE_WIDTH,
            Updater.ALPHA_LABEL: LinePlotStyle.UNSELECTED_LINE_ALPHA,
            Updater.STYLE_LABEL: "Dash line",
            Updater.ANTIALIAS_LABEL: True,
        }
        self.sel_line_settings = {
            Updater.WIDTH_LABEL: LinePlotStyle.SELECTED_LINE_WIDTH,
            Updater.ALPHA_LABEL: LinePlotStyle.SELECTED_LINE_ALPHA,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
            Updater.ANTIALIAS_LABEL: False,
        }
        self.sel_missing_line_settings = {
            Updater.WIDTH_LABEL: LinePlotStyle.SELECTED_LINE_WIDTH,
            Updater.ALPHA_LABEL: LinePlotStyle.SELECTED_LINE_ALPHA,
            Updater.STYLE_LABEL: "Dash line",
            Updater.ANTIALIAS_LABEL: False,
        }
        self.range_settings = {
            Updater.ALPHA_LABEL: LinePlotStyle.RANGE_ALPHA,
        }
        self.sel_range_settings = {
            Updater.ALPHA_LABEL: LinePlotStyle.SELECTED_RANGE_ALPHA,
        }

        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
                self.LEGEND_LABEL: self.FONT_SETTING,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
                self.X_AXIS_LABEL: {self.TITLE_LABEL: ("", "")},
                self.Y_AXIS_LABEL: {self.TITLE_LABEL: ("", "")},
            },
            self.PLOT_BOX: {
                self.MEAN_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15), LinePlotStyle.MEAN_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          Updater.DEFAULT_LINE_STYLE),
                },
                self.LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15),
                                          LinePlotStyle.UNSELECTED_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          Updater.DEFAULT_LINE_STYLE),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.UNSELECTED_LINE_ALPHA),
                    Updater.ANTIALIAS_LABEL: (None, True),
                },
                self.MISSING_LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15),
                                          LinePlotStyle.UNSELECTED_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          "Dash line"),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.UNSELECTED_LINE_ALPHA),
                    Updater.ANTIALIAS_LABEL: (None, True),
                },
                self.SEL_LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15),
                                          LinePlotStyle.SELECTED_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          Updater.DEFAULT_LINE_STYLE),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.SELECTED_LINE_ALPHA),
                    Updater.ANTIALIAS_LABEL: (None, False),
                },
                self.SEL_MISSING_LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15),
                                          LinePlotStyle.SELECTED_LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          "Dash line"),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.SELECTED_LINE_ALPHA),
                    Updater.ANTIALIAS_LABEL: (None, True),
                },
                self.RANGE_LABEL: {
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.RANGE_ALPHA),
                },
                self.SEL_RANGE_LABEL: {
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          LinePlotStyle.SELECTED_RANGE_ALPHA),
                },
            }
        }

        def update_mean(**settings):
            self.mean_settings.update(**settings)
            Updater.update_lines(self.mean_lines_items, **self.mean_settings)

        def update_lines(**settings):
            self.line_settings.update(**settings)
            Updater.update_lines(self.lines_items, **self.line_settings)

        def update_missing_lines(**settings):
            self.missing_line_settings.update(**settings)
            Updater.update_lines(self.missing_lines_items,
                                 **self.missing_line_settings)

        def update_sel_lines(**settings):
            self.sel_line_settings.update(**settings)
            Updater.update_lines(self.sel_lines_items, **self.sel_line_settings)

        def update_sel_missing_lines(**settings):
            self.sel_missing_line_settings.update(**settings)
            Updater.update_lines(self.sel_missing_lines_items,
                                 **self.sel_missing_line_settings)

        def _update_brush(items, **settings):
            for item in items:
                brush = item.brush()
                color = brush.color()
                color.setAlpha(settings[Updater.ALPHA_LABEL])
                brush.setColor(color)
                item.setBrush(brush)

        def update_range(**settings):
            self.range_settings.update(**settings)
            _update_brush(self.range_items, **settings)

        def update_sel_range(**settings):
            self.sel_range_settings.update(**settings)
            _update_brush(self.sel_range_items, **settings)

        self._setters[self.PLOT_BOX] = {
            self.MEAN_LABEL: update_mean,
            self.LINE_LABEL: update_lines,
            self.MISSING_LINE_LABEL: update_missing_lines,
            self.SEL_LINE_LABEL: update_sel_lines,
            self.SEL_MISSING_LINE_LABEL: update_sel_missing_lines,
            self.RANGE_LABEL: update_range,
            self.SEL_RANGE_LABEL: update_sel_range,
        }

    @property
    def title_item(self):
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self):
        return self.master.legend.items

    @property
    def mean_lines_items(self):
        return [group.mean for group in self.master.groups]

    @property
    def lines_items(self):
        return [group.profiles for group in self.master.groups]

    @property
    def missing_lines_items(self):
        return [group.missing_profiles for group in self.master.groups]

    @property
    def sel_lines_items(self):
        return [group.sel_profiles for group in self.master.groups] + \
               [group.sub_profiles for group in self.master.groups]

    @property
    def sel_missing_lines_items(self):
        return [group.sel_missing_profiles for group in self.master.groups] + \
               [group.sub_missing_profiles for group in self.master.groups]

    @property
    def range_items(self):
        return [group.range for group in self.master.groups]

    @property
    def sel_range_items(self):
        return [group.sel_range for group in self.master.groups]

    @property
    def getAxis(self):
        return self.master.getAxis


# Customizable plot widget
class LinePlotGraph(PlotWidget):
    def __init__(self, parent):
        self.groups: List[ProfileGroup] = []
        self.bottom_axis = BottomAxisItem(orientation="bottom")
        self.bottom_axis.setLabel("")
        left_axis = AxisItem(orientation="left")
        left_axis.setLabel("")
        super().__init__(parent, viewBox=LinePlotViewBox(),
                         enableMenu=False,
                         axisItems={"bottom": self.bottom_axis,
                                    "left": left_axis})
        self.view_box = self.getViewBox()
        self.selection = set()
        self.legend = self._create_legend(((1, 0), (1, 0)))
        self.getPlotItem().buttonsHidden = True
        self.setRenderHint(QPainter.Antialiasing, True)

        self.parameter_setter = ParameterSetter(self)

    def _create_legend(self, anchor):
        legend = LegendItem()
        legend.setParentItem(self.view_box)
        legend.restoreAnchor(anchor)
        legend.hide()
        return legend

    def update_legend(self, variable):
        self.legend.clear()
        self.legend.hide()
        if variable and variable.is_discrete:
            for name, color in zip(variable.values, variable.colors):
                c = QColor(*color)
                dots = pg.ScatterPlotItem(pen=c, brush=c, size=10, shape="s")
                self.legend.addItem(dots, escape(name))
            self.legend.show()
        Updater.update_legend_font(self.parameter_setter.legend_items,
                                   **self.parameter_setter.legend_settings)

    def select(self, indices):
        keys = QApplication.keyboardModifiers()
        indices = set(indices)
        if keys & Qt.ControlModifier:
            self.selection ^= indices
        elif keys & Qt.AltModifier:
            self.selection -= indices
        elif keys & Qt.ShiftModifier:
            self.selection |= indices
        else:
            self.selection = indices

    def reset(self):
        self.selection = set()
        self.view_box.reset()
        self.clear()
        self.getAxis('bottom').set_ticks(None)
        self.legend.hide()
        self.groups = []

    def select_button_clicked(self):
        self.view_box.set_graph_state(SELECT)
        self.view_box.setMouseMode(self.view_box.RectMode)

    def pan_button_clicked(self):
        self.view_box.set_graph_state(PANNING)
        self.view_box.setMouseMode(self.view_box.PanMode)

    def zoom_button_clicked(self):
        self.view_box.set_graph_state(ZOOMING)
        self.view_box.setMouseMode(self.view_box.RectMode)

    def reset_button_clicked(self):
        self.view_box.autoRange()
        self.view_box.enableAutoRange()


class ProfileGroup:
    def __init__(self, data, indices, color, graph):
        self.x_data = np.arange(1, data.X.shape[1] + 1)
        self.y_data = data.X
        self.indices = indices
        self.ids = data.ids
        self.color = color
        self.graph = graph

        self._profiles_added = False
        self._sub_profiles_added = False
        self._range_added = False
        self._mean_added = False
        self._error_bar_added = False

        self.graph_items = []
        self.__mean = nanmean(self.y_data, axis=0)
        self.__create_curves()

    def __create_curves(self):
        self.profiles = self._get_profiles_curve()
        self.missing_profiles = self._get_missing_profiles_curve()
        self.sub_profiles = self._get_sel_profiles_curve()
        self.sub_missing_profiles = self._get_sel_missing_profiles_curve()
        self.sel_profiles = self._get_sel_profiles_curve()
        self.sel_missing_profiles = self._get_sel_missing_profiles_curve()
        self.range = self._get_range_curve()
        self.sel_range = self._get_sel_range_curve()
        self.mean = self._get_mean_curve()
        self.error_bar = self._get_error_bar()
        self.graph_items = [
            self.mean, self.range, self.sel_range, self.profiles,
            self.sub_profiles, self.sel_profiles, self.error_bar,
            self.missing_profiles, self.sel_missing_profiles,
            self.sub_missing_profiles,
        ]

    def _get_profiles_curve(self):
        x, y, con = self.__get_disconnected_curve_data(self.y_data)
        pen = self.make_pen(self.color)
        curve = pg.PlotCurveItem(x=x, y=y, connect=con, pen=pen)
        Updater.update_lines([curve], **self.graph.parameter_setter.line_settings)
        return curve

    def _get_missing_profiles_curve(self):
        x, y, con = self.__get_disconnected_curve_missing_data(self.y_data)
        pen = self.make_pen(self.color)
        curve = pg.PlotCurveItem(x=x, y=y, connect=con, pen=pen)
        settings = self.graph.parameter_setter.missing_line_settings
        Updater.update_lines([curve], **settings)
        return curve

    def _get_sel_profiles_curve(self):
        curve = pg.PlotCurveItem(x=None, y=None, pen=self.make_pen(self.color))
        Updater.update_lines([curve], **self.graph.parameter_setter.sel_line_settings)
        return curve

    def _get_sel_missing_profiles_curve(self):
        curve = pg.PlotCurveItem(x=None, y=None, pen=self.make_pen(self.color))
        settings = self.graph.parameter_setter.sel_missing_line_settings
        Updater.update_lines([curve], **settings)
        return curve

    def _get_range_curve(self):
        color = QColor(self.color)
        color.setAlpha(self.graph.parameter_setter.range_settings[Updater.ALPHA_LABEL])
        bottom, top = nanmin(self.y_data, axis=0), nanmax(self.y_data, axis=0)
        return pg.FillBetweenItem(
            pg.PlotDataItem(x=self.x_data, y=bottom),
            pg.PlotDataItem(x=self.x_data, y=top), brush=color
        )

    def _get_sel_range_curve(self):
        color = QColor(self.color)
        color.setAlpha(self.graph.parameter_setter.sel_range_settings[Updater.ALPHA_LABEL])
        curve1 = curve2 = pg.PlotDataItem(x=self.x_data, y=self.__mean)
        return pg.FillBetweenItem(curve1, curve2, brush=color)

    def _get_mean_curve(self):
        pen = self.make_pen(self.color.darker(LinePlotStyle.MEAN_DARK_FACTOR))
        curve = pg.PlotCurveItem(x=self.x_data, y=self.__mean, pen=pen)
        Updater.update_lines([curve], **self.graph.parameter_setter.mean_settings)
        return curve

    def _get_error_bar(self):
        std = nanstd(self.y_data, axis=0)
        return pg.ErrorBarItem(x=self.x_data, y=self.__mean,
                               bottom=std, top=std, beam=0.01)

    def remove_items(self):
        for item in self.graph_items:
            self.graph.removeItem(item)
        self.graph_items = []

    def set_visible_profiles(self, show_profiles=True, show_range=True, **_):
        if not self._profiles_added and show_profiles:
            self._profiles_added = True
            self.graph.addItem(self.profiles)
            self.graph.addItem(self.missing_profiles)
            self.graph.addItem(self.sel_profiles)
            self.graph.addItem(self.sel_missing_profiles)
        if not self._sub_profiles_added and (show_profiles or show_range):
            self._sub_profiles_added = True
            self.graph.addItem(self.sub_profiles)
            self.graph.addItem(self.sub_missing_profiles)
        self.profiles.setVisible(show_profiles)
        self.missing_profiles.setVisible(show_profiles)
        self.sel_profiles.setVisible(show_profiles)
        self.sel_missing_profiles.setVisible(show_profiles)
        self.sub_profiles.setVisible(show_profiles or show_range)
        self.sub_missing_profiles.setVisible(show_profiles or show_range)

    def set_visible_range(self, show_profiles=True, show_range=True, **_):
        if not self._range_added and show_range:
            self._range_added = True
            self.graph.addItem(self.range)
            self.graph.addItem(self.sel_range)
        if not self._sub_profiles_added and (show_profiles or show_range):
            self._sub_profiles_added = True
            self.graph.addItem(self.sub_profiles)
        self.range.setVisible(show_range)
        self.sel_range.setVisible(show_range)
        self.sub_profiles.setVisible(show_profiles or show_range)

    def set_visible_mean(self, show_mean=True, **_):
        if not self._mean_added and show_mean:
            self._mean_added = True
            self.graph.addItem(self.mean)
        self.mean.setVisible(show_mean)

    def set_visible_error(self, show_error=True, **_):
        if not self._error_bar_added and show_error:
            self._error_bar_added = True
            self.graph.addItem(self.error_bar)
        self.error_bar.setVisible(show_error)

    def update_profiles_color(self, selection):
        color = QColor(self.color)
        alpha = self.graph.parameter_setter.line_settings[Updater.ALPHA_LABEL] \
            if not selection else LinePlotStyle.UNSELECTED_LINE_ALPHA_SEL
        color.setAlpha(alpha)
        pen = self.profiles.opts["pen"]
        pen.setColor(color)
        self.profiles.setPen(pen)

        color = QColor(self.color)
        alpha = self.graph.parameter_setter.missing_line_settings[
            Updater.ALPHA_LABEL] if not selection else \
            LinePlotStyle.UNSELECTED_LINE_ALPHA_SEL
        color.setAlpha(alpha)
        pen = self.missing_profiles.opts["pen"]
        pen.setColor(color)
        self.missing_profiles.setPen(pen)

    def update_sel_profiles(self, y_data):
        x, y, connect = self.__get_disconnected_curve_data(y_data) \
            if y_data is not None else (None, None, None)
        self.sel_profiles.setData(x=x, y=y, connect=connect)

        x, y, connect = self.__get_disconnected_curve_missing_data(y_data) \
            if y_data is not None else (None, None, None)
        self.sel_missing_profiles.setData(x=x, y=y, connect=connect)

    def update_sel_profiles_color(self, subset):
        color = QColor(Qt.black) if subset else QColor(self.color)
        color.setAlpha(self.graph.parameter_setter.sel_line_settings[Updater.ALPHA_LABEL])
        pen = self.sel_profiles.opts["pen"]
        pen.setColor(color)
        self.sel_profiles.setPen(pen)

        color = QColor(Qt.black) if subset else QColor(self.color)
        alpha = self.graph.parameter_setter.sel_missing_line_settings[
            Updater.ALPHA_LABEL]
        color.setAlpha(alpha)
        pen = self.sel_missing_profiles.opts["pen"]
        pen.setColor(color)
        self.sel_missing_profiles.setPen(pen)

    def update_sub_profiles(self, y_data):
        x, y, connect = self.__get_disconnected_curve_data(y_data) \
            if y_data is not None else (None, None, None)
        self.sub_profiles.setData(x=x, y=y, connect=connect)

        x, y, connect = self.__get_disconnected_curve_missing_data(y_data) \
            if y_data is not None else (None, None, None)
        self.sub_missing_profiles.setData(x=x, y=y, connect=connect)

    def update_sel_range(self, y_data):
        if y_data is None:
            curve1 = curve2 = pg.PlotDataItem(x=self.x_data, y=self.__mean)
        else:
            curve1 = pg.PlotDataItem(x=self.x_data, y=nanmin(y_data, axis=0))
            curve2 = pg.PlotDataItem(x=self.x_data, y=nanmax(y_data, axis=0))
        self.sel_range.setCurves(curve1, curve2)

    @staticmethod
    def __get_disconnected_curve_data(y_data):
        m, n = y_data.shape
        x = np.arange(m * n) % n + 1
        y = y_data.A.flatten() if sp.issparse(y_data) else y_data.flatten()
        connect = ~np.isnan(y_data.A if sp.issparse(y_data) else y_data)
        connect[:, -1] = False
        connect = connect.flatten()
        return x, y, connect

    @staticmethod
    def __get_disconnected_curve_missing_data(y_data):
        m, n = y_data.shape
        x = np.arange(m * n) % n + 1
        y = y_data.A.flatten() if sp.issparse(y_data) else y_data.flatten()
        connect = np.isnan(y_data.A if sp.issparse(y_data) else y_data)
        # disconnect until the first non nan
        first_non_nan = np.argmin(connect, axis=1)
        for row in np.flatnonzero(first_non_nan):
            connect[row, :first_non_nan[row]] = False
        connect[:, -1] = False
        connect = connect.flatten()
        return x, y, connect

    @staticmethod
    def make_pen(color, width=1):
        pen = QPen(color, width)
        pen.setCosmetic(True)
        return pen


MAX_FEATURES = 200
SEL_MAX_INSTANCES = 10000


class OWLinePlot(OWWidget):
    name = "Line Plot"
    description = "Visualization of data profiles (e.g., time series)."
    icon = "icons/LinePlot.svg"
    priority = 180

    buttons_area_orientation = Qt.Vertical
    enable_selection = Signal(bool)

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = DomainContextHandler()
    group_var = ContextSetting(None)
    show_profiles = Setting(False)
    show_range = Setting(True)
    show_mean = Setting(True)
    show_error = Setting(False)
    auto_commit = Setting(True)
    selection = Setting(None, schema_only=True)
    visual_settings = Setting({}, schema_only=True)

    graph_name = "graph.plotItem"

    class Error(OWWidget.Error):
        not_enough_attrs = Msg("Need at least one numeric feature.")

    class Warning(OWWidget.Warning):
        no_display_option = Msg("No display option is selected.")

    class Information(OWWidget.Information):
        too_many_features = Msg("Data has too many features. Only first {}"
                                " are shown.".format(MAX_FEATURES))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__groups = []
        self.data = None
        self.subset_data = None
        self.subset_indices = None
        self.__pending_selection = self.selection
        self.graph_variables = []
        self.graph = None
        self.group_vars = None
        self.group_view = None
        self.setup_gui()

        VisualSettingsDialog(self, self.graph.parameter_setter.initial_settings)
        self.graph.view_box.selection_changed.connect(self.selection_changed)
        self.enable_selection.connect(self.graph.view_box.enable_selection)

    def setup_gui(self):
        self._add_graph()
        self._add_controls()

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = LinePlotGraph(self)
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        displaybox = gui.widgetBox(self.controlArea, "Display")
        gui.checkBox(displaybox, self, "show_profiles", "Lines",
                     callback=self.__show_profiles_changed,
                     tooltip="Plot lines")
        gui.checkBox(displaybox, self, "show_range", "Range",
                     callback=self.__show_range_changed,
                     tooltip="Plot range between 10th and 90th percentile")
        gui.checkBox(displaybox, self, "show_mean", "Mean",
                     callback=self.__show_mean_changed,
                     tooltip="Plot mean curve")
        gui.checkBox(displaybox, self, "show_error", "Error bars",
                     callback=self.__show_error_changed,
                     tooltip="Show standard deviation")

        self.group_vars = DomainModel(
            placeholder="None", separators=False, valid_types=DiscreteVariable)
        self.group_view = gui.listView(
            self.controlArea, self, "group_var", box="Group by",
            model=self.group_vars, callback=self.__group_var_changed,
            sizeHint=QSize(30, 100), viewType=ListViewSearch,
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self.group_view.setEnabled(False)

        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def __show_profiles_changed(self):
        self.check_display_options()
        self._update_visibility("profiles")

    def __show_range_changed(self):
        self.check_display_options()
        self._update_visibility("range")

    def __show_mean_changed(self):
        self.check_display_options()
        self._update_visibility("mean")

    def __show_error_changed(self):
        self._update_visibility("error")

    def __group_var_changed(self):
        if self.data is None or not self.graph_variables:
            return
        self.plot_groups()
        self._update_profiles_color()
        self._update_sel_profiles_and_range()
        self._update_sel_profiles_color()
        self._update_sub_profiles()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.data = data
        self.clear()
        self.check_data()
        self.check_display_options()

        if self.data is not None:
            self.group_vars.set_domain(self.data.domain)
            self.group_view.setEnabled(len(self.group_vars) > 1)
            self.group_var = self.data.domain.class_var \
                if self.data.domain.has_discrete_class else None

        self.openContext(data)
        self.setup_plot()
        self.commit.now()

    def check_data(self):
        def error(err):
            err()
            self.data = None

        self.clear_messages()
        if self.data is not None:
            self.graph_variables = [var for var in self.data.domain.attributes
                                    if var.is_continuous]
            if len(self.graph_variables) < 1:
                error(self.Error.not_enough_attrs)
            else:
                if len(self.graph_variables) > MAX_FEATURES:
                    self.Information.too_many_features()
                    self.graph_variables = self.graph_variables[:MAX_FEATURES]

    def check_display_options(self):
        self.Warning.no_display_option.clear()
        if self.data is not None:
            if not (self.show_profiles or self.show_range or self.show_mean):
                self.Warning.no_display_option()
            enable = (self.show_profiles or self.show_range) and \
                len(self.data) < SEL_MAX_INSTANCES
            self.enable_selection.emit(enable)

    @Inputs.data_subset
    @check_sql_input
    def set_subset_data(self, subset):
        self.subset_data = subset

    def handleNewSignals(self):
        self.set_subset_ids()
        if self.data is not None:
            self._update_profiles_color()
            self._update_sel_profiles_color()
            self._update_sub_profiles()

    def set_subset_ids(self):
        sub_ids = {e.id for e in self.subset_data} \
            if self.subset_data is not None else {}
        self.subset_indices = None
        if self.data is not None and sub_ids:
            self.subset_indices = [x.id for x in self.data if x.id in sub_ids]

    def setup_plot(self):
        if self.data is None:
            return

        ticks = [a.name for a in self.graph_variables]
        self.graph.getAxis("bottom").set_ticks(ticks)
        self.plot_groups()
        self.apply_selection()
        self.graph.view_box.enableAutoRange()
        self.graph.view_box.updateAutoRange()

    def plot_groups(self):
        self._remove_groups()
        data = self.data[:, self.graph_variables]
        if self.group_var is None:
            self._plot_group(data, np.arange(len(data)))
        else:
            class_col_data, _ = self.data.get_column_view(self.group_var)
            for index in range(len(self.group_var.values)):
                indices = np.flatnonzero(class_col_data == index)
                if len(indices) == 0:
                    continue
                group_data = self.data[indices, self.graph_variables]
                self._plot_group(group_data, indices, index)
        self.graph.update_legend(self.group_var)
        self.graph.groups = self.__groups
        self.graph.view_box.add_profiles(data.X)

    def _remove_groups(self):
        for group in self.__groups:
            group.remove_items()
        self.graph.view_box.remove_profiles()
        self.graph.groups = []
        self.__groups = []

    def _plot_group(self, data, indices, index=None):
        color = self.__get_group_color(index)
        group = ProfileGroup(data, indices, color, self.graph)
        kwargs = self.__get_visibility_flags()
        group.set_visible_error(**kwargs)
        group.set_visible_mean(**kwargs)
        group.set_visible_range(**kwargs)
        group.set_visible_profiles(**kwargs)
        self.__groups.append(group)

    def __get_group_color(self, index):
        if self.group_var is not None:
            return QColor(*self.group_var.colors[index])
        return QColor(LinePlotStyle.DEFAULT_COLOR)

    def __get_visibility_flags(self):
        return {"show_profiles": self.show_profiles,
                "show_range": self.show_range,
                "show_mean": self.show_mean,
                "show_error": self.show_error}

    def _update_profiles_color(self):
        # color alpha depends on subset and selection; with selection or
        # subset profiles color has more opacity
        if not self.show_profiles:
            return
        for group in self.__groups:
            has_sel = bool(self.subset_indices) or bool(self.selection)
            group.update_profiles_color(has_sel)

    def _update_sel_profiles_and_range(self):
        # mark selected instances and selected range
        if not (self.show_profiles or self.show_range):
            return
        for group in self.__groups:
            inds = [i for i in group.indices if self.__in(i, self.selection)]
            table = self.data[inds, self.graph_variables].X if inds else None
            if self.show_profiles:
                group.update_sel_profiles(table)
            if self.show_range:
                group.update_sel_range(table)

    def _update_sel_profiles_color(self):
        # color depends on subset; when subset is present,
        # selected profiles are black
        if not self.selection or not self.show_profiles:
            return
        for group in self.__groups:
            group.update_sel_profiles_color(bool(self.subset_indices))

    def _update_sub_profiles(self):
        # mark subset instances
        if not (self.show_profiles or self.show_range):
            return
        for group in self.__groups:
            inds = [i for i, _id in zip(group.indices, group.ids)
                    if self.__in(_id, self.subset_indices)]
            table = self.data[inds, self.graph_variables].X if inds else None
            group.update_sub_profiles(table)

    def _update_visibility(self, obj_name):
        if len(self.__groups) == 0:
            return
        self._update_profiles_color()
        self._update_sel_profiles_and_range()
        self._update_sel_profiles_color()
        kwargs = self.__get_visibility_flags()
        for group in self.__groups:
            getattr(group, "set_visible_{}".format(obj_name))(**kwargs)
        self.graph.view_box.updateAutoRange()

    def apply_selection(self):
        if self.data is not None and self.__pending_selection is not None:
            sel = [i for i in self.__pending_selection if i < len(self.data)]
            mask = np.zeros(len(self.data), dtype=bool)
            mask[sel] = True
            self.selection_changed(mask)
            self.__pending_selection = None

    def selection_changed(self, mask):
        if self.data is None:
            return
        indices = np.arange(len(self.data))[mask]
        self.graph.select(indices)
        old = self.selection
        self.selection = None if self.data and isinstance(self.data, SqlTable)\
            else list(self.graph.selection)
        if not old and self.selection or old and not self.selection:
            self._update_profiles_color()
        self._update_sel_profiles_and_range()
        self._update_sel_profiles_color()
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        selected = self.data[self.selection] \
            if self.data is not None and bool(self.selection) else None
        annotated = create_annotated_table(self.data, self.selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def send_report(self):
        if self.data is None:
            return

        caption = report.render_items_vert((("Group by", self.group_var),))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def sizeHint(self):
        return QSize(1132, 708)

    def clear(self):
        self.selection = None
        self.__groups = []
        self.graph_variables = []
        self.graph.reset()
        self.group_vars.set_domain(None)
        self.group_view.setEnabled(False)

    @staticmethod
    def __in(obj, collection):
        return collection is not None and obj in collection

    def set_visual_settings(self, key, value):
        self.graph.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value


if __name__ == "__main__":
    brown = Table("brown-selected")
    WidgetPreview(OWLinePlot).run(set_data=brown, set_subset_data=brown[:30])
