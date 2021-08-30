from typing import List, Optional, Tuple, Union, Dict
from functools import lru_cache
from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import Qt, QPointF, QSize, Signal, QRectF
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QApplication, QToolTip, QGraphicsSceneHelpEvent

import pyqtgraph as pg

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog, \
    KeyType, ValueType

from Orange.data import Table, DiscreteVariable, ContinuousVariable, \
    StringVariable, Variable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler, SettingProvider
from Orange.widgets.utils.annotated_data import create_annotated_table, \
    ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils import instance_tooltip
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import AxisItem, \
    HelpEventDelegate
from Orange.widgets.widget import OWWidget, Input, Output, Msg

MAX_INSTANCES = 200


class BarPlotViewBox(pg.ViewBox):
    def __init__(self, parent):
        super().__init__()
        self.graph = parent
        self.setMouseMode(self.RectMode)

    def mouseDragEvent(self, ev, axis=None):
        if self.graph.state == SELECT and axis is None:
            ev.accept()
            if ev.button() == Qt.LeftButton:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    p1, p2 = ev.buttonDownPos(ev.button()), ev.pos()
                    p1 = self.mapToView(p1)
                    p2 = self.mapToView(p2)
                    self.graph.select_by_rectangle(QRectF(p1, p2))
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.graph.select_by_click(self.mapSceneToView(ev.scenePos()))
            ev.accept()


class ParameterSetter(CommonParameterSetter):
    GRID_LABEL, SHOW_GRID_LABEL = "Gridlines", "Show"
    DEFAULT_ALPHA_GRID, DEFAULT_SHOW_GRID = 80, True
    BOTTOM_AXIS_LABEL, GROUP_AXIS_LABEL = "Bottom axis", "Group axis"
    IS_VERTICAL_LABEL = "Vertical ticks"

    def __init__(self, master):
        self.grid_settings: Dict = None
        self.master: BarPlotGraph = master
        super().__init__()

    def update_setters(self):
        self.grid_settings = {
            Updater.ALPHA_LABEL: self.DEFAULT_ALPHA_GRID,
            self.SHOW_GRID_LABEL: self.DEFAULT_SHOW_GRID,
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
            },
            self.PLOT_BOX: {
                self.GRID_LABEL: {
                    self.SHOW_GRID_LABEL: (None, True),
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          self.DEFAULT_ALPHA_GRID),
                },
                self.BOTTOM_AXIS_LABEL: {
                    self.IS_VERTICAL_LABEL: (None, True),
                },
                self.GROUP_AXIS_LABEL: {
                    self.IS_VERTICAL_LABEL: (None, False),
                },
            },
        }

        def update_grid(**settings):
            self.grid_settings.update(**settings)
            self.master.showGrid(y=self.grid_settings[self.SHOW_GRID_LABEL],
                          alpha=self.grid_settings[Updater.ALPHA_LABEL] / 255)

        def update_bottom_axis(**settings):
            axis = self.master.getAxis("bottom")
            axis.setRotateTicks(settings[self.IS_VERTICAL_LABEL])

        def update_group_axis(**settings):
            axis = self.master.group_axis
            axis.setRotateTicks(settings[self.IS_VERTICAL_LABEL])

        self._setters[self.PLOT_BOX] = {
            self.GRID_LABEL: update_grid,
            self.BOTTOM_AXIS_LABEL: update_bottom_axis,
            self.GROUP_AXIS_LABEL: update_group_axis,
        }

    @property
    def title_item(self):
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self):
        return self.master.legend.items


class BarPlotGraph(pg.PlotWidget):
    selection_changed = Signal(list)
    bar_width = 0.7

    def __init__(self, master, parent=None):
        self.selection = []
        self.master: OWBarPlot = master
        self.state: int = SELECT
        self.bar_item: pg.BarGraphItem = None
        super().__init__(
            parent=parent,
            viewBox=BarPlotViewBox(self),
            background="w", enableMenu=False,
            axisItems={"bottom": AxisItem(orientation="bottom",
                                          rotate_ticks=True),
                       "left": AxisItem(orientation="left")}
        )
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 0, 0, 10)
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        self.group_axis = AxisItem("bottom")
        self.group_axis.hide()
        self.group_axis.linkToView(self.getViewBox())
        self.getPlotItem().layout.addItem(self.group_axis, 4, 1)

        self.legend = self._create_legend()

        self.tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self.tooltip_delegate)

        self.parameter_setter = ParameterSetter(self)

        self.showGrid(y=self.parameter_setter.DEFAULT_SHOW_GRID,
                      alpha=self.parameter_setter.DEFAULT_ALPHA_GRID / 255)

    def _create_legend(self):
        legend = LegendItem()
        legend.setParentItem(self.getViewBox())
        legend.anchor((1, 0), (1, 0), offset=(-3, 1))
        legend.hide()
        return legend

    def update_legend(self):
        self.legend.clear()
        self.legend.hide()
        for color, text in self.master.get_legend_data():
            dot = pg.ScatterPlotItem(
                pen=pg.mkPen(color=color),
                brush=pg.mkBrush(color=color)
            )
            self.legend.addItem(dot, escape(text))
            self.legend.show()
        Updater.update_legend_font(self.legend.items,
                                   **self.parameter_setter.legend_settings)

    def reset_graph(self):
        self.clear()
        self.update_bars()
        self.update_axes()
        self.update_group_lines()
        self.update_legend()
        self.reset_view()

    def update_bars(self):
        if self.bar_item is not None:
            self.removeItem(self.bar_item)
            self.bar_item = None

        values = self.master.get_values()
        if values is None:
            return

        self.bar_item = pg.BarGraphItem(
            x=np.arange(len(values)),
            height=values,
            width=self.bar_width,
            pen=pg.mkPen(QColor(Qt.white)),
            labels=self.master.get_labels(),
            brushes=self.master.get_colors(),
        )
        self.addItem(self.bar_item)
        self.__select_bars()

    def update_axes(self):
        if self.bar_item is not None:
            self.showAxis("left")
            self.showAxis("bottom")
            self.group_axis.show()

            vals_label, group_label, annot_label = self.master.get_axes()
            self.setLabel(axis="left", text=vals_label)
            self.setLabel(axis="bottom", text=annot_label)
            self.group_axis.setLabel(group_label)

            ticks = [list(enumerate(self.master.get_labels()))]
            self.getAxis('bottom').setTicks(ticks)

            labels = np.array(self.master.get_group_labels())
            _, indices, counts = \
                np.unique(labels, return_index=True, return_counts=True)
            ticks = [[(i + (c - 1) / 2, labels[i]) for i, c in
                      zip(indices, counts)]]
            self.group_axis.setTicks(ticks)

            if not group_label:
                self.group_axis.hide()
            elif not annot_label:
                self.hideAxis("bottom")
        else:
            self.hideAxis("left")
            self.hideAxis("bottom")
            self.group_axis.hide()

    def reset_view(self):
        if self.bar_item is None:
            return
        values = np.append(self.bar_item.opts["height"], 0)
        min_ = np.nanmin(values)
        max_ = -min_ + np.nanmax(values)
        rect = QRectF(-0.5, min_, len(values) - 1, max_)
        self.getViewBox().setRange(rect)

    def zoom_button_clicked(self):
        self.state = ZOOMING
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def pan_button_clicked(self):
        self.state = PANNING
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)

    def select_button_clicked(self):
        self.state = SELECT
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def reset_button_clicked(self):
        self.reset_view()

    def update_group_lines(self):
        if self.bar_item is None:
            return

        labels = np.array(self.master.get_group_labels())
        if labels is None or len(labels) == 0:
            return

        _, indices = np.unique(labels, return_index=True)
        offset = self.bar_width / 2 + (1 - self.bar_width) / 2
        for index in sorted(indices)[1:]:
            line = pg.InfiniteLine(pos=index - offset, angle=90)
            self.addItem(line)

    def select_by_rectangle(self, rect: QRectF):
        if self.bar_item is None:
            return

        x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
        y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
        x = self.bar_item.opts["x"]
        height = self.bar_item.opts["height"]
        d = self.bar_width / 2
        # positive bars
        mask = (x0 <= x + d) & (x1 >= x - d) & (y0 <= height) & (y1 > 0)
        # negative bars
        mask |= (x0 <= x + d) & (x1 >= x - d) & (y0 <= 0) & (y1 > height)
        self.select_by_indices(list(np.flatnonzero(mask)))

    def select_by_click(self, p: QPointF):
        if self.bar_item is None:
            return

        index = self.__get_index_at(p)
        self.select_by_indices([index] if index is not None else [])

    def __get_index_at(self, p: QPointF):
        x = p.x()
        index = round(x)
        heights = self.bar_item.opts["height"]
        if 0 <= index < len(heights) and abs(x - index) <= self.bar_width / 2:
            height = heights[index]  # pylint: disable=unsubscriptable-object
            if 0 <= p.y() <= height or height <= p.y() <= 0:
                return index
        return None

    def select_by_indices(self, indices: List):
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ControlModifier:
            self.selection = list(set(self.selection) ^ set(indices))
        elif keys & Qt.AltModifier:
            self.selection = list(set(self.selection) - set(indices))
        elif keys & Qt.ShiftModifier:
            self.selection = list(set(self.selection) | set(indices))
        else:
            self.selection = list(set(indices))
        self.__select_bars()
        self.selection_changed.emit(self.selection)

    def __select_bars(self):
        if self.bar_item is None:
            return

        n = len(self.bar_item.opts["height"])
        pens = np.full(n, pg.mkPen(QColor(Qt.white)))
        pen = pg.mkPen(QColor(Qt.black))
        pen.setStyle(Qt.DashLine)
        pens[self.selection] = pen
        self.bar_item.setOpts(pens=pens)

    def help_event(self, ev: QGraphicsSceneHelpEvent):
        if self.bar_item is None:
            return False

        index = self.__get_index_at(self.bar_item.mapFromScene(ev.scenePos()))
        text = ""
        if index is not None:
            text = self.master.get_tooltip(index)
        if text:
            QToolTip.showText(ev.screenPos(), text, widget=self)
            return True
        else:
            return False


class OWBarPlot(OWWidget):
    name = "Bar Plot"
    description = "Visualizes comparisons among categorical variables."
    icon = "icons/BarPlot.svg"
    priority = 190
    keywords = ["chart"]

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    buttons_area_orientation = Qt.Vertical
    settingsHandler = DomainContextHandler()
    selected_var = ContextSetting(None)
    group_var = ContextSetting(None)
    annot_var = ContextSetting(None)
    color_var = ContextSetting(None)
    auto_commit = Setting(True)
    selection = Setting(None, schema_only=True)
    visual_settings = Setting({}, schema_only=True)

    graph = SettingProvider(BarPlotGraph)
    graph_name = "graph.plotItem"

    class Error(OWWidget.Error):
        no_cont_features = Msg("Plotting requires a numeric feature.")

    class Information(OWWidget.Information):
        too_many_instances = Msg("Data has too many instances. Only first {}"
                                 " are shown.".format(MAX_INSTANCES))

    enumeration = "Enumeration"

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.orig_data: Optional[Table] = None
        self.subset_data: Optional[Table] = None
        self.subset_indices = []
        self.graph: Optional[BarPlotGraph] = None
        self._selected_var_model: Optional[DomainModel] = None
        self._group_var_model: Optional[DomainModel] = None
        self._annot_var_model: Optional[DomainModel] = None
        self._color_var_model: Optional[DomainModel] = None
        self.__pending_selection = self.selection

        self.setup_gui()
        VisualSettingsDialog(
            self, self.graph.parameter_setter.initial_settings
        )

    def setup_gui(self):
        self._add_graph()
        self._add_controls()

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = BarPlotGraph(self)
        self.graph.selection_changed.connect(self.__selection_changed)
        box.layout().addWidget(self.graph)

    def __selection_changed(self, indices: List):
        self.selection = list(set(self.grouped_indices[indices]))
        self.commit.deferred()

    def _add_controls(self):
        box = gui.vBox(self.controlArea, True)
        gui.rubber(self.controlArea)
        self._selected_var_model = DomainModel(valid_types=ContinuousVariable)
        gui.comboBox(
            box, self, "selected_var", label="Values:",
            model=self._selected_var_model, contentsLength=12, searchable=True,
            orientation=Qt.Horizontal, callback=self.__parameter_changed,
        )

        self._group_var_model = DomainModel(
            placeholder="None", valid_types=DiscreteVariable
        )
        gui.comboBox(
            box, self, "group_var", label="Group by:",
            model=self._group_var_model, contentsLength=12, searchable=True,
            orientation=Qt.Horizontal, callback=self.__group_var_changed,
        )

        self._annot_var_model = DomainModel(
            placeholder="None",
            valid_types=(DiscreteVariable, StringVariable)
        )
        self._annot_var_model.order = self._annot_var_model.order[:1] + \
                                      (self.enumeration,) + \
                                      self._annot_var_model.order[1:]
        gui.comboBox(
            box, self, "annot_var", label="Annotations:",
            model=self._annot_var_model, contentsLength=12, searchable=True,
            orientation=Qt.Horizontal, callback=self.__parameter_changed,
        )

        self._color_var_model = DomainModel(
            placeholder="(Same color)", valid_types=DiscreteVariable
        )
        gui.comboBox(
            box, self, "color_var", label="Color:",
            model=self._color_var_model,
            contentsLength=12, searchable=True, orientation=Qt.Horizontal,
            callback=self.__parameter_changed,
        )

        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)

        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def __parameter_changed(self):
        self.graph.reset_graph()

    def __group_var_changed(self):
        self.clear_cache()
        self.graph.selection = self.grouped_indices_inverted
        self.__parameter_changed()

    @property
    @lru_cache()
    def grouped_indices(self):
        indices = []
        if self.data:
            indices = np.arange(len(self.data))
            if self.group_var:
                group_by = self.data.get_column_view(self.group_var)[0]
                indices = np.argsort(group_by, kind="mergesort")
        return indices

    @property
    def grouped_indices_inverted(self):
        mask = np.isin(self.grouped_indices, self.selection)
        return np.flatnonzero(mask)

    @property
    def grouped_data(self):
        return self.data[self.grouped_indices]

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.clear()
        self.orig_data = self.data = data
        self.check_data()
        self.init_attr_values()
        self.openContext(self.data)
        self.clear_cache()
        self.commit.now()

    def check_data(self):
        self.clear_messages()
        if self.data is not None:
            if self.data.domain.has_continuous_attributes(True, True) == 0:
                self.Error.no_cont_features()
                self.data = None
            elif len(self.data) > MAX_INSTANCES:
                self.Information.too_many_instances()
                self.data = self.data[:MAX_INSTANCES]

    def init_attr_values(self):
        domain = self.data.domain if self.data else None
        for model, var in ((self._selected_var_model, "selected_var"),
                           (self._group_var_model, "group_var"),
                           (self._annot_var_model, "annot_var"),
                           (self._color_var_model, "color_var")):
            model.set_domain(domain)
            setattr(self, var, None)

        if self._selected_var_model:
            self.selected_var = self._selected_var_model[0]
        if domain is not None and domain.has_discrete_class:
            self.color_var = domain.class_var

    @Inputs.data_subset
    @check_sql_input
    def set_subset_data(self, data: Optional[Table]):
        self.subset_data = data

    def handleNewSignals(self):
        self._handle_subset_data()
        self.setup_plot()

    def _handle_subset_data(self):
        sub_ids = {e.id for e in self.subset_data} \
            if self.subset_data is not None else {}
        self.subset_indices = []
        if self.data is not None and sub_ids:
            self.subset_indices = [x.id for x in self.data if x.id in sub_ids]

    def get_values(self) -> Optional[np.ndarray]:
        if not self.data or not self.selected_var:
            return None
        return self.grouped_data.get_column_view(self.selected_var)[0]

    def get_labels(self) -> Optional[Union[List, np.ndarray]]:
        if not self.data:
            return None
        elif not self.annot_var:
            return []
        elif self.annot_var == self.enumeration:
            return np.arange(1, len(self.data) + 1)[self.grouped_indices]
        else:
            return [self.annot_var.str_val(row[self.annot_var])
                    for row in self.grouped_data]

    def get_group_labels(self) -> Optional[List]:
        if not self.data:
            return None
        elif not self.group_var:
            return []
        else:
            return [self.group_var.str_val(row[self.group_var])
                    for row in self.grouped_data]

    def get_legend_data(self) -> List:
        if not self.data or not self.color_var:
            return []
        else:
            assert self.color_var.is_discrete
            return [(QColor(*color), text) for color, text in
                    zip(self.color_var.colors, self.color_var.values)]

    def get_colors(self) -> Optional[List[QColor]]:
        def create_color(i, id_):
            lighter = id_ not in self.subset_indices and self.subset_indices
            alpha = 50 if lighter else 255
            if np.isnan(i):
                return QColor(*(128, 128, 128, alpha))
            return QColor(*np.append(self.color_var.colors[int(i)], alpha))

        if not self.data:
            return None
        elif not self.color_var:
            return [create_color(np.nan, id_) for id_ in self.grouped_data.ids]
        else:
            assert self.color_var.is_discrete
            col = self.grouped_data.get_column_view(self.color_var)[0]
            return [create_color(i, id_) for id_, i in
                    zip(self.grouped_data.ids, col)]

    def get_tooltip(self, index: int) -> str:
        if not self.data:
            return ""
        row = self.grouped_data[index]
        attrs = [self.selected_var]
        if self.group_var and self.group_var not in attrs:
            attrs.append(self.group_var)
        if isinstance(self.annot_var, Variable) and self.annot_var not in attrs:
            attrs.append(self.annot_var)
        if self.color_var and self.color_var not in attrs:
            attrs.append(self.color_var)
        text = "<br/>".join(escape('{} = {}'.format(var.name, row[var]))
                            for var in attrs)
        others = instance_tooltip(self.data.domain, row, skip_attrs=attrs)
        if others:
            text = "<b>{}</b><br/><br/>{}".format(text, others)
        return text

    def get_axes(self) -> Optional[Tuple[str, str, str]]:
        if not self.data:
            return None
        return (self.selected_var.name,
                self.group_var.name if self.group_var else "",
                self.annot_var if self.annot_var else "")

    def setup_plot(self):
        self.graph.reset_graph()
        self.apply_selection()

    def apply_selection(self):
        if self.data and self.__pending_selection is not None:
            self.selection = [i for i in self.__pending_selection
                              if i < len(self.data)]
            self.graph.select_by_indices(self.grouped_indices_inverted)
            self.__pending_selection = None

    @gui.deferred
    def commit(self):
        selected = None
        if self.data is not None and bool(self.selection):
            selected = self.data[self.selection]
        annotated = create_annotated_table(self.orig_data, self.selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def clear(self):
        self.selection = None
        self.graph.selection = []
        self.clear_cache()

    @staticmethod
    def clear_cache():
        OWBarPlot.grouped_indices.fget.cache_clear()

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value

    def sizeHint(self):  # pylint: disable=no-self-use
        return QSize(1132, 708)

    def showEvent(self, event):
        super().showEvent(event)
        self.graph.reset_view()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    iris = Table("iris")
    WidgetPreview(OWBarPlot).run(set_data=iris[::3],
                                 set_subset_data=iris[::15])
