import math
from typing import List, Callable, Optional
from xml.sax.saxutils import escape

import numpy as np
import scipy.stats as ss
from scipy.stats import linregress
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from AnyQt.QtCore import Qt, QTimer, QPointF
from AnyQt.QtGui import QColor, QFont, QFontMetrics
from AnyQt.QtWidgets import QGroupBox, QSizePolicy, QPushButton

import pyqtgraph as pg

from orangewidget.utils import load_styled_icon
from orangewidget.utils.combobox import ComboBoxSearch

from Orange.data import Table, Domain, DiscreteVariable, Variable
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess.score import ReliefF, RReliefF

from Orange.widgets import gui, report
from Orange.widgets.io import MatplotlibFormat, MatplotlibPDFFormat
from Orange.widgets.settings import (
    Setting, ContextSetting, SettingProvider, IncompatibleContext)
from Orange.widgets.utils import get_variable_values_sorted
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase, \
    ScatterBaseParameterSetter
from Orange.widgets.visualize.utils.error_bars_dialog import ErrorBarsDialog
from Orange.widgets.visualize.utils.vizrank import VizRankDialogAttrPair, \
    VizRankMixin
from Orange.widgets.visualize.utils.customizableplot import Updater
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import AttributeList, Msg, Input, Output


class ScatterPlotVizRank(VizRankDialogAttrPair):
    minK = 10

    def compute_score(self, state):
        # pylint: disable=invalid-unary-operand-type
        attrs = [self.attr_order[i] for i in state]
        data = self.data
        data = data.transform(Domain(attrs, self.attr_color))
        data = data[~np.isnan(data.X).any(axis=1) & ~np.isnan(data.Y).T]
        if len(data) < self.minK:
            return None
        n_neighbors = min(self.minK, len(data) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(data.X)
        ind = knn.kneighbors(return_distance=False)
        if data.domain.has_discrete_class:
            return -np.sum(data.Y[ind] == data.Y.reshape(-1, 1)) / \
                   n_neighbors / len(data.Y)
        else:
            return -r2_score(data.Y, np.mean(data.Y[ind], axis=1)) * \
                   (len(data.Y) / len(self.data))

    def score_attributes(self):
        assert self.attr_color is not None
        attrs = [
            v
            for v in self.attrs  # same attributes that are in xy combos
            if v is not self.attr_color and v.is_primitive()
        ]
        domain = Domain(attributes=attrs, class_vars=self.attr_color)
        data = self.data.transform(domain)
        relief = ReliefF if isinstance(domain.class_var, DiscreteVariable) \
            else RReliefF
        weights = relief(
            n_iterations=100, k_nearest=self.minK, random_state=0)(data)
        attrs = sorted(zip(weights, domain.attributes),
                       key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]


class ParameterSetter(ScatterBaseParameterSetter):
    DEFAULT_LINE_WIDTH = 3
    DEFAULT_LINE_ALPHA = 255

    def __init__(self, master):
        super().__init__(master)
        self.reg_line_label_font = QFont()
        self.reg_line_settings = {
            Updater.WIDTH_LABEL: self.DEFAULT_LINE_WIDTH,
            Updater.ALPHA_LABEL: self.DEFAULT_LINE_ALPHA,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
        }

    def update_setters(self):
        super().update_setters()
        self.initial_settings[self.LABELS_BOX].update({
            self.AXIS_TITLE_LABEL: self.FONT_SETTING,
            self.AXIS_TICKS_LABEL: self.FONT_SETTING,
            self.LINE_LAB_LABEL: self.FONT_SETTING
        })
        self.initial_settings[self.PLOT_BOX] = {}
        self.initial_settings[self.PLOT_BOX][self.LINE_LABEL] = {
            Updater.WIDTH_LABEL: (range(1, 10), self.DEFAULT_LINE_WIDTH),
            Updater.ALPHA_LABEL: (range(0, 255, 5), self.DEFAULT_LINE_ALPHA),
            Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                  Updater.DEFAULT_LINE_STYLE),
        }

        def update_lines(**settings):
            self.reg_line_settings.update(**settings)
            Updater.update_inf_lines(self.reg_line_items,
                                     **self.reg_line_settings)
            Updater.update_lines(self.ellipse_items,
                                 **self.reg_line_settings)
            self.master.update_reg_line_label_colors()

        def update_line_label(**settings):
            self.reg_line_label_font = \
                Updater.change_font(self.reg_line_label_font, settings)
            Updater.update_label_font(self.reg_line_label_items,
                                      self.reg_line_label_font)

        self._setters[self.LABELS_BOX][self.LINE_LAB_LABEL] = update_line_label
        self._setters[self.PLOT_BOX] = {self.LINE_LABEL: update_lines}

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.plot_widget.plotItem.axes.values()]

    @property
    def reg_line_items(self):
        return self.master.reg_line_items

    @property
    def reg_line_label_items(self):
        return [line.label for line in self.master.reg_line_items
                if hasattr(line, "label")]

    @property
    def ellipse_items(self):
        return self.master.ellipse_items


class OWScatterPlotGraph(OWScatterPlotBase):
    show_reg_line = Setting(False)
    orthonormal_regression = Setting(False)
    show_ellipse = Setting(False)
    jitter_continuous = Setting(False)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.parameter_setter = ParameterSetter(self)
        self.reg_line_items = []
        self.ellipse_items: List[pg.PlotCurveItem] = []
        self.error_bars_items: List[pg.ErrorBarItem] = []
        self.view_box.sigResized.connect(self.update_error_bars)
        self.view_box.sigRangeChanged.connect(self.update_error_bars)

    def clear(self):
        super().clear()
        self.reg_line_items.clear()
        self.ellipse_items.clear()
        self.error_bars_items.clear()

    def update_coordinates(self):
        super().update_coordinates()
        self.set_aggregation()
        self.update_axes()
        self.update_error_bars()
        # Don't update_regression line here: update_coordinates is always
        # followed by update_point_props, which calls update_colors

    def update_colors(self):
        super().update_colors()
        self.update_regression_line()
        self.update_ellipse()

    def jitter_coordinates(self, x, y):
        if self.jitter_size == 0:
            return x, y

        def get_span(attr):
            if attr.is_discrete:
                # Assuming the maximal jitter size is 10, a span of 4 will
                # jitter by 4 * 10 / 100 = 0.4, so there will be no overlap
                return 4
            elif self.jitter_continuous:
                return None  # Let _jitter_data determine the span
            else:
                return 0  # No jittering
        span_x = get_span(self.master.attr_x)
        span_y = get_span(self.master.attr_y)
        if span_x == 0 and span_y == 0:
            return x, y
        return self._jitter_data(x, y, span_x, span_y)

    def update_axes(self):
        for axis, var in self.master.get_axes().items():
            axis_item = self.plot_widget.plotItem.getAxis(axis)
            if var and var.is_discrete:
                ticks = [list(enumerate(get_variable_values_sorted(var)))]
                axis_item.setTicks(ticks)
            else:
                axis_item.setTicks(None)
            use_time = var and var.is_time
            self.plot_widget.plotItem.getAxis(axis).use_time(use_time)
            self.plot_widget.setLabel(axis=axis, text=var or "")
            if not var:
                self.plot_widget.hideAxis(axis)

    @staticmethod
    def _orthonormal_line(x, y, color, width, style=Qt.SolidLine):
        # https://en.wikipedia.org/wiki/Deming_regression, with δ=0.
        pen = pg.mkPen(color=color, width=width, style=style)
        xm = np.mean(x)
        ym = np.mean(y)
        sxx, sxy, _, syy = np.cov(x, y, ddof=1).flatten()

        if sxy != 0:  # also covers sxx != 0 and syy != 0
            slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) \
                    / (2 * sxy)
            intercept = ym - slope * xm
            xmin = x.min()
            return pg.InfiniteLine(
                QPointF(xmin, xmin * slope + intercept),
                np.degrees(np.arctan(slope)),
                pen)
        elif (sxx == 0) == (syy == 0):  # both zero or non-zero -> can't draw
            return None
        elif sxx != 0:
            return pg.InfiniteLine(QPointF(x.min(), ym), 0, pen)
        else:
            return pg.InfiniteLine(QPointF(xm, y.min()), 90, pen)

    @staticmethod
    def _regression_line(x, y, color, width, style=Qt.SolidLine):
        min_x, max_x = np.min(x), np.max(x)
        if min_x == max_x:
            return None
        slope, intercept, rvalue, _, _ = linregress(x, y)
        angle = np.degrees(np.arctan(slope))
        start_y = min_x * slope + intercept
        l_opts = dict(color=color, position=0.85,
                      rotateAxis=(1, 0), movable=True)
        return pg.InfiniteLine(
            pos=QPointF(min_x, start_y), angle=angle,
            pen=pg.mkPen(color=color, width=width, style=style),
            label=f"r = {rvalue:.2f}", labelOpts=l_opts)

    def _add_line(self, x, y, color):
        width = self.parameter_setter.reg_line_settings[Updater.WIDTH_LABEL]
        alpha = self.parameter_setter.reg_line_settings[Updater.ALPHA_LABEL]
        style = self.parameter_setter.reg_line_settings[Updater.STYLE_LABEL]
        style = Updater.LINE_STYLES[style]
        color.setAlpha(alpha)
        if self.orthonormal_regression:
            line = self._orthonormal_line(x, y, color, width, style)
        else:
            line = self._regression_line(x, y, color, width, style)
        if line is None:
            return
        self.plot_widget.addItem(line)
        self.reg_line_items.append(line)

        if hasattr(line, "label"):
            Updater.update_label_font(
                [line.label], self.parameter_setter.reg_line_label_font
            )

    def update_reg_line_label_colors(self):
        for line in self.reg_line_items:
            if hasattr(line, "label"):
                color = 0.0 if self.class_density \
                    else line.pen.color().darker(175)
                line.label.setColor(color)

    def update_density(self):
        super().update_density()
        self.update_reg_line_label_colors()

    def update_regression_line(self):
        self._update_curve(self.reg_line_items,
                           self.show_reg_line,
                           self._add_line)
        self.update_reg_line_label_colors()

    def update_ellipse(self):
        self._update_curve(self.ellipse_items,
                           self.show_ellipse,
                           self._add_ellipse)

    def _update_curve(self, items: List, show: bool, add: Callable):
        for item in items:
            self.plot_widget.removeItem(item)
        items.clear()
        if not (show and self.master.can_draw_regression_line()):
            return
        x, y = self.master.get_coordinates_data()
        if x is None or len(x) < 2:
            return
        add(x, y, QColor("#505050"))
        if self.master.is_continuous_color() or self.palette is None \
                or len(self.palette) == 0:
            return
        c_data = self.master.get_color_data()
        if c_data is None:
            return
        c_data = c_data.astype(int)
        for val in range(c_data.max() + 1):
            mask = c_data == val
            if mask.sum() > 1:
                add(x[mask], y[mask], self.palette[val].darker(135))

    def _add_ellipse(self, x: np.ndarray, y: np.ndarray, color: QColor) -> np.ndarray:
        # https://github.com/ChristianGoueguel/HotellingEllipse/blob/master/R/ellipseCoord.R
        points = np.vstack([x, y]).T
        mu = np.mean(points, axis=0)
        cov = np.cov(*(points - mu).T)
        vals, vects = np.linalg.eig(cov)
        angle = math.atan2(vects[1, 0], vects[0, 0])
        matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])

        n = len(x)
        f = ss.f.ppf(0.95, 2, n - 2)
        f = f * 2 * (n - 1) / (n - 2)
        m = [np.pi * i / 100 for i in range(201)]
        cx = np.cos(m) * np.sqrt(vals[0] * f)
        cy = np.sin(m) * np.sqrt(vals[1] * f)

        pts = np.vstack([cx, cy])
        pts = matrix.dot(pts)
        cx = pts[0] + mu[0]
        cy = pts[1] + mu[1]

        width = self.parameter_setter.reg_line_settings[Updater.WIDTH_LABEL]
        alpha = self.parameter_setter.reg_line_settings[Updater.ALPHA_LABEL]
        style = self.parameter_setter.reg_line_settings[Updater.STYLE_LABEL]
        style = Updater.LINE_STYLES[style]
        color.setAlpha(alpha)

        pen = pg.mkPen(color=color, width=width, style=style)
        ellipse = pg.PlotCurveItem(cx, cy, pen=pen)
        self.plot_widget.addItem(ellipse)
        self.ellipse_items.append(ellipse)

    def update_jittering(self):
        super().update_jittering()
        self.update_error_bars()

    def allow_aggregation(self):
        # Reimplement to allow aggregation when jittering is non-zero but
        # disabled for continuous variables
        return (
            (self.selection is None or len(self.selection) == 0)
            and not self.subset_is_shown
            and (self.labels is None or len(self.labels) == 0)
            and (self.jitter_size == 0
                 or (self.master.attr_x.is_continuous
                     and self.master.attr_y.is_continuous
                     and not self.jitter_continuous)
                 )
        )

    def update_error_bars(self):
        for item in self.error_bars_items:
            self.plot_widget.removeItem(item)
        self.error_bars_items.clear()
        if not self.master.can_draw_regression_line():
            return

        x, y = self.get_coordinates()
        if x is None:
            return

        top, bottom, left, right = self.master.get_errors_data()
        if top is None and bottom is None and left is None and right is None:
            return

        px, py = self.view_box.viewPixelSize()
        pen = pg.mkPen(color=QColor("#505050"))

        # x axis
        error_bars = pg.ErrorBarItem(x=x, y=y, left=left, right=right,
                                     beam=py * 10, pen=pen)
        error_bars.setZValue(-1)
        self.plot_widget.addItem(error_bars)
        self.error_bars_items.append(error_bars)

        # y axis
        error_bars = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom,
                                     beam=px * 10, pen=pen)
        error_bars.setZValue(-1)
        self.plot_widget.addItem(error_bars)
        self.error_bars_items.append(error_bars)


class OWScatterPlot(OWDataProjectionWidget, VizRankMixin(ScatterPlotVizRank)):
    """Scatterplot visualization with explorative analysis and intelligent
    data visualization enhancements."""

    name = "Scatter Plot"
    description = "Interactive scatter plot visualization with " \
                  "intelligent data visualization enhancements."
    icon = "icons/ScatterPlot.svg"
    priority = 140
    keywords = "scatter plot"

    class Inputs(OWDataProjectionWidget.Inputs):
        features = Input("Features", AttributeList)

    class Outputs(OWDataProjectionWidget.Outputs):
        features = Output("Features", AttributeList, dynamic=False)

    settings_version = 5
    auto_sample = Setting(True)
    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    attr_x_upper = ContextSetting(None)
    attr_x_lower = ContextSetting(None)
    attr_x_is_abs = Setting(False)
    attr_y_upper = ContextSetting(None)
    attr_y_lower = ContextSetting(None)
    attr_y_is_abs = Setting(False)
    tooltip_shows_all = Setting(True)

    GRAPH_CLASS = OWScatterPlotGraph
    graph = SettingProvider(OWScatterPlotGraph)
    embedding_variables_names = None

    class Warning(OWDataProjectionWidget.Warning):
        missing_coords = Msg(
            "Plot cannot be displayed because '{}' or '{}' "
            "is missing for all data points.")

    class Information(OWDataProjectionWidget.Information):
        sampled_sql = Msg("Large SQL table; showing a sample.")
        missing_coords = Msg(
            "Points with missing '{}' or '{}' are not displayed")

    def __init__(self):
        self.attr_box: QGroupBox = None
        self.xy_model: DomainModel = None
        self.cb_attr_x: ComboBoxSearch = None
        self.cb_attr_y: ComboBoxSearch = None
        self.button_attr_x: QPushButton = None
        self.button_attr_y: QPushButton = None
        self.__x_axis_dlg: ErrorBarsDialog = None
        self.__y_axis_dlg: ErrorBarsDialog = None
        self.sampling: QGroupBox = None
        self._xy_invalidated: bool = True

        self.sql_data = None  # Orange.data.sql.table.SqlTable
        self.attribute_selection_list = None  # list of Orange.data.Variable
        self.__timer = QTimer(self, interval=1200)
        self.__timer.timeout.connect(self.add_data)
        super().__init__()

        # manually register Matplotlib file writers
        self.graph_writers = self.graph_writers.copy()
        for w in [MatplotlibFormat, MatplotlibPDFFormat]:
            self.graph_writers.append(w)

    def _add_controls(self):
        self._add_controls_axis()
        self._add_controls_sampling()
        super()._add_controls()
        self.gui.add_widget(self.gui.JitterNumericValues, self._effects_box)
        self.gui.add_widgets(
            [self.gui.ShowGridLines,
             self.gui.ToolTipShowsAll,
             self.gui.RegressionLine],
            self._plot_box)
        gui.checkBox(
            self._plot_box, self,
            value="graph.orthonormal_regression",
            label="Treat variables as independent",
            callback=self.graph.update_regression_line,
            tooltip=
            "If checked, fit line to group (minimize distance from points);\n"
            "otherwise fit y as a function of x (minimize vertical distances)",
            disabledBy=self.cb_reg_line)
        gui.checkBox(
            self._plot_box, self,
            value="graph.show_ellipse",
            label="Show confidence ellipse",
            tooltip="Hotelling's T² confidence ellipse (α=95%)",
            callback=self.graph.update_ellipse)

    def _add_controls_axis(self):
        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            contentsLength=12, searchable=True
        )
        self.attr_box = gui.vBox(self.controlArea, 'Axes',
                                 spacing=2 if gui.is_macstyle() else 8)
        dmod = DomainModel
        self.xy_model = DomainModel(dmod.MIXED, valid_types=dmod.PRIMITIVE)

        hor_icon, ver_icon = self.__get_bar_icons()
        width = 3 * QFontMetrics(self.font()).horizontalAdvance("m")
        hbox = gui.hBox(self.attr_box, spacing=0)
        self.cb_attr_x = gui.comboBox(
            hbox, self, "attr_x", label="Axis x:",
            callback=self.set_attr_from_combo,
            model=self.xy_model, **common_options,
        )
        self.button_attr_x = gui.button(
            hbox, self, "", callback=self.__on_x_button_clicked,
            autoDefault=False, width=width, enabled=False,
            sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed)
        )
        self.button_attr_x.setIcon(hor_icon)

        hbox = gui.hBox(self.attr_box, spacing=0)
        self.cb_attr_y = gui.comboBox(
            hbox, self, "attr_y", label="Axis y:",
            callback=self.set_attr_from_combo,
            model=self.xy_model, **common_options,
        )
        self.button_attr_y = gui.button(
            hbox, self, "", callback=self.__on_y_button_clicked,
            autoDefault=False, width=width, enabled=False,
            sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed)
        )
        self.button_attr_y.setIcon(ver_icon)

        vizrank_box = gui.hBox(self.attr_box)
        button = self.vizrank_button("Find Informative Projections")
        vizrank_box.layout().addWidget(button)
        self.vizrankSelectionChanged.connect(self.set_attr)

        self.__x_axis_dlg = ErrorBarsDialog(self)
        self.__x_axis_dlg.changed.connect(self.__on_x_dlg_changed)
        self.__y_axis_dlg = ErrorBarsDialog(self)
        self.__y_axis_dlg.changed.connect(self.__on_y_dlg_changed)

    def __on_x_button_clicked(self):
        self.__show_bars_dlg(
            self.__x_axis_dlg, self.button_attr_x,
            self.attr_x_upper, self.attr_x_lower, self.attr_x_is_abs)

    def __on_y_button_clicked(self):
        self.__show_bars_dlg(
            self.__y_axis_dlg, self.button_attr_y,
            self.attr_y_upper, self.attr_y_lower, self.attr_y_is_abs)

    def __show_bars_dlg(self, dlg, button, upper, lower, is_abs):
        pos = button.mapToGlobal(button.rect().bottomLeft())
        dlg.show_dlg(self.data.domain,
                     pos.x(), pos.y(),
                     upper, lower, is_abs)

    def __on_x_dlg_changed(self):
        self.attr_x_upper, self.attr_x_lower, self.attr_x_is_abs = \
            self.__x_axis_dlg.get_data()
        self.graph.update_error_bars()

    def __on_y_dlg_changed(self):
        self.attr_y_upper, self.attr_y_lower, self.attr_y_is_abs = \
            self.__y_axis_dlg.get_data()
        self.graph.update_error_bars()

    def _add_controls_sampling(self):
        self.sampling = gui.auto_commit(
            self.controlArea, self, "auto_sample", "Sample", box="Sampling",
            callback=self.switch_sampling, commit=lambda: self.add_data(1))
        self.sampling.setVisible(False)

    @property
    def effective_variables(self) -> list[Variable]:
        variables = []
        if self.attr_x and self.attr_y:
            variables.append(self.attr_x)
            if self.attr_x.name != self.attr_y.name:
                variables.append(self.attr_y)
        for var in (self.attr_x_upper, self.attr_x_lower,
                    self.attr_y_upper, self.attr_y_lower):
            # set is not used to preserve order
            if var and var not in variables:
                variables.append(var)
        return variables

    @property
    def effective_data(self):
        return self.data.transform(Domain(self.effective_variables))

    def init_vizrank(self):
        err_msg = ""
        if self.data is None:
            err_msg = "No data on input"
        elif self.data.is_sparse():
            err_msg = "Data is sparse"
        elif len(self.xy_model) < 3:
            err_msg = "Not enough features for ranking"
        elif self.attr_color is None:
            err_msg = "Color variable is not selected"
        elif np.isnan(self.data.get_column(self.attr_color)).all():
            err_msg = "Color variable has no values"
        if not err_msg:
            super().init_vizrank(self.data, list(self.xy_model), self.attr_color)
        else:
            self.disable_vizrank(err_msg)

    @OWDataProjectionWidget.Inputs.data
    def set_data(self, data):
        super().set_data(data)
        self.init_vizrank()

        def findvar(name, iterable):
            """Find a Orange.data.Variable in `iterable` by name"""
            for el in iterable:
                if isinstance(el, Variable) and el.name == name:
                    return el
            return None

        # handle restored settings from  < 3.3.9 when attr_* were stored
        # by name
        if isinstance(self.attr_x, str):
            self.attr_x = findvar(self.attr_x, self.xy_model)
        if isinstance(self.attr_y, str):
            self.attr_y = findvar(self.attr_y, self.xy_model)
        if isinstance(self.attr_label, str):
            self.attr_label = findvar(self.attr_label, self.gui.label_model)
        if isinstance(self.attr_color, str):
            self.attr_color = findvar(self.attr_color, self.gui.color_model)
        if isinstance(self.attr_shape, str):
            self.attr_shape = findvar(self.attr_shape, self.gui.shape_model)
        if isinstance(self.attr_size, str):
            self.attr_size = findvar(self.attr_size, self.gui.size_model)

    def check_data(self):
        super().check_data()
        self.__timer.stop()
        self.sampling.setVisible(False)
        self.sql_data = None
        if isinstance(self.data, SqlTable):
            if self.data.approx_len() < 4000:
                self.data = Table(self.data)
            else:
                self.Information.sampled_sql()
                self.sql_data = self.data
                data_sample = self.data.sample_time(0.8, no_cache=True)
                data_sample.download_data(2000, partial=True)
                self.data = Table(data_sample)
                self.sampling.setVisible(True)
                if self.auto_sample:
                    self.__timer.start()

        if self.data is not None and (len(self.data) == 0 or
                                      len(self.data.domain.variables) == 0):
            self.data = None

    def enable_controls(self):
        super().enable_controls()
        enabled = bool(self.data) and \
            self.data.domain.has_continuous_attributes(include_class=True,
                                                       include_metas=True)
        self.button_attr_x.setEnabled(enabled)
        self.button_attr_y.setEnabled(enabled)

    def get_embedding(self):
        self.valid_data = None
        if self.data is None:
            return None

        x_data = self.get_column(self.attr_x, filter_valid=False)
        y_data = self.get_column(self.attr_y, filter_valid=False)
        if x_data is None or y_data is None:
            return None

        self.Warning.missing_coords.clear()
        self.Information.missing_coords.clear()
        self.valid_data = np.isfinite(x_data) & np.isfinite(y_data)
        if self.valid_data is not None and not np.all(self.valid_data):
            msg = self.Information if np.any(self.valid_data) else self.Warning
            msg.missing_coords(self.attr_x.name, self.attr_y.name)
        return np.vstack((x_data, y_data)).T

    def get_errors_data(self) -> tuple[
        Optional[np.ndarray], Optional[np.ndarray],
        Optional[np.ndarray], Optional[np.ndarray]
    ]:
        x_data = self.get_column(self.attr_x)
        y_data = self.get_column(self.attr_y)
        top, bottom, left, right = [None] * 4
        if self.attr_x_upper:
            right = self.get_column(self.attr_x_upper)
            if self.attr_x_is_abs:
                right = right - x_data
        if self.attr_x_lower:
            left = self.get_column(self.attr_x_lower)
            if self.attr_x_is_abs:
                left = x_data - left
        if self.attr_y_upper:
            top = self.get_column(self.attr_y_upper)
            if self.attr_y_is_abs:
                top = top - y_data
        if self.attr_y_lower:
            bottom = self.get_column(self.attr_y_lower)
            if self.attr_y_is_abs:
                bottom = y_data - bottom
        return top, bottom, left, right

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        point_data = self.data[point_id]
        xy_attrs = (self.attr_x, self.attr_y)
        text = "<br/>".join(
            escape('{} = {}'.format(var.name, point_data[var]))
            for var in xy_attrs)
        if self.tooltip_shows_all:
            others = super()._point_tooltip(point_id, skip_attrs=xy_attrs)
            if others:
                text = "<b>{}</b><br/><br/>{}".format(text, others)
        return text

    def can_draw_regression_line(self):
        return self.data is not None and \
               self.data.domain is not None and \
               self.attr_x is not None and self.attr_y is not None and \
               self.attr_x.is_continuous and \
               self.attr_y.is_continuous

    def add_data(self, time=0.4):
        if self.data and len(self.data) > 2000:
            self.__timer.stop()
            return
        data_sample = self.sql_data.sample_time(time, no_cache=True)
        if data_sample:
            data_sample.download_data(2000, partial=True)
            data = Table(data_sample)
            self.data = Table.concatenate((self.data, data), axis=0)
            self.handleNewSignals()

    def init_attr_values(self):
        super().init_attr_values()
        data = self.data
        domain = data.domain if data and len(data) else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x
        self.attr_x_upper, self.attr_x_lower = None, None
        self.attr_y_upper, self.attr_y_lower = None, None

    def switch_sampling(self):
        self.__timer.stop()
        if self.auto_sample and self.sql_data:
            self.add_data()
            self.__timer.start()

    @OWDataProjectionWidget.Inputs.data_subset
    def set_subset_data(self, subset: Optional[Table]):
        self.warning()
        if isinstance(subset, SqlTable):
            if subset.approx_len() < AUTO_DL_LIMIT:
                subset = Table(subset)
            else:
                self.warning("Data subset does not support large Sql tables")
                subset = None
        super().set_subset_data(subset)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        self.attr_box.setEnabled(True)
        if self.attribute_selection_list and self.data is not None and \
                self.data.domain is not None:
            self.attr_box.setEnabled(False)
            if all(attr in self.xy_model for attr in self.attribute_selection_list):
                self.attr_x, self.attr_y = self.attribute_selection_list
            else:
                self.attr_x, self.attr_y = None, None
            self.attr_x_upper, self.attr_x_lower = None, None
            self.attr_y_upper, self.attr_y_lower = None, None
        self._invalidated = self._invalidated or self._xy_invalidated
        self._xy_invalidated = False
        super().handleNewSignals()
        if self._domain_invalidated:
            self.graph.update_axes()
            self.graph.update_error_bars()
            self._domain_invalidated = False
        if self.attribute_selection_list:
            self.graph.update_error_bars()
        can_plot = self.can_draw_regression_line()
        self.cb_reg_line.setEnabled(can_plot)
        self.graph.controls.show_ellipse.setEnabled(can_plot)

    @Inputs.features
    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
            self._xy_invalidated = self._xy_invalidated \
                or self.attr_x != attributes[0] \
                or self.attr_y != attributes[1]
        else:
            if self.attr_x is None or self.attr_y is None:
                # scenario happens when features input removed and features
                # were invalid or hidden and those attr_x and attr_h were None
                self.init_attr_values()
            self.attribute_selection_list = None

    def set_attr(self, attrs):
        if attrs != [self.attr_x,self.attr_y]:
            self.attr_x, self.attr_y = attrs
            self.attr_changed()

    def set_attr_from_combo(self):
        self.attr_changed()
        self.vizrankAutoSelect.emit([self.attr_x, self.attr_y])

    def attr_changed(self):
        can_plot = self.can_draw_regression_line()
        self.cb_reg_line.setEnabled(can_plot)
        self.graph.controls.show_ellipse.setEnabled(can_plot)
        self.setup_plot()
        self.commit.deferred()

    def get_axes(self):
        return {"bottom": self.attr_x, "left": self.attr_y}

    def colors_changed(self):
        super().colors_changed()
        self.init_vizrank()

    @gui.deferred
    def commit(self):
        super().commit()
        self.send_features()

    def send_features(self):
        features = [attr for attr in [self.attr_x, self.attr_y] if attr]
        self.Outputs.features.send(AttributeList(features) or None)

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attr_x.name, self.attr_y.name)
        return None

    def _get_send_report_caption(self):
        return report.render_items_vert((
            ("Color", self._get_caption_var_name(self.attr_color)),
            ("Label", self._get_caption_var_name(self.attr_label)),
            ("Shape", self._get_caption_var_name(self.attr_shape)),
            ("Size", self._get_caption_var_name(self.attr_size)),
            ("Jittering", (self.attr_x.is_discrete or
                           self.attr_y.is_discrete or
                           self.graph.jitter_continuous) and
             self.graph.jitter_size)))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2 and "selection" in settings and settings["selection"]:
            settings["selection_group"] = [(a, 1) for a in settings["selection"]]
        if version < 3:
            if "auto_send_selection" in settings:
                settings["auto_commit"] = settings["auto_send_selection"]
            if "selection_group" in settings:
                settings["selection"] = settings["selection_group"]
        if version < 5:
            if "graph" in settings and \
                    "jitter_continuous" not in settings["graph"]:
                settings["graph"]["jitter_continuous"] = True

    @classmethod
    def migrate_context(cls, context, version):
        values = context.values
        if version < 3:
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]
        if version < 4:
            if values["attr_x"][1] % 100 == 1 or values["attr_y"][1] % 100 == 1:
                raise IncompatibleContext()

    __HorizontalBarIcon = None
    __VerticalBarIcon = None

    @classmethod
    def __get_bar_icons(cls):
        if cls.__HorizontalBarIcon is None:
            cls.__HorizontalBarIcon = load_styled_icon(
                "Orange.widgets.visualize", "icons/interval-horizontal.svg")
            cls.__VerticalBarIcon = load_styled_icon(
                "Orange.widgets.visualize", "icons/interval-vertical.svg")
        return cls.__HorizontalBarIcon, cls.__VerticalBarIcon


if __name__ == "__main__":  # pragma: no cover
    table = Table("iris")
    WidgetPreview(OWScatterPlot).run(set_data=table)#,
                                     #set_subset_data=table[:30])
