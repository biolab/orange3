from functools import partial
import collections
from xml.sax.saxutils import escape

import numpy as np
import pyqtgraph as pg

from AnyQt.QtWidgets import QSizePolicy, QLabel, QListView, QToolTip
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QPicture, QPalette
from AnyQt.QtCore import Qt, QRectF
from scipy.stats import norm, rayleigh, beta, gamma, pareto, expon

import Orange.data
from Orange.statistics import distribution, contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input

from Orange.widgets.visualize.owscatterplotgraph import \
    LegendItem as SPGLegendItem
from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate


def selected_index(view):
    indices = view.selectedIndexes()
    assert len(indices) < 2, "View must be in single selection mode"
    if indices:
        return indices[0].row()
    else:
        return -1


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pg.graphicsItems.ScatterPlotItem.Symbols

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


class LegendItem(SPGLegendItem):
    def __init__(self):
        super().__init__()
        self.items = []

    def clear(self):
        items = list(self.items)
        self.items = []
        for sample, label in items:
            # yes, the LegendItem shadows QGraphicsWidget.layout() with
            # an instance attribute.
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
            sample.hide()
            label.hide()

        self.updateSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            event.accept()
            if self.parentItem() is not None:
                self.autoAnchor(
                    self.pos() + (event.pos() - event.lastPos()) / 2)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()


class DistributionBarItem(pg.GraphicsObject):
    def __init__(self, x, width, dist, colors, padding=0,
                 stacked=True, expanded=False):
        super().__init__()
        self.x = x
        self.width = width
        self.dist = dist
        self.colors = colors
        self.padding = padding
        self.stacked = stacked
        self.expanded = expanded
        self.__picture = None

    def paint(self, painter, options, widget):
        if self.__picture is None:
            self.__paint()
        painter.drawPicture(0, 0, self.__picture)

    def boundingRect(self):
        if self.expanded:
            height = 1
        elif self.stacked:
            height = sum(self.dist)
        else:
            height = max(self.dist)
        return QRectF(self.x, 0, self.width, height)

    def __paint(self):
        picture = self.__picture = QPicture()
        if self.expanded:
            tot = np.sum(self.dist)
            if tot == 0:
                return
            dist = self.dist / tot
        else:
            dist = self.dist

        painter = QPainter(picture)

        coord_padding = self.mapRectFromDevice(
            QRectF(0, 0, self.padding, 0)).width()
        sx = self.x + coord_padding
        padded_width = self.width - 2 * coord_padding

        if self.stacked:
            painter.setPen(Qt.NoPen)
            y = 0
            for freq, color in zip(dist, self.colors):
                painter.setBrush(QBrush(color))
                painter.drawRect(QRectF(sx, y, padded_width, freq))
                y += freq
        else:
            pen = QPen(QBrush(Qt.white), 0.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            wsingle = padded_width / len(self.dist)
            for i, (freq, color) in enumerate(zip(dist, self.colors)):
                painter.setBrush(QBrush(color))
                painter.drawRect(
                    QRectF(sx + wsingle * i, 0, wsingle, freq))

        painter.end()


class OWDistributions(widget.OWWidget):
    name = "Distributions"
    description = "Display value distributions of a data feature in a graph."
    icon = "icons/Distribution.svg"
    priority = 120
    keywords = []

    class Inputs:
        data = Input("Data", Orange.data.Table, doc="Set the input dataset")

    settingsHandler = settings.DomainContextHandler(
        match_values=settings.DomainContextHandler.MATCH_VALUES_ALL)
    variable_idx = settings.ContextSetting(-1)
    groupvar_idx = settings.ContextSetting(0)

    stacked_columns = settings.Setting(True)
    cumulative_distr = settings.Setting(False)

    number_of_bins = settings.Setting(5)
    fitted_distribution = settings.Setting(0)
    show_probs = settings.ContextSetting(True)

    graph_name = "plot"

    bins = [2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50]

    Fitters = (
        ("None", None, ()),
        ("Normal", norm, ("loc", "scale")),
        ("Beta", beta, ("a", "b", "loc", "scale")),
        ("Gamma", gamma, ("a", "loc", "scale")),
        ("Rayleigh", rayleigh, ("loc", "scale")),
        ("Pareto", pareto, ("b", "loc", "scale")),
        ("Exponential", expon, ("loc", "scale")),
    )

    def __init__(self):
        super().__init__()
        self.data = None

        self.distributions = None
        self.contingencies = None
        self.var = self.cvar = None
        varbox = gui.vBox(self.controlArea, "Variable")

        self.varmodel = itemmodels.VariableListModel()
        self.groupvarmodel = []

        self.varview = QListView(
            selectionMode=QListView.SingleSelection,
            uniformItemSizes=True,
        )
        self.varview.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.varview.setModel(self.varmodel)
        self.varview.setSelectionModel(
            itemmodels.ListSingleSelectionModel(self.varmodel))
        self.varview.selectionModel().selectionChanged.connect(
            self._on_variable_idx_changed)
        varbox.layout().addWidget(self.varview)

        box = self.continuous_box = gui.vBox(self.controlArea, "Distribution")
        gui.hSlider(
            box, self, "number_of_bins",
            label="Number of bins", orientation=Qt.Horizontal,
            minValue=0, maxValue=len(self.bins) - 1,
            callback=self._on_set_bins, createLabel=False)
        gui.comboBox(
            box, self, "fitted_distribution", label="Fitted distribution",
            orientation=Qt.Horizontal, items=(name[0] for name in self.Fitters),
            callback=self._setup)
        gui.checkBox(
            box, self, "cumulative_distr", "Show cumulative distribution",
            callback=self._on_cumulative_distr_changed,
            tooltip="Show the cumulative distribution function.")

        box = gui.vBox(self.controlArea, "Columns")
        self.icons = gui.attributeIconDict
        gui.comboBox(
            box, self, "groupvar_idx",
            label="Split by", orientation=Qt.Horizontal,
            callback=self._on_groupvar_idx_changed,
            valueType=str, contentsLength=18)
        box2 = gui.indentedBox(box, sep=4)
        gui.checkBox(
            box2, self, "stacked_columns", "Stack columns",
            callback=self._on_stacked_changed)
        gui.checkBox(
            box2, self, "show_probs", "Show probabilities",
            callback=self._on_relative_freq_changed)

        self.plotview = pg.PlotWidget(background=None)
        self.plotview.setRenderHint(QPainter.Antialiasing)
        self.mainArea.layout().addWidget(self.plotview)
        w = QLabel()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mainArea.layout().addWidget(w, Qt.AlignCenter)
        self.ploti = pg.PlotItem()
        self.plot = self.ploti.vb
        self.ploti.hideButtons()
        self.plotview.setCentralItem(self.ploti)

        self.plot_pdf = pg.ViewBox()
        self.ploti.hideAxis('right')
        self.ploti.scene().addItem(self.plot_pdf)
        axis = pg.AxisItem("right")
        axis.linkToView(self.plot_pdf)
        self.plot_pdf.setZValue(10)
        self.plot_pdf.setXLink(self.ploti)
        self.ploti.vb.sigResized.connect(self.update_views)

        self.plot_prob = pg.ViewBox()
        self.ploti.scene().addItem(self.plot_prob)
        axis = self.ploti.getAxis("right")
        axis.linkToView(self.plot_prob)
        axis.setLabel("Probability")
        self.plot_prob.setZValue(15)
        self.plot_prob.setXLink(self.plot_prob)

        self.update_views()
        self.ploti.vb.sigResized.connect(self.update_views)

        def disable_mouse(plot):
            plot.setMouseEnabled(False, False)
            plot.setMenuEnabled(False)

        disable_mouse(self.plot)
        disable_mouse(self.plot_pdf)
        disable_mouse(self.plot_prob)

        self.tooltip_items = []
        self.plot.scene().installEventFilter(
            HelpEventDelegate(self.help_event, self))

        pen = QPen(self.palette().color(QPalette.Text))
        for axis in ("left", "bottom"):
            self.ploti.getAxis(axis).setPen(pen)

        self._legend = LegendItem()
        self._legend.setParentItem(self.plot)
        self._legend.hide()
        self._legend.anchor((1, 0), (1, 0))

    def update_views(self):
        self.plot_pdf.setGeometry(self.plot.sceneBoundingRect())
        self.plot_pdf.linkedViewChanged(self.plot, self.plot_pdf.XAxis)
        self.plot_prob.setGeometry(self.plot.sceneBoundingRect())
        self.plot_prob.linkedViewChanged(self.plot, self.plot_prob.XAxis)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.warning()
        self.data = data
        self.distributions = None
        self.contingencies = None
        if self.data is not None:
            if not self.data:
                self.warning("Empty input data cannot be visualized")
                return
            domain = self.data.domain
            self.varmodel[:] = list(domain.variables) + \
                               [meta for meta in domain.metas
                                if meta.is_continuous or meta.is_discrete]
            groupvarview = self.controls.groupvar_idx
            groupvarview.clear()
            self.groupvarmodel = \
                ["(None)"] + [var for var in domain.variables if var.is_discrete] + \
                [meta for meta in domain.metas if meta.is_discrete]
            groupvarview.addItem("(None)")
            for var in self.groupvarmodel[1:]:
                groupvarview.addItem(self.icons[var], var.name)
            if domain.has_discrete_class:
                self.groupvar_idx = \
                    self.groupvarmodel[1:].index(domain.class_var) + 1
            self.openContext(domain)
            self.variable_idx = min(max(self.variable_idx, 0),
                                    len(self.varmodel) - 1)
            self.groupvar_idx = min(max(self.groupvar_idx, 0),
                                    len(self.groupvarmodel) - 1)
            itemmodels.select_row(self.varview, self.variable_idx)
            self._setup()

    def clear(self):
        self.plot.clear()
        self.plot_pdf.clear()
        self.plot_prob.clear()
        self.varmodel[:] = []
        self.groupvarmodel = []
        self.variable_idx = -1
        self.groupvar_idx = 0
        self._legend.clear()
        self._legend.hide()
        self.controls.groupvar_idx.clear()

    def _setup(self):
        self.plot.clear()
        self.plot_pdf.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self._legend.hide()

        self.tooltip_items = []

        varidx = self.variable_idx
        self.var = self.cvar = None
        if varidx >= 0:
            self.var = self.varmodel[varidx]
        data = self.data
        if self.groupvar_idx > 0:
            self.cvar = self.groupvarmodel[self.groupvar_idx]
        if self.var is None:
            return
        column = self.data.get_column_view(self.var)[0]
        self.valid_mask = ~np.isnan(column)
        if self.cvar:
            ccolumn = self.data.get_column_view(self.cvar)[0]
            self.valid_mask *= ~np.isnan(ccolumn)
            self.valid_group_data = ccolumn[self.valid_mask]
        else:
            self.valid_group_data = None
        self.valid_data = column[self.valid_mask]

        self.set_left_axis_name()
        self.distributions = self.contingencies = None
        if self.var.is_discrete:
            self.continuous_box.hide()
            if self.cvar:
                self.contingencies = \
                    contingency.get_contingency(data, self.cvar, self.var)
            else:
                self.distributions = \
                    distribution.get_distribution(data, self.var)
        else:
            self.continuous_box.show()

        bottomaxis = self.ploti.getAxis("bottom")
        bottomaxis.setLabel(self.var.name)
        bottomaxis.resizeEvent()

        if self.cvar:
            self.display_contingency()
        else:
            self.display_distribution()
        self.plot.autoRange()

    def help_event(self, ev):
        self.plot.mapSceneToView(ev.scenePos())
        ctooltip = []
        for vb, item in self.tooltip_items:
            mouse_over_curve = isinstance(item, pg.PlotCurveItem) \
                and item.mouseShape().contains(vb.mapSceneToView(ev.scenePos()))
            mouse_over_bar = isinstance(item, DistributionBarItem) \
                and item.boundingRect().contains(vb.mapSceneToView(ev.scenePos()))
            if mouse_over_curve or mouse_over_bar:
                ctooltip.append(item.tooltip)
        if ctooltip:
            QToolTip.showText(ev.screenPos(), "\n\n".join(ctooltip), widget=self.plotview)
            return True
        return False

    def disc_dist_histogram(self):
        var = self.var
        self.ploti.getAxis("bottom").setTicks([list(enumerate(var.values))])
        colors = [QColor(0, 128, 255)]
        for i, freq in enumerate(self.distributions):
            item = DistributionBarItem(i - 0.5, 1, [freq], colors, padding=20)
            self.plot.addItem(item)
            item.tooltip = "Frequency for %s: %r" % (var.values[i], freq)
            self.tooltip_items.append((self.plot, item))

    def disc_cont_histogram(self):
        var = self.var
        self.ploti.getAxis("bottom").setTicks([list(enumerate(var.values))])
        gcolors = [QColor(*col) for col in self.cvar.colors]
        for i, cont in enumerate(self.contingencies):
            item = DistributionBarItem(i - 0.5, 1, cont, gcolors,
                                       padding=20, stacked=self.stacked_columns,
                                       expanded=self.show_probs)
            self.plot.addItem(item)
#            item.tooltip = "Frequency for %s: %r" % (var.values[i], freq)
            self.tooltip_items.append((self.plot, item))

    def cont_dist_histogram(self):
        self.ploti.getAxis("bottom").setTicks(None)
        data = self.valid_data
        y, x = np.histogram(data, bins=self.bins[self.number_of_bins])
        total = len(data)
        colors = [QColor(0, 128, 255).lighter(130 if self.fitted_distribution else 100)]
        tot_freq = 0
        for (x0, x1), freq in zip(zip(x, x[1:]), y):
            tot_freq += freq
            item = DistributionBarItem(
                x0, x1 - x0, [tot_freq if self.cumulative_distr else freq],
                colors, padding=0.5)
            self.plot.addItem(item)
            item.tooltip = self._str_int(x0, x1) \
                           + f"{freq} ({100 * freq / total:.2f} %)"
            self.tooltip_items.append((self.plot, item))

        if self.fitted_distribution:
            self._plot_approximations(
                x[0], x[-1], [self._fit_approximation(data)], [QColor(0, 0, 0)], (1,))

    def cont_cont_histogram(self):
        self.ploti.getAxis("bottom").setTicks(None)
        data = self.valid_data
        _, bins = np.histogram(data, bins=self.bins[self.number_of_bins])
        gvalues = self.cvar.values
        varcolors = [QColor(*col) for col in self.cvar.colors]
        if self.fitted_distribution:
            gcolors = [c.lighter(130) for c in varcolors]
        else:
            gcolors = varcolors
        nvalues = len(gvalues)
        ys = []
        fitters = []
        prior_sizes = []
        for val_idx in range(nvalues):
            group_data = data[self.valid_group_data == val_idx]
            prior_sizes.append(len(group_data))
            ys.append(np.histogram(group_data, bins)[0])
            if self.fitted_distribution:
                fitters.append(self._fit_approximation(group_data))
        total = len(data)
        prior_sizes = np.array(prior_sizes)
        tot_freqs = np.zeros(len(ys))

        for x0, x1, freqs in zip(bins, bins[1:], zip(*ys)):
            tot_freqs += freqs
            plotfreqs = tot_freqs.copy() if self.cumulative_distr else freqs
            item = DistributionBarItem(
                x0, x1 - x0, plotfreqs, gcolors,
                padding=0.5 if self.stacked_columns else 6,
                stacked=self.stacked_columns, expanded=self.show_probs)
            self.plot.addItem(item)
            if self.show_probs:
                total = np.sum(plotfreqs)
            if total == 0:
                total = 1
            item.tooltip = \
                self._str_int(x0, x1) + \
                "".join(f"\n - {value}: {freq} ({100 * freq / total:.2f} %)"
                        for value, freq in zip(gvalues, plotfreqs))
            self.tooltip_items.append((self.plot, item))

        if fitters:
            self._plot_approximations(bins[0], bins[-1], fitters, varcolors,
                                      prior_sizes / len(data))

    def _plot_approximations(self, x0, x1, fitters, colors, prior_probs=0):
        x = np.linspace(x0, x1, 100)
        ys = np.empty((len(fitters), 100))
        for y, fitter, color in zip(ys, fitters, colors):
            y[:] = fitter(x)
            if self.cumulative_distr:
                y[:] = np.cumsum(y)
        tots = np.sum(ys, axis=0)

        show_probs = self.show_probs and self.cvar is not None
        plot = self.ploti if show_probs else self.plot_pdf

        for y, prior_prob, color in zip(ys, prior_probs, colors):
            if show_probs:
                y_p = y * prior_prob
                y = y_p / (y_p + (tots - y) * (1 - prior_prob))
            plot.addItem(pg.PlotCurveItem(
                x=x, y=y,
                pen=pg.mkPen(width=5, color=color),
                shadowPen=pg.mkPen(width=8, color=color.darker(120))))
        self.plot_pdf.autoRange()
        self.plot_prob.autoRange()

    def _str_int(self, x0, x1):
        var = self.var
        if self.cumulative_distr:
            return f"{var.name} < {var.repr_val(x1)}"
        else:
            return f"{var.name} in {var.repr_val(x0)} - {var.repr_val(x1)}:"

    def _fit_approximation(self, y):
        _, distribution, names = self.Fitters[self.fitted_distribution]
        params = {name: val for name, val in zip(names, distribution.fit(y))}
        return partial(distribution.pdf, **params)

    def display_distribution(self):
        var = self.var

        self.set_left_axis_name()
        if var.is_continuous:
            self.cont_dist_histogram()
        else:
            dist = self.distributions
            if dist is None or not len(dist):
                return
            self.disc_dist_histogram()

    def _enable_show_probabilities(self):
        self.controls.show_probs.setDisabled(
            self.var is None or self.cvar is None)

    def _on_stacked_changed(self):
        self._setup()

    def _on_relative_freq_changed(self):
        self._setup()

    def display_contingency(self):
        if self.var.is_continuous:
            self.cont_cont_histogram()
        else:
            self.disc_cont_histogram()

        cvar_values = self.cvar.values
        colors = [QColor(*col) for col in self.cvar.colors]
        for color, name in zip(colors, cvar_values):
            self._legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, shape="s"),
                escape(name)
            )
        self._legend.show()

    def set_left_axis_name(self):
        leftaxis = self.ploti.getAxis("left")
        if self.show_probs and self.var is not None and self.cvar is not None:
            leftaxis.setLabel(
                f"Probability of '{self.cvar.name}' at given '{self.var.name}'")
        else:
            leftaxis.setLabel("Frequency")
        leftaxis.resizeEvent()

    def _on_variable_idx_changed(self):
        self.variable_idx = selected_index(self.varview)
        self._enable_show_probabilities()
        self._setup()

    def _on_groupvar_idx_changed(self):
        self._setup()
        self._enable_show_probabilities()

    def _on_set_bins(self):
        self._setup()

    def _on_cumulative_distr_changed(self):
        self._setup()

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()

    def get_widget_name_extension(self):
        if self.variable_idx >= 0:
            return self.varmodel[self.variable_idx]
        return None

    def send_report(self):
        self.plotview.scene().setSceneRect(self.plotview.sceneRect())
        if self.variable_idx < 0:
            return
        self.report_plot()
        text = "Distribution of '{}'".format(
            self.varmodel[self.variable_idx])
        if self.groupvar_idx:
            group_var = self.groupvarmodel[self.groupvar_idx]
            text += " grouped by '{}'".format(group_var)
            if self.relative_freq:
                text += " (relative frequencies)"
        self.report_caption(text)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDistributions).run(Orange.data.Table("heart_disease.tab"))
