"""
Distributions
-------------

A widget for plotting attribute distributions.

"""
from itertools import chain
from math import sqrt
import sys
import collections
from xml.sax.saxutils import escape

from AnyQt.QtWidgets import QSizePolicy, QLabel, QListView, QToolTip
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QPicture, QPalette
from AnyQt.QtCore import Qt, QRectF

import numpy as np
import pyqtgraph as pg

import Orange.data
from Orange.data import DiscreteVariable
from Orange.preprocess import Discretize, EqualWidth
from Orange.statistics import distribution, contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input
from Orange.widgets.visualize.owlinearprojection import LegendItem, ScatterPlotItem

from Orange.widgets.visualize.owscatterplotgraph import HelpEventDelegate


def selected_variable(view):
    """Return the selected Variable `var` (row) in the view.

    If no index is selected return None

    `view` must be in single selection mode.
    """
    indices = view.selectedIndexes()
    assert len(indices) < 2, "View must be in single selection mode"
    if indices:
        return indices[0].data(Qt.EditRole)
    return None


class DistributionBarItem(pg.GraphicsObject):
    def __init__(self, geometry, dist, colors):
        super().__init__()
        self.geometry = geometry
        self.dist = dist
        self.colors = colors
        self.__picture = None

    def paint(self, painter, options, widget):
        if self.__picture is None:
            self.__paint()
        painter.drawPicture(0, 0, self.__picture)

    def boundingRect(self):
        return self.geometry

    def __paint(self):
        picture = QPicture()
        painter = QPainter(picture)
        pen = QPen(QBrush(Qt.white), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)

        geom = self.geometry
        x, y = geom.x(), geom.y()
        w, h = geom.width(), geom.height()
        wsingle = w / len(self.dist)
        for d, c in zip(self.dist, self.colors):
            painter.setBrush(QBrush(c))
            painter.drawRect(QRectF(x, y, wsingle, d * h))
            x += wsingle
        painter.end()

        self.__picture = picture


class OWDistributions(widget.OWWidget):
    name = "Distributions"
    description = "Display value distributions of a data feature in a graph."
    icon = "icons/Distribution.svg"
    priority = 120

    class Inputs:
        data = Input("Data", Orange.data.Table, doc="Set the input data set")

    settingsHandler = settings.DomainContextHandler(
        match_values=settings.DomainContextHandler.MATCH_VALUES_ALL)
    #: Selected variable index
    var = settings.ContextSetting(None)
    #: Selected group variable
    group_var = settings.ContextSetting(None)

    relative_freq = settings.Setting(False)
    disc_cont = settings.Setting(False)

    smoothing_index = settings.Setting(5)
    show_prob = settings.ContextSetting(0)

    graph_name = "plot"

    ASH_HIST = 50

    bins = [2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50]
    smoothing_facs = list(reversed([0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 4, 6, 10]))

    def __init__(self):
        super().__init__()
        self.data = None

        self.distributions = None
        self.contingencies = None
        varbox = gui.vBox(self.controlArea, "Variable")

        self.varmodel = DomainModel(
            order=(DomainModel.ATTRIBUTES, DomainModel.CLASSES,
                   DomainModel.METAS), valid_types=DomainModel.PRIMITIVE)
        self.groupvarmodel = DomainModel(placeholder="(None)", valid_types=DiscreteVariable)

        self.varview = QListView(
            selectionMode=QListView.SingleSelection)
        self.varview.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.varview.setModel(self.varmodel)
        self.varview.setSelectionModel(
            itemmodels.ListSingleSelectionModel(self.varmodel))
        self.varview.selectionModel().selectionChanged.connect(
            self._on_variable_changed)
        varbox.layout().addWidget(self.varview)

        box = gui.vBox(self.controlArea, "Precision")

        gui.separator(self.controlArea, 4, 4)

        box2 = gui.hBox(box)
        self.l_smoothing_l = gui.widgetLabel(box2, "Smooth")
        gui.hSlider(box2, self, "smoothing_index",
                    minValue=0, maxValue=len(self.smoothing_facs) - 1,
                    callback=self._on_set_smoothing, createLabel=False)
        self.l_smoothing_r = gui.widgetLabel(box2, "Precise")

        self.cb_disc_cont = gui.checkBox(
            gui.indentedBox(box, sep=4),
            self, "disc_cont", "Bin numeric variables",
            callback=self._on_groupvar_changed,
            tooltip="Show numeric variables as categorical.")

        box = gui.vBox(self.controlArea, "Group by")
        self.icons = gui.attributeIconDict
        self.groupvarview = gui.comboBox(
            box, self, "group_var",
            callback=self._on_groupvar_changed,
            contentsLength=12, model=self.groupvarmodel)
        box2 = gui.indentedBox(box, sep=4)
        self.cb_rel_freq = gui.checkBox(
            box2, self, "relative_freq", "Show relative frequencies",
            callback=self._on_relative_freq_changed,
            tooltip="Normalize probabilities so that probabilities "
                    "for each group-by value sum to 1.")
        gui.separator(box2)
        self.cb_prob = gui.comboBox(
            box2, self, "show_prob", label="Show probabilities:",
            orientation=Qt.Horizontal,
            callback=self._on_relative_freq_changed,
            tooltip="Show probabilities for a chosen group-by value "
                    "(at each point probabilities for all group-by values sum to 1).")

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

        self.plot_prob = pg.ViewBox()
        self.ploti.hideAxis('right')
        self.ploti.scene().addItem(self.plot_prob)
        self.ploti.getAxis("right").linkToView(self.plot_prob)
        self.ploti.getAxis("right").setLabel("Probability")
        self.plot_prob.setZValue(10)
        self.plot_prob.setXLink(self.ploti)
        self.update_views()
        self.ploti.vb.sigResized.connect(self.update_views)
        self.plot_prob.setRange(yRange=[0, 1])

        def disable_mouse(plot):
            plot.setMouseEnabled(False, False)
            plot.setMenuEnabled(False)

        disable_mouse(self.plot)
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
            self.varmodel.set_domain(domain)
            self.groupvarview.clear()
            self.groupvarmodel.set_domain(domain)
            if self.varmodel:
                self.var = self.varmodel[0]
            if domain.class_var in self.groupvarmodel:
                self.group_var = domain.class_var
            self.openContext(domain)
            # See the comment in migrate_context
            self.migrate_context(None, None, self=self, domain=domain)
            itemmodels.select_row(self.varview, self.varmodel.indexOf(self.var))
            self._setup()

    def clear(self):
        self.plot.clear()
        self.plot_prob.clear()
        self.varmodel.set_domain(None)
        self.groupvarmodel.set_domain(None)
        self.var = None
        self.group_var = None
        self._legend.clear()
        self._legend.hide()
        self.groupvarview.clear()
        self.cb_prob.clear()

    def _setup_smoothing(self):
        if not self.disc_cont and self.var and self.var.is_continuous:
            self.cb_disc_cont.setText("Bin numeric variables")
            self.l_smoothing_l.setText("Smooth")
            self.l_smoothing_r.setText("Precise")
        else:
            self.cb_disc_cont.setText("Bin numeric variables into {} bins".
                                      format(self.bins[self.smoothing_index]))
            self.l_smoothing_l.setText(" " + str(self.bins[0]))
            self.l_smoothing_r.setText(" " + str(self.bins[-1]))

    @property
    def smoothing_factor(self):
        return self.smoothing_facs[self.smoothing_index]

    def _setup(self):
        self.plot.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self._legend.hide()

        if self.group_var is not None:
            self.cb_prob.clear()
            self.cb_prob.addItem("(None)")
            self.cb_prob.addItems(self.group_var.values)
            self.cb_prob.addItem("(All)")
            self.show_prob = min(max(self.show_prob, 0),
                                 len(self.group_var.values) + 1)
        data = self.data
        self._setup_smoothing()
        if self.var is None:
            return
        if self.disc_cont:
            domain = Orange.data.Domain(
                [self.var, self.group_var] if self.group_var else [self.var])
            data = Orange.data.Table(domain, data)
            disc = EqualWidth(n=self.bins[self.smoothing_index])
            data = Discretize(method=disc, remove_const=False)(data)
            self.var = data.domain[0]
        self.set_left_axis_name()
        self.enable_disable_rel_freq()
        if self.group_var:
            self.contingencies = \
                contingency.get_contingency(data, self.var, self.group_var)
            self.display_contingency()
        else:
            self.distributions = \
                distribution.get_distribution(data, self.var)
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

    def display_distribution(self):
        dist = self.distributions
        var = self.var
        if dist is None or not len(dist):
            return
        self.plot.clear()
        self.plot_prob.clear()
        self.ploti.hideAxis('right')
        self.tooltip_items = []

        bottomaxis = self.ploti.getAxis("bottom")
        bottomaxis.setLabel(var.name)
        bottomaxis.resizeEvent()

        self.set_left_axis_name()
        if var and var.is_continuous:
            bottomaxis.setTicks(None)
            if not len(dist[0]):
                return
            edges, curve = ash_curve(dist, None, m=OWDistributions.ASH_HIST,
                                     smoothing_factor=self.smoothing_factor)
            edges = edges + (edges[1] - edges[0])/2
            edges = edges[:-1]
            item = pg.PlotCurveItem()
            pen = QPen(QBrush(Qt.white), 3)
            pen.setCosmetic(True)
            item.setData(edges, curve, antialias=True, stepMode=False,
                         fillLevel=0, brush=QBrush(Qt.gray), pen=pen)
            self.plot.addItem(item)
            item.tooltip = "Density"
            self.tooltip_items.append((self.plot, item))
        else:
            bottomaxis.setTicks([list(enumerate(var.values))])
            for i, w in enumerate(dist):
                geom = QRectF(i - 0.33, 0, 0.66, w)
                item = DistributionBarItem(geom, [1.0],
                                           [QColor(128, 128, 128)])
                self.plot.addItem(item)
                item.tooltip = "Frequency for %s: %r" % (var.values[i], w)
                self.tooltip_items.append((self.plot, item))

    def _on_relative_freq_changed(self):
        self.set_left_axis_name()
        if self.group_var and self.group_var.is_discrete:
            self.display_contingency()
        else:
            self.display_distribution()
        self.plot.autoRange()

    def display_contingency(self):
        """
        Set the contingency to display.
        """
        cont = self.contingencies
        var, group_var = self.var, self.group_var
        if cont is None or not len(cont):
            return
        self.plot.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self.tooltip_items = []

        if self.show_prob:
            self.ploti.showAxis('right')
        else:
            self.ploti.hideAxis('right')

        bottomaxis = self.ploti.getAxis("bottom")
        bottomaxis.setLabel(var.name)
        bottomaxis.resizeEvent()

        group_var_values = group_var.values
        colors = [QColor(*col) for col in group_var.colors]

        if var and var.is_continuous:
            bottomaxis.setTicks(None)

            weights, cols, group_var_values, curves = [], [], [], []
            for i, dist in enumerate(cont):
                v, W = dist
                if len(v):
                    weights.append(np.sum(W))
                    cols.append(colors[i])
                    group_var_values.append(group_var.values[i])
                    curves.append(ash_curve(
                        dist, cont, m=OWDistributions.ASH_HIST,
                        smoothing_factor=self.smoothing_factor))
            weights = np.array(weights)
            sumw = np.sum(weights)
            weights /= sumw
            colors = cols
            curves = [(X, Y * w) for (X, Y), w in zip(curves, weights)]

            curvesline = [] #from histograms to lines
            for X, Y in curves:
                X = X + (X[1] - X[0])/2
                X = X[:-1]
                X = np.array(X)
                Y = np.array(Y)
                curvesline.append((X, Y))

            for t in ["fill", "line"]:
                curve_data = list(zip(curvesline, colors, weights, group_var_values))
                for (X, Y), color, w, cval in reversed(curve_data):
                    item = pg.PlotCurveItem()
                    pen = QPen(QBrush(color), 3)
                    pen.setCosmetic(True)
                    color = QColor(color)
                    color.setAlphaF(0.2)
                    item.setData(X, Y/(w if self.relative_freq else 1),
                                 antialias=True, stepMode=False,
                                 fillLevel=0 if t == "fill" else None,
                                 brush=QBrush(color), pen=pen)
                    self.plot.addItem(item)
                    if t == "line":
                        item.tooltip = "{}\n{}={}".format(
                            "Normalized density " if self.relative_freq else "Density ",
                            group_var.name, cval)
                        self.tooltip_items.append((self.plot, item))

            if self.show_prob:
                all_X = np.array(np.unique(np.hstack([X for X, _ in curvesline])))
                inter_X = np.array(np.linspace(all_X[0], all_X[-1], len(all_X)*2))
                curvesinterp = [np.interp(inter_X, X, Y) for (X, Y) in curvesline]
                sumprob = np.sum(curvesinterp, axis=0)
                legal = sumprob > 0.05 * np.max(sumprob)

                i = len(curvesinterp) + 1
                show_all = self.show_prob == i
                for Y, color, cval in reversed(list(zip(curvesinterp, colors, group_var_values))):
                    i -= 1
                    if show_all or self.show_prob == i:
                        item = pg.PlotCurveItem()
                        pen = QPen(QBrush(color), 3, style=Qt.DotLine)
                        pen.setCosmetic(True)
                        prob = Y[legal] / sumprob[legal]
                        item.setData(
                            inter_X[legal], prob, antialias=True, stepMode=False,
                            fillLevel=None, brush=None, pen=pen)
                        self.plot_prob.addItem(item)
                        item.tooltip = "Probability that \n" + group_var.name + "=" + cval
                        self.tooltip_items.append((self.plot_prob, item))

        elif var and var.is_discrete:
            bottomaxis.setTicks([list(enumerate(var.values))])

            cont = np.array(cont)

            maxh = 0 #maximal column height
            maxrh = 0 #maximal relative column height
            sgroup_var = cont.sum(axis=1)
            #a group_var with sum=0 with allways have distribution counts 0,
            #therefore we can divide it by anything
            sgroup_var[sgroup_var == 0] = 1
            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                maxh = max(maxh, max(dist))
                maxrh = max(maxrh, max(dist/sgroup_var))

            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                dsum = sum(dist)
                geom = QRectF(i - 0.333, 0, 0.666,
                              maxrh if self.relative_freq else maxh)
                item = DistributionBarItem(geom, dist/sgroup_var/maxrh
                                           if self.relative_freq
                                           else dist/maxh, colors)
                self.plot.addItem(item)
                tooltip = "\n".join(
                    "%s: %.*f" % (n, 3 if self.relative_freq else 1, v)
                    for n, v in zip(group_var_values, dist/sgroup_var if self.relative_freq else dist))
                item.tooltip = "{} ({}={}):\n{}".format(
                    "Normalized frequency " if self.relative_freq else "Frequency ",
                    group_var.name, value, tooltip)
                self.tooltip_items.append((self.plot, item))

                if self.show_prob:
                    item.tooltip += "\n\nProbabilities:"
                    for ic, a in enumerate(dist):
                        if self.show_prob - 1 != ic and \
                                self.show_prob - 1 != len(dist):
                            continue
                        position = -0.333 + ((ic+0.5)*0.666/len(dist))
                        if dsum < 1e-6:
                            continue
                        prob = a / dsum
                        if not 1e-6 < prob < 1 - 1e-6:
                            continue
                        ci = 1.96 * sqrt(prob * (1 - prob) / dsum)
                        item.tooltip += "\n%s: %.3f ± %.3f" % (group_var_values[ic], prob, ci)
                        mark = pg.ScatterPlotItem()
                        errorbar = pg.ErrorBarItem()
                        pen = QPen(QBrush(QColor(0)), 1)
                        pen.setCosmetic(True)
                        errorbar.setData(x=[i+position], y=[prob],
                                         bottom=min(np.array([ci]), prob),
                                         top=min(np.array([ci]), 1 - prob),
                                         beam=np.array([0.05]),
                                         brush=QColor(1), pen=pen)
                        mark.setData([i+position], [prob], antialias=True, symbol="o",
                                     fillLevel=None, pxMode=True, size=10,
                                     brush=QColor(colors[ic]), pen=pen)
                        self.plot_prob.addItem(errorbar)
                        self.plot_prob.addItem(mark)

        for color, name in zip(colors, group_var_values):
            self._legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, shape="s"),
                escape(name)
            )
        self._legend.show()

    def set_left_axis_name(self):
        leftaxis = self.ploti.getAxis("left")
        set_label = leftaxis.setLabel
        if self.var and self.var.is_continuous:
            set_label(["Density", "Relative density"]
                      [self.group_var is not None and self.relative_freq])
        else:
            set_label(["Frequency", "Relative frequency"]
                      [self.group_var is not None and self.relative_freq])
        leftaxis.resizeEvent()

    def enable_disable_rel_freq(self):
        self.cb_prob.setDisabled(self.var is None or self.group_var is None)
        self.cb_rel_freq.setDisabled(
            self.var is None or self.group_var is None)

    def _on_variable_changed(self):
        self.var = selected_variable(self.varview)
        self._setup()

    def _on_groupvar_changed(self):
        self._setup()

    def _on_set_smoothing(self):
        self._setup()

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()

    def get_widget_name_extension(self):
        return self.var

    def send_report(self):
        self.plotview.scene().setSceneRect(self.plotview.sceneRect())
        if self.var is None:
            return
        self.report_plot()
        text = "Distribution of '{}'".format(self.var)
        if self.group_var:
            group_var = self.group_var
            prob = self.cb_prob
            indiv_probs = 0 < prob.currentIndex() < prob.count() - 1
            if not indiv_probs or self.relative_freq:
                text += " grouped by '{}'".format(group_var)
                if self.relative_freq:
                    text += " (relative frequencies)"
            if indiv_probs:
                text += "; probabilites for '{}={}'".format(
                    group_var, prob.currentText())
        self.report_caption(text)

    @classmethod
    def migrate_context(cls, context, version, *, self=None, domain=None):
        """
        Migrate old indices into combos (`variable_idx`, `groupvar_idx`) to
        variables (`var`, `group_var`). This cannot be done through usual
        migrations because it requires a domain. The code is still put into
        this method just so that anybody looking for it will find it here.
        The patch is called after `openContext`
        """
        if version is None or version < 2:
            if domain is not None:  # Called after openContext, not migrations
                settings.migrate_idx_to_variable(
                    self, "variable_idx", "var", self.varmodel,
                    (var for var in chain(domain.variables, domain.metas)
                     if var.is_primitive()))
                settings.migrate_idx_to_variable(
                    self, "groupvar_idx", "group_var", self.groupvarmodel,
                    (var
                     for var in chain([None], domain.variables, domain.metas)
                     if var is None or var.is_discrete))
            else:
                super().migrate_context(context, version)
                # Further migrations go here, not above or outside!


def dist_sum(dXW1, dXW2):
    """
    A sum of two continuous distributions.
    """
    X1, W1 = dXW1
    X2, W2 = dXW2
    X = np.r_[X1, X2]
    W = np.r_[W1, W2]
    sort_ind = np.argsort(X)
    X, W = X[sort_ind], W[sort_ind]

    unique, uniq_index = np.unique(X, return_index=True)
    spans = np.diff(np.r_[uniq_index, len(X)])
    W = [np.sum(W[start:start + span])
         for start, span in zip(uniq_index, spans)]
    W = np.array(W)
    assert W.shape[0] == unique.shape[0]
    return unique, W


def ash_curve(dist, cont=None, bandwidth=None, m=3, smoothing_factor=1):
    dist = np.asarray(dist)
    X, W = dist
    if bandwidth is None:
        std = weighted_std(X, weights=W)
        size = X.size
        # if only one sample in the class
        if std == 0 and cont is not None:
            std = weighted_std(cont.values, weights=np.sum(cont.counts, axis=0))
            size = cont.values.size
        # if attr is constant or contingencies is None (no class variable)
        if std == 0:
            std = 0.1
            size = X.size
        bandwidth = 3.5 * std * (size ** (-1 / 3))

    hist, edges = average_shifted_histogram(X, bandwidth, m, weights=W,
                                            smoothing=smoothing_factor)
    return edges, hist


def average_shifted_histogram(a, h, m=3, weights=None, smoothing=1):
    """
    Compute the average shifted histogram.

    Parameters
    ----------
    a : array-like
        Input data.
    h : float
        Base bin width.
    m : int
        Number of shifted histograms.
    weights : array-like
        An array of weights of the same shape as `a`
    """
    a = np.asarray(a)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a")
        weights = weights.ravel()

    a = a.ravel()

    amin, amax = a.min(), a.max()
    h = h * 0.5 * smoothing
    delta = h / m
    wfac = 4 #extended windows for gaussian smoothing
    offset = (wfac * m - 1) * delta
    nbins = max(np.ceil((amax - amin + 2 * offset) / delta), 2 * m * wfac - 1)

    bins = np.linspace(amin - offset, amax + offset, nbins + 1,
                       endpoint=True)
    hist, edges = np.histogram(a, bins, weights=weights, density=True)

    kernel = gaussian_kernel((np.arange(2 * wfac * m - 1) - (wfac * m - 1)) / (wfac * m), wfac)
    kernel = kernel / np.sum(kernel)
    ash = np.convolve(hist, kernel, mode="same")

    ash = ash / np.diff(edges) / ash.sum()
#     assert abs((np.diff(edges) * ash).sum()) <= 1e-6
    return ash, edges


def triangular_kernel(x):
    return np.clip(1, 0, 1 - np.abs(x))


def gaussian_kernel(x, k):
    #fit k standard deviations into available space from [-1 .. 1]
    return 1/(np.sqrt(2 * np.pi)) * np.exp(-(x*k)**2 / 2)


def weighted_std(a, axis=None, weights=None, ddof=0):
    mean = np.average(a, axis=axis, weights=weights)

    if axis is not None:
        shape = shape_reduce_keep_dims(a.shape, axis)
        mean = mean.reshape(shape)

    sq_diff = np.power(a - mean, 2)
    mean_sq_diff, wsum = np.average(
        sq_diff, axis=axis, weights=weights, returned=True
    )

    if ddof != 0:
        mean_sq_diff *= wsum / (wsum - ddof)

    return np.sqrt(mean_sq_diff)


def weighted_quantiles(a, prob=[0.25, 0.5, 0.75], alphap=0.4, betap=0.4,
                       axis=None, weights=None):
    a = np.asarray(a)
    prob = np.asarray(prob)

    sort_ind = np.argsort(a, axis)
    a = a[sort_ind]

    if weights is None:
        weights = np.ones_like(a)
    else:
        weights = np.asarray(weights)
        weights = weights[sort_ind]

    n = np.sum(weights)
    k = np.cumsum(weights, axis)

    # plotting positions for the known n knots
    pk = (k - alphap * weights) / (n + 1 - alphap * weights - betap * weights)

#     m = alphap + prob * (1 - alphap - betap)

    return np.interp(prob, pk, a, left=a[0], right=a[-1])


def shape_reduce_keep_dims(shape, axis):
    if shape is None:
        return ()

    shape = list(shape)
    if isinstance(axis, collections.Sequence):
        for ax in axis:
            shape[ax] = 1
    else:
        shape[axis] = 1
    return tuple(shape)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    import gc
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWDistributions()
    w.show()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"
    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())
