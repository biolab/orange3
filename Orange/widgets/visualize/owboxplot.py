# -*- coding: utf-8 -*-

import operator
import math
from scipy import stats
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui

from Orange.data import  DiscreteVariable, Table, Domain
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import datacaching
from Orange.widgets.widget import Default
from Orange.widgets.utils.plot import owaxis

#from OWColorPalette import ColorPixmap, ColorPaletteGenerator

class BoxItem(QtGui.QGraphicsItemGroup):
    WIDTH = 20

    pen_light = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0xff, 0xff, 0xff)), 2)
    pen_dark = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    for pen in (pen_dark, pen_light):
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)

    def __init__(self, index, label, a_min, mean, a_max, q25, q50, q75):
        super().__init__()
        self.index = index
        self.mean = mean
        self.median = q50
        self.label = label

        Line = QtGui.QGraphicsLineItem
        width = self.WIDTH
        whisker1 = Line(-width/8, a_min, width/8, a_min, self)
        whisker2 = Line(-width/8, a_max, width/8, a_max, self)
        vert_line = Line(0, a_min, 0, a_max, self)
        mean_line = Line(-0.7*width, mean, 0.7*width, mean, self)
        for it in (whisker1, whisker2, vert_line, mean_line):
            it.setPen(self.pen_dark)

        box = QtGui.QGraphicsRectItem(-width/2, q25, width, q75 - q25, self)
        box.setBrush(QtGui.QBrush(QtGui.QColor(0x33, 0x88, 0xff, 0xc0)))
        box.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        box.setZValue(100)

        median_line = Line(-width/2, q50, width/2, q50, self)
        median_line.setPen(self.pen_light)
        median_line.setZValue(200)

        # not inserted into the group in order to avoid scaling
        self.label_text = t = QtGui.QGraphicsSimpleTextItem(label)
        t.setY(a_min)
        t.setFlags(t.flags() |
                   QtGui.QGraphicsItem.ItemIgnoresTransformations)


class OWBoxPlot(widget.OWWidget):
    _name = "Box plot"
    _description = "Shows box plots"
    _long_description = """Shows box plots, either one for or multiple
    box plots for data split by an attribute value."""
    #_icon = "icons/Boxplot.svg"
    _priority = 100
    _author = "Amela Rakanović, Janez Demšar"
    inputs = [("Data", Table, "data")]
    outputs = [("Basic statistic", Table, Default),
               ("Significant columns", Table),
               ("Selected Data", Table)]

    settingsHandler = DomainContextHandler()
    sorting_select = Setting(0)
    grouping_select = ContextSetting([0])
    attributes_select = ContextSetting([0])
    stattest = Setting(0)
    sig_threshold = Setting(0.05)

    _sorting_criteria_attrs = ["index", "label", "median", "mean"]

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.grouping = []
        self.attributes = []

        self.slider_intervals = 1
        self.ddataset = None

        ## Control Area
        self.attr_list_box = gui.listBox(self.controlArea, self,
            "attributes_select", "attributes", box="Variable",
            selectionMode=QtGui.QListWidget.SingleSelection,
            callback=self.process_change)

        gb = gui.widgetBox(self.controlArea, orientation=1)
        self.attrCombo = gui.listBox(gb, self,
            'grouping_select', "grouping", box="Grouping",
            selectionMode=QtGui.QListWidget.SingleSelection,
            callback=self.process_change)
        self.sorting_combo = gui.comboBox(gb, self,
            'sorting_select', callback=self.sorting_update,
            items=["Show in original order",
                   "Sort by label", "Sort by median", "Sort by mean"])

        b = gui.widgetBox(self.controlArea, "Statistics")
        gui.doubleSpin(b, self, "sig_threshold", 0.05, 0.5, step=0.05,
            label="Significance", controlWidth=75,
            alignment=QtCore.Qt.AlignRight)
        gui.widgetLabel(b, "Test type")
        gui.radioButtonsInBox(gui.indentedBox(b, sep=10), self, "stattest",
            ["Parametric", "Non-parametric"],
            callback = self.run_selected_test)

        gui.rubber(self.controlArea)

        ## Main Area
        result = gui.widgetBox(self.mainArea, addSpace=True)
        self.boxScene = QtGui.QGraphicsScene()
        self.boxView = QtGui.QGraphicsView(self.boxScene)
        self.boxView.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.boxView)
        self.no_values = gui.widgetLabel(self.mainArea,
            "<center><big><b>Too many values.</b></big></center>")
        self.no_values.hide()

        e = gui.widgetBox(self.mainArea, addSpace=False, orientation=0)
        self.infot1 = gui.widgetLabel(e, "<center>No test results.</center>")

        self.warning = gui.widgetBox(self.controlArea, "Warning:")
        self.warning_info = gui.widgetLabel(self.warning, "")
        self.warning.hide()

        #self.controlArea.setFixedWidth(250)
        self.mainArea.setMinimumWidth(650)

    def data(self, dataset):
        if dataset is not None and (
                not len(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.ddataset = dataset
        self.grouping_select = []
        self.attributes_select = []
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.openContext(self.ddataset)
        if dataset:
            self.attrCombo.addItem("(none)")
            attributes = []
            grouping = ["None"]
            for attr in self.ddataset.domain:
                if isinstance(attr, DiscreteVariable):
                    grouping.append((attr.name, attr.var_type))
                else:
                    attributes.append((attr.name, attr.var_type))
            self.attributes = attributes
            self.grouping = grouping
            self.grouping_select = [0]
            self.attributes_select = [0]
            self.process_change()
        else:
            self.reset_all_data()

    def reset_all_data(self):
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.graph.clear_data()
        self.send("Basic statistic", None)
        self.send("Significant data", None)

    def process_change(self):
        self.draw_selected()
#        self.run_selected_test()
#        self.basic_stat = stat_basic_full_tab(self.ddataset)
#        self.send("Basic statistic", self.basic_stat)

    def draw_selected(self):
        dataset = self.ddataset
        if dataset is None:
            return

        def compute_box(dist):
            N = np.sum(dist[1])
            if N == 0:
                return None
            a_min, a_max = dist[0, 0], dist[0, -1]
            mean = np.sum(dist[0] * dist[1]) / N
            box_data = [a_min, mean, a_max]
            s = 0
            thresholds = [N/4, N/2, 3*N/4]
            thresh_i = 0
            for i, e in enumerate(dist[1]):
                s += e
                if s >= thresholds[thresh_i]:
                    if s == thresholds[thresh_i] and i + 1 < dist.shape[1]:
                        box_data.append((dist[0, i] + dist[0, i + 1]) / 2)
                    else:
                        box_data.append(dist[0, i])
                    thresh_i += 1
                    if thresh_i == 3:
                        break
            while len(box_data) != 6:
                box_data.append(box_data[-1])
            return box_data

        self.warning.hide()
        attr = self.attributes[self.attributes_select[0]][0]
        attr_ind = dataset.domain.index(attr)
        attr_desc = dataset.domain[attr_ind]
        group_by = self.grouping_select[0]
        if group_by:
            group_attr = self.grouping[group_by][0]
            group_ind = dataset.domain.index(group_attr)
            self.conts = datacaching.getCached(dataset,
                contingency.get_contingency,
                (dataset, attr_ind, group_ind))
            boxes = [(value, compute_box(self.conts[i]))
                for i, value in enumerate(dataset.domain[group_ind].values)]

            # TODO: add another box for the entire data set
            self.sorting_combo.setDisabled(False)
        else:
            self.dist = datacaching.getCached(dataset,
                distribution.get_distribution, (attr_ind, dataset))
            boxes = [(attr, compute_box(self.dist))]
            self.sorting_combo.setDisabled(True)

        self.boxScene.clear()
        a_min = min(box[0] for _, box in boxes if box)
        a_max = max(box[2] for _, box in boxes if box)
        self.boxes = []
        for i, (label, box_data) in enumerate(boxes):
            if box_data:
                box = BoxItem(i, label, *tuple(box_data))
                self.boxScene.addItem(box)
                self.boxScene.addItem(box.label_text)
                self.boxes.append(box)
        self.set_positions()

        first_val, step = owaxis.OWAxis.compute_scale(a_min, a_max)
        while a_min < first_val:
            first_val -= step
        a_min = first_val
        no_ticks = math.ceil((a_max - first_val) / step) + 1
        a_max = max(a_max, first_val + (no_ticks - 1) * step)

        r = QtCore.QRectF(-80, a_min, len(boxes)*60+60, a_max - a_min)
        self.boxScene.setSceneRect(r)

        # This code comes from the implementation of QGraphicsView::fitInView
        # (http://qt.gitorious.org/qt/qt/blobs/4.8/src/gui/graphicsview/
        # qgraphicsview.cpp), but modified to scale only vertically
        d = self.boxView
        d.resetTransform()
        unity = d.matrix().mapRect(QtCore.QRectF(0, 0, 1, 1))
        d.scale(1, 1 / unity.height())
        viewRect = d.viewport().rect().adjusted(15, 15, -15, -15)
        yratio = viewRect.height() / (a_max - a_min)
        d.scale(1, -yratio)
        d.centerOn((-80 + len(boxes) * 60 + 60) / 2, (a_max - a_min) / 2)

        # This comes last because we need to position the text to the
        # appropriate transformed coordinates
        font = QtGui.QFont()
        font.setPixelSize(12)
        val = first_val
        gray_pen = QtGui.QPen(QtCore.Qt.lightGray)
        while True:
            self.boxScene.addLine(-65, val, -60, val, gray_pen)
            t = self.boxScene.addSimpleText(attr_desc.repr_val(val), font)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(-70 - r.width(), val + r.height() / 2 / yratio)
            if val >= a_max:
                break
            val += step
        a_max = val
        self.boxScene.addLine(-60, first_val, -60, a_max, gray_pen)

    def set_positions(self):
        for i, box in enumerate(self.boxes):
            box.setX(i * 60)
            box.label_text.setX(
                i * 60 - box.label_text.boundingRect().width() / 2)

    def sorting_update(self):
        criterion = self._sorting_criteria_attrs[self.sorting_select]
        self.boxes.sort(key=operator.attrgetter(criterion))
        self.set_positions()

    def send_to_graph(self, dataset, attr, box_label, y_label):
        if dataset:
            np_data = data_to_npcol(dataset, dataset.domain.index(attr))
            stat_graph1 = stat_graph(np_data)
            self.graph.append_data(box_label, stat_graph1, y_label)


    def run_selected_test(self):
        return

        """ Runs selected tests """

        if self.grouping_select and self.attributes_select and self.grouping and self.attributes:
            self.graph.show()
            self.graph.removeBoxes()
            self.no_values.hide()

            slected_grouping_str = self.grouping[self.grouping_select[0]][0]

            if self.grouping_select[0] == 0:
                self.infot1.setText("<center>No test results<center>")
                if self.attributes_select:

                    self.sorting_update()
                    self.draw_selected(self.ddataset)

            elif len(self.ddataset.domain[slected_grouping_str].values) < 20:

                filtered, depr= self.filtered(self.grouping, self.grouping_select)
                number_tab = len(self.ddataset.domain[slected_grouping_str].values) - len(depr) # # of nondeprecated data

                if number_tab == 2:
                    a = stat_ttest(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))
                    b = stat_wilc(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))
                    self.infot1.setText("<center>Student's t: %.3f (p=%.3f), Mann-Whitney's U: %.1f (p=%.3f)</center>" % (a[0], a[1], b[0], b[1]))

                elif number_tab > 2:
                    a = stat_anova(filtered, str(self.attributes[self.attributes_select[0]][0]))
                    c = stat_kruskal(filtered, str(self.attributes[self.attributes_select[0]][0]))
                    self.infot1.setText("<center>ANOVA: %.3f (p=%.3f), Kruskal Wallis's U: %.1f (p=%.3f)</center>" % (a[0], a[1], c[0], c[1]))

                elif number_tab < 2:
                    self.infot1.setText("<center>Not enough data examples.<center>")

                self.draw_selected(self.ddataset)

            else:
                self.no_values.setText("<center><b>Too many values to be drawn.</b><center>")
                self.infot1.setText("")
                self.graph.hide()
                self.no_values.show()

            if self.stattest == 0:
                self.send("Significant data", self.ouput_sig_parametric_data())
            elif self.stattest == 1:
                self.send("Significant data", self.ouput_sig_nonparametric_data())

        else:
            self.graph.hide()
            self.graph.clear_data()
            self.infot1.setText("")

    def ouput_sig_parametric_data(self):
        tab = []

        for grou in self.grouping[1:]:
            filtered, depr = self.filtered(self.grouping, [self.grouping.index(grou)])
            if len(filtered) > 1:
                if len(self.ddataset.domain[grou[0]].values) == 2:
                    a = stat_ttest(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))
                    if a[1] < self.sig_threshold:
                        tab.append(grou[0])
                elif len(self.ddataset.domain[grou[0]].values) > 2:
                    a = stat_anova(filtered, str(self.attributes[self.attributes_select[0]][0]))
                    if a[1] < self.sig_threshold:
                        tab.append(grou[0])
        for attr in self.attributes:
            tab.append(attr[0])
        return self.ddataset.select(tab)

    def ouput_sig_nonparametric_data(self):
        tab = []
        for grou in self.grouping[1:]:
            filtered, depr = self.filtered(self.grouping, [self.grouping.index(grou)])
            if len(filtered) > 1:
                if len(self.ddataset.domain[grou[0]].values) == 2:
                    a = stat_wilc(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))
                    if a[1] < self.sig_threshold:
                        tab.append(grou[0])
                elif len(self.ddataset.domain[grou[0]].values) > 2:
                    a = stat_kruskal(filtered, str(self.attributes[self.attributes_select[0]][0]))
                    if a[1] < self.sig_threshold:
                        tab.append(grou[0])
        for attr in self.attributes:
            tab.append(attr[0])
        return self.ddataset.select(tab)

# ***************************************************************************
# ***************************************************************************
# Statistics

def data_to_npcol(dataset, attrs):
    """Extracts columns from dataset in numpy format

    Usage::
      (tab1, tab2) = data_to_npcol(dataset, [selected, i])
      tab1 = data_to_npcol(dataset, selected)

    :param dataset:  orange dataset
    :param attrs:    list of attribute strings or positions
    :return:         list of numpy columns
    """
    is_single = False
    if not isinstance(attrs, list):
        is_single = True
        attrs = [attrs]

    domain = Domain([ dataset.domain[a]  for a in attrs ])
    dataset_sel = Table(domain, dataset)

    f = filter.IsDefined(domain=domain)
    (dataset_np,) = f(dataset_sel).to_numpy('ac')

    if is_single:
        return dataset_np.T[0]
    else:
        return dataset_np.T

# Basic statistic -------------------------------------------------------------

def stat_basic(np_column):
    """Returns basic statistics on numpy array
    :param dataset:  numpy array
    :return:              (min, max, mean, median, standard deviation, variance)
    """
    return [np.amin(np_column), np.amax(np_column), np.mean(np_column), np.median(np_column), np.std(np_column), np.var(np_column)]

def stat_basic_full_tab(dataset):
    """Runs basic stat on dataset and returns dataset
    :param dataset: orange dataset
    :return:        orange dataset
    """
    a = np.empty([0,6])
    for i in xrange(len(dataset.domain)):
        datacol = data_to_npcol(dataset,i)
        if datacol != []:
            statb = stat_basic(datacol)
            a = np.vstack((a, statb))

    #d = Orange.data.Domain([0])
    d = Orange.data.Domain([Orange.feature.Continuous('min'), Orange.feature.Continuous('max'), Orange.feature.Continuous('mean'), Orange.feature.Continuous('median'), Orange.feature.Continuous('st. deviation'), Orange.feature.Continuous('variance')])
    return Orange.data.Table(d, a)

def stat_graph(np_column):
    """Return data to be sended to boxplot
    :param np_column:   numpy array
    :return:            (min, qdown, meadian, qup, max, mean, std)
    """
    return [np.amin(np_column), stats.scoreatpercentile(np_column, 25), stats.scoreatpercentile(np_column, 50), stats.scoreatpercentile(np_column, 75), np.amax(np_column), np.mean(np_column), np.std(np_column)]

# TESTS
# t-test ----------------------------------------------------------------------
# parametric

def stat_ttest(arr1, arr2, selected):
    """T-test
    :param arr1:    orange dataset filtered by discrete attribute A,
    :param arr1:    orange dataset filtered by discrete attribute A
    :return:        (U, p)
    """
    tab1 = data_to_npcol(arr1, selected)
    tab2 = data_to_npcol(arr2, selected)
    if tab1.any() and tab2.any():
        return stats.ttest_ind(tab1, tab2)
    else:
        return(0, 0)
# Mann Whitney test ---------------------------------------------------------------
# nonparametric

def stat_wilc(arr1, arr2, selected):
    """Mann Whitney test
    :param arr1:    orange dataset filtered by discrete attribute A,
    :param arr1:    orange dataset filtered by discrete attribute A
    :return:        (U, p)
    """
    tab1 = data_to_npcol(arr1, selected)
    tab2 = data_to_npcol(arr2, selected)
    if tab1.any() and tab2.any():
        return stats.mannwhitneyu(tab1,tab2)
    else:
        return(0, 0)
# ANOVA -----------------------------------------------------------------------
# parametric

def stat_anova(filtered, selected):
    """ANOVA
    :param filtered:    list of orange datasets,
    :param selected:    name of attribute
    :return:            (F, p):
    """
    tab = [ data_to_npcol(f, selected)  for f in filtered ]
    return stats.f_oneway(*tab)

# Kruskal Wallis  --------------------------------------------------------------
# nonparametric

def stat_kruskal(filtered, selected):
    """Kruskal Wallis
    :param filtered:    list of orange datasets,
    :param selected:    name of attribute
    :return:            (F, p)
    """
    tab = [ data_to_npcol(f, selected)  for f in filtered ]
    return stats.kruskal(*tab)

# ***************************************************************************
# ***************************************************************************

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QtGui.QApplication([])
    ow = OWBoxPlot()
    ow.show()
    # /home/ami/orange/Orange/doc/datasets
    dataset = Table('../doc/datasets/heart_disease.tab')
    ow.data(dataset)
    appl.exec_()
