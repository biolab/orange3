# -*- coding: utf-8 -*-
"""
<name>Comparison of mean/medians</name>
<description></description>
<icon></icon>
<priority>10</priority>
<category>Statistics</category>
"""

import Orange

from PyQt4.QtCore import *

from OWWidget import *
import OWGUI

from scipy import stats
import numpy as np

from OWColorPalette import ColorPixmap, ColorPaletteGenerator
from plot.owplot import *
import textwrap


#
# Statistics: Comparison of mean/medians
#


class BoxItem(OWPlotItem):
    
    def __init__(self, text = None):
        OWPlotItem.__init__(self)
        self._item1 = QGraphicsPathItem(self)
        self._item2 = QGraphicsPathItem(self)       
        self._item3 = QGraphicsPathItem(self)
        self._item4 = QGraphicsPathItem(self)
        self._xindex = 0
        self._data = []
        self._label = text
    
    def set_xindex(self,xindex):
        self._xindex=xindex
        self.update_properties()
        
    def set_label(self, label):
        self._xlabel = label
        self.update_properties()
        
    def get_label(self):
        return self._xlabel
    
    def update_properties(self):
        OWPlotItem.update_properties(self)
        (xBottom, yLeft) = self.axes()

        d = self.data()
        if d:
            wbox = 0.3
            smallw = 0.1
            
            self.set_data_rect(QRectF(self._xindex-wbox/2, d[0], wbox, d[4]-d[0]))
            
            #min, max, hline
            q = QPainterPath()
            pen = QPen(QColor(255, 165, 0))
            pen.setWidth(3)
            brush = QBrush(Qt.blue)
            
            q.moveTo(-(smallw/2), d[0])
            q.lineTo((smallw/2), d[0])
            q.moveTo(0, d[0])
            q.lineTo(0, d[4])
            q.moveTo(-(smallw/2), d[4])
            q.lineTo((smallw/2), d[4])

            q.translate(self._xindex,0)
            self._item1.setPath(self.graph_transform().map(q))
            self._item1.setPen(pen)
            self._item1.setBrush(brush)
            self._item1.show()
                        
            #box
            u = QPainterPath()
            pen = QPen(QColor(255, 165, 0))
            pen.setWidth(3)
            brush = QBrush(QColor(  255,228,181))
            
            u.addRect(-(wbox/2), d[1], wbox, d[3]-d[1])
            
            u.translate(self._xindex,0)
            self._item2.setPath(self.graph_transform().map(u))
            self._item2.setPen(pen)
            self._item2.setBrush(brush)
            self._item2.show()
                        
            #median
            p = QPainterPath()            
            pen = QPen(Qt.red)
            pen.setWidth(3)
            
            p.moveTo(-(wbox/2), d[2])
            p.lineTo((wbox/2), d[2])
            
            p.translate(self._xindex,0)
            self._item3.setPath(self.graph_transform().map(p))
            self._item3.setPen(pen)
            self._item3.show()
    
            #median and distribution
            r = QPainterPath()        
            #pen = QPen(QColor(  34,139,34))
            pen = QPen(Qt.darkGreen)
            pen.setWidth(3)
            
            AA = 0.8*smallw/2
            r.moveTo(-AA, d[5])
            r.lineTo(AA, d[5])
            
            r.moveTo(0, d[5]-d[6])
            r.lineTo(0, d[5]+d[6])
            
            r.translate(self._xindex,0)
            self._item4.setPath(self.graph_transform().map(r))
            self._item4.setPen(pen)
            self._item4.setBrush(QBrush(Qt.red))
            self._item4.show()
            
        else:
            pass
            #self._item.hide()

    def data(self):
        return self._data
        
    def set_data(self, data):
        # [dmin, qdown, dmean, qup, dmax]
        self._data = data
        self.update_properties()

class OWBoxPlotQt(OWPlot):
    def __init__(self, settingsWidget = None, parent = None, name = None):
        OWPlot.__init__(self, parent, name)
        self.parent = parent

        # initialize settings
        self.settingsWidget = settingsWidget
        self.boxes = []
        self.labels = []
        self.set_axis_labels(2, self.labels)
        self.set_axis_title(0, '')
        self.set_show_axis_title(0, 1)
        self.sizeHint()
        
        self.axis_margin = 20
        self.y_axis_extra_margin = 35
        self.title_margin = 40
        self.graph_margin = 10

        self.grid_curve.set_pen(QPen(3))
        #self.grid_curve.set_x_enabled(False)

    def addBoxItem(self, label, xAxis = xBottom, yAxis = yLeft, visible = 1):
        box = BoxItem(label)
        #tooltip = QToolTip()
        #tooltip.showText
        #box.setToolTip(tooltip)
        box.set_axes(xAxis, yAxis)
        box.setVisible(visible)
        return OWPlot.add_custom_curve(self, box, enableLegend=0)

    def sizeHint(self):
        return QSize(100, 100)

    def sort_by_label(self):
        self.labels2 = []
        self.boxes.sort(key=lambda x: x._label, reverse=False)
        for i in xrange (len(self.boxes)):
            self.boxes[i].set_xindex(i)
            self.labels2.append(str(self.boxes[i].get_label()))
        self.set_axis_labels(2, self.labels2)
        self.replot()

    def sort_by_median(self):
        self.labels2 = []
        self.boxes.sort(key=lambda x: x._data[2], reverse=True)
        for i in xrange (len(self.boxes)):
            self.boxes[i].set_xindex(i)
            self.labels2.append(str(self.boxes[i].get_label()))
        self.set_axis_labels(2, self.labels2)
        self.replot()

    def sort_by_average(self):
        self.labels2 = []
        self.boxes.sort(key=lambda x: x._data[5], reverse=True)
        for i in xrange (len(self.boxes)):
            self.boxes[i].set_xindex(i)
            self.labels2.append(str(self.boxes[i].get_label()))
        self.set_axis_labels(2, self.labels2)
        self.replot()

    def append_data(self, label, data, y_label):
        self.labels.append(label)
        self.set_axis_title(0, y_label)
        self.boxes.append(self.addBoxItem(label, xBottom, yLeft))
        self.boxes[-1].set_data(data)
        self.boxes[-1].set_xindex(len(self.boxes)-1)
        self.boxes[-1].set_label(label)
        self.set_axis_labels(2, self.labels)

        self.replot()

    def clear_data(self):
        for i in self.boxes:
            self.remove_item(i)
        self.boxes = []
        self.labels = []
        self.set_axis_labels(2, self.labels)

class OWComparisonQt(OWWidget):
    settingsList=["sorting_select", "stattest", "sig_threshold"]

    contextHandlers = {"": DomainContextHandler("", ["attributes_select", "grouping_select"])}

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Comparison of mean/medians')

        self.grouping = []
        self.attributes = []

        self.grouping_select = [0]
        self.attributes_select = [0]

        #local variables     
        self.slider_intervals = 1

        self.inputs = [("Data", ExampleTable, self.data)]
        self.outputs = [("Basic statistic", ExampleTable), ("Significant data", ExampleTable)]

        self.sorting_select = ''
        self.stattest = 0
        self.sig_threshold = 0.05
        
        self.ddataset = None

        self.loadSettings()

        # GUI      
        self.controlArea.setFixedWidth(250)
        self.mainArea.setMinimumWidth(650)
        
        ## Control Area        
        self.attr_list_box = OWGUI.listBox(self.controlArea, self, "attributes_select", "attributes", box="Variable", selectionMode=QListWidget.SingleSelection, callback=self.run_selected_test) 
        
        gb = OWGUI.widgetBox(self.controlArea, orientation=1)
        self.attrCombo = OWGUI.listBox(gb, self, 'grouping_select', "grouping", box="Grouping", selectionMode=QListWidget.SingleSelection, callback=self.run_selected_test)

        self.sorting_combo = OWGUI.comboBox(gb, self, 'sorting_select', sendSelectedValue = 1, 
        emptyString="Sort by label", callback=self.sorting_update)
        self.sorting_combo.addItem("Sort by label")
        self.sorting_combo.addItem("Sort by median")
        self.sorting_combo.addItem("Sort by average")
        
        b = OWGUI.widgetBox(self.controlArea, "Selection of output variables")
        OWGUI.doubleSpin(b, self, "sig_threshold", 0.05, 0.5, step=0.05, label="Significance threshold", controlWidth=75, alignment=Qt.AlignRight)
        OWGUI.widgetLabel(b, "Statistical test")
        OWGUI.radioButtonsInBox(OWGUI.indentedBox(b, sep=10), self, "stattest",
                                ["Parametric (t-test)", "Non-parametric (Mann-Whitney)"], callback = self.run_selected_test)
        
        OWGUI.rubber(self.controlArea)
        
        ## Main Area   
        self.diagram_tab = self.mainArea
        result = OWGUI.widgetBox(self.diagram_tab, addSpace=True)
        self.graph = OWBoxPlotQt(self, self.diagram_tab)
        self.diagram_tab.layout().addWidget(self.graph)
        self.no_values = OWGUI.widgetLabel(self.diagram_tab, "<center><big><b>Toooo many values.</b></big></center>")
        self.no_values.hide()
        
        e = OWGUI.widgetBox(self.mainArea, addSpace=False, orientation=0)
        self.infot1 = OWGUI.widgetLabel(e, "<center>No test results.</center>")
       
        self.warrning = OWGUI.widgetBox(self.controlArea, "Warning:")
        self.warning_info = OWGUI.widgetLabel(self.warrning, "")
        self.warrning.hide()
    
    def sorting_update(self):
        if self.sorting_select:
            if self.sorting_select == "Sort by median":
                self.graph.sort_by_median()
            elif self.sorting_select == "Sort by average":
                self.graph.sort_by_average()
        else:
            self.graph.sort_by_label()
             
    def filtered(self, grou, grou_select):
        fya = Orange.data.filter.Values()
        fya.domain = self.ddataset.domain
        filtered = []
        deprecated = []
        select_combo = self.ddataset.domain[str(grou[grou_select[0]][0])]
        for aya in select_combo.values:
            filt = Orange.data.filter.ValueFilterDiscrete(position=self.ddataset.domain.index(select_combo), values=[Orange.data.Value(select_combo, aya)] )
            fya.conditions = [filt]
            if len(fya(self.ddataset)) > 3:
                filtered.append(fya(self.ddataset))
            else:
                deprecated.append(aya)
        return (filtered, deprecated)
    
    def run_selected_test(self):  
        """ Runs selected tests """
        
        if self.grouping_select and self.attributes_select and self.grouping and self.attributes:
            self.graph.show()
            self.graph.clear_data()
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
                    print 'Running Student\'s t-test and Mann Whitney test'                  
                    a = stat_ttest(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))
                    b = stat_wilc(filtered[0], filtered[1], str(self.attributes[self.attributes_select[0]][0]))          
                    self.infot1.setText("<center>Student's t: %.3f (p=%.3f), Mann-Whitney's U: %.1f (p=%.3f)</center>" % (a[0], a[1], b[0], b[1]))
                    
                elif number_tab > 2:
                    print 'Running ANOVA and comp'
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
    
    def data(self, dataset):
        if dataset is not None and (len(dataset) == 0 or len(dataset.domain) == 0):
            dataset = None
        if self.ddataset and dataset and self.ddataset.checksum() == dataset.checksum():
            return    # check if the new data set is the same as the old one
        
        self.closeContext()
        
        self.ddataset = dataset
        self.grouping_select = [0]
        self.attributes_select = [0]
        self.openContext("", self.ddataset)
        
        if dataset:
            (grouping_select, attributes_select) = (self.grouping_select[:], self.attributes_select[:])

            self.attributes = []
            self.grouping = []

            self.attr_list_box.clear()
            self.attrCombo.clear()
            
            self.attrCombo.addItem("(none)")
            self.grouping.append('None')

            for attr in self.ddataset.domain:
                if attr.varType == orange.VarTypes.Discrete:
                    self.grouping.append((attr.name, attr.varType))
                else:
                    self.attributes.append((attr.name, attr.varType))
            #force redrawing
            self.attributes = self.attributes
            self.grouping = self.grouping
            
            (self.grouping_select, self.attributes_select) = (grouping_select, attributes_select)
            
            self.run_selected_test()
            
            self.basic_stat = stat_basic_full_tab(self.ddataset)
            self.send("Basic statistic", self.basic_stat)
        
        else:
            self.reset_all_data()        
    
    def reset_all_data(self):
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.graph.clear_data()
        self.send("Basic statistic", None)
        self.send("Significant data", None)
        
    def draw_selected(self, dataset):
        """ Sends apappropriate selections to send_to_graph"""
        self.warrning.hide()
        fya = Orange.data.filter.Values()
        fya.domain = dataset.domain
        if self.grouping_select[0] != 0:
            select_combo = dataset.domain[str(self.grouping[self.grouping_select[0]][0])]
            self.attr_list_box.setSelectionMode(QListWidget.SingleSelection)    
            if self.attributes_select:
                if len(self.attributes_select)>1:
                    self.attributes_select = [self.attributes_select[0]]
                omitted = []
                omitted_dirty = 0
                for aya in select_combo.values:
                    filt = Orange.data.filter.ValueFilterDiscrete(position=dataset.domain.index(select_combo), values=[Orange.data.Value(select_combo, aya)] )
                    fya.conditions = [filt]
                    filtered = fya(dataset)        
                    
                    if len(filtered) < 3:     
                        omitted_dirty = 1
                        omitted.append(aya)
                    
                    else:
                        self.send_to_graph(filtered, self.attributes[self.attributes_select[0]][0], str(Orange.data.Value(select_combo, aya)), self.attributes[self.attributes_select[0]][0])
                        
                if omitted_dirty == 1:
                    X = textwrap.fill("Omitted: "  + ', '.join(omitted) , 25) 
                    self.warning_info.setText('<center>' + X + '</center>') 
                    self.warrning.show()
                
                else:
                    self.warrning.hide()
        
        else:
            if self.attributes_select:
                for aya in self.attributes_select:
                    self.send_to_graph(dataset, self.attributes[aya][0], self.attributes[aya][0], '')

            else:
                self.graph.clear_data()
    
    def send_to_graph(self, dataset, attr, box_label, y_label):
        if dataset:
            np_data = data_to_npcol(dataset, dataset.domain.index(attr))
            stat_graph1 = stat_graph(np_data)
            self.graph.append_data(box_label, stat_graph1, y_label)

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
    
    domain = Orange.data.Domain([ dataset.domain[a]  for a in attrs ])
    dataset_sel = Orange.data.Table(domain, dataset)
    
    f = Orange.data.filter.IsDefined(domain=domain)
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
    appl = QApplication(sys.argv)
    ow = OWComparisonQt()
    ow.show()
    # /home/ami/orange/Orange/doc/datasets
    dataset = orange.ExampleTable('../doc/datasets/heart_disease.tab')
    ow.data(dataset)
    appl.exec_()
