import sys

import numpy as np
from PyQt4.QtCore import QSize, Qt
from PyQt4 import QtGui
from PyQt4.QtGui import QApplication

import Orange
from Orange.data import Table, Variable
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.widgets import gui
from Orange.widgets.settings import \
    DomainContextHandler, Setting, ContextSetting, SettingProvider
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.utils.toolbar import ZoomSelectToolbar
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph
from Orange.widgets.widget import OWWidget, Default, AttributeList


def font_resize(font, factor, minsize=None, maxsize=None):
    font = QtGui.QFont(font)
    fontinfo = QtGui.QFontInfo(font)
    size = fontinfo.pointSizeF() * factor

    if minsize is not None:
        size = max(size, minsize)
    if maxsize is not None:
        size = min(size, maxsize)

    font.setPointSizeF(size)
    return font


class OWScatterPlot(OWWidget):
    name = 'Scatter Plot'
    description = 'Scatter plot visualization.'
    icon = "icons/ScatterPlot.svg"

    inputs = [("Data", Table, "set_data", Default),
              ("Data Subset", Table, "set_subset_data"),
              ("Features", AttributeList, "set_shown_attributes")]

    outputs = [("Selected Data", Table),
               ("Other Data", Table)]

    settingsHandler = DomainContextHandler()

    auto_send_selection = Setting(True)
    toolbar_selection = Setting(0)
    color_settings = Setting(None)
    selected_schema_index = Setting(0)

    attr_x = ContextSetting("")
    attr_y = ContextSetting("")

    graph = SettingProvider(OWScatterPlotGraph)
    zoom_select_toolbar = SettingProvider(ZoomSelectToolbar)

    jitter_sizes = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.mainArea, True, margin=0)
        self.graph = OWScatterPlotGraph(self, box, "ScatterPlot")
        box.layout().addWidget(self.graph.plot_widget)
        plot = self.graph.plot_widget

        axisfont = font_resize(self.font(), 0.8, minsize=11)
        axispen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        axis = plot.getAxis("bottom")
        axis.setPen(axispen)

        axis = plot.getAxis("left")
        axis.setPen(axispen)

        self.data = None  # Orange.data.Table
        self.subset_data = None  # Orange.data.Table
        self.attribute_selection_list = None  # list of Orange.data.Variable

        common_options = {"labelWidth": 50, "orientation": "horizontal",
                          "sendSelectedValue": True, "valueType": str}
        box = gui.widgetBox(self.controlArea, "Axis Data")
        self.cb_attr_x = gui.comboBox(box, self, "attr_x", label="Axis x:",
                                      callback=self.major_graph_update,
                                      **common_options)
        self.cb_attr_y = gui.comboBox(box, self, "attr_y", label="Axis y:",
                                      callback=self.major_graph_update,
                                      **common_options)
        gui.valueSlider(
            box, self, value='graph.jitter_size',  label='Jittering: ',
            values=self.jitter_sizes, callback=self.reset_graph_data,
            labelFormat=lambda x:
            "None" if x == 0 else ("%.1f %%" if x < 1 else "%d %%") % x)
        gui.checkBox(
            gui.indentedBox(box), self, 'graph.jitter_continuous',
            'Jitter continuous values', callback=self.reset_graph_data)

        box = gui.widgetBox(self.controlArea, "Points")
        self.cb_attr_color = gui.comboBox(
            box, self, "graph.attr_color", label="Color:",
            emptyString="(Same color)", callback=self.graph.update_colors,
            **common_options)
        self.cb_attr_label = gui.comboBox(
            box, self, "graph.attr_label", label="Label:",
            emptyString="(No labels)", callback=self.graph.update_labels,
            **common_options)
        self.cb_attr_shape = gui.comboBox(
            box, self, "graph.attr_shape", label="Shape:",
            emptyString="(Same shape)", callback=self.graph.update_shapes,
            **common_options)
        self.cb_attr_size = gui.comboBox(
            box, self, "graph.attr_size", label="Size:",
            emptyString="(Same size)", callback=self.graph.update_sizes,
            **common_options)

        g = self.graph.gui
        box2 = g.point_properties_box(self.controlArea, box)
        gui.button(box2, self, "Set Colors", self.set_colors)

        box = gui.widgetBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend, g.ShowGridLines], box)
        gui.checkBox(box, self, value='graph.tooltip_shows_all',
                     label='Show all data on mouse hover')

        gui.separator(self.controlArea, 8, 8)
        self.zoom_select_toolbar = g.zoom_select_toolbar(
            self.controlArea, nomargin=True,
            buttons=[g.StateButtonsBegin, g.SimpleSelect, g.Pan, g.Zoom,
                     g.StateButtonsEnd, g.ZoomReset]
        )
        buttons = self.zoom_select_toolbar.buttons
        buttons[g.Zoom].clicked.connect(self.graph.zoom_button_clicked)
        buttons[g.Pan].clicked.connect(self.graph.pan_button_clicked)
        buttons[g.SimpleSelect].clicked.connect(self.graph.select_button_clicked)
        buttons[g.ZoomReset].clicked.connect(self.graph.reset_button_clicked)
        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        dlg = self.create_color_dialog()
        self.graph.continuous_palette = dlg.getContinuousPalette("contPalette")
        self.graph.discrete_palette = dlg.getDiscretePalette("discPalette")
        dlg.deleteLater()

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)

        gui.auto_commit(self.controlArea, self, "auto_send_selection",
                        "Send Selection")

        def zoom(s):
            """Zoom in/out by factor `s`."""
            viewbox = plot.getViewBox()
            # scaleBy scales the view's bounds (the axis range)
            viewbox.scaleBy((1 / s, 1 / s))

        def fit_to_view():
            viewbox = plot.getViewBox()
            viewbox.autoRange()

        zoom_in = QtGui.QAction(
            "Zoom in", self, triggered=lambda: zoom(1.25)
        )
        zoom_in.setShortcuts([QtGui.QKeySequence(QtGui.QKeySequence.ZoomIn),
                              QtGui.QKeySequence(self.tr("Ctrl+="))])
        zoom_out = QtGui.QAction(
            "Zoom out", self, shortcut=QtGui.QKeySequence.ZoomOut,
            triggered=lambda: zoom(1 / 1.25)
        )
        zoom_fit = QtGui.QAction(
            "Fit in view", self,
            shortcut=QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=fit_to_view
        )
        self.addActions([zoom_in, zoom_out, zoom_fit])

        # self.vizrank = OWVizRank(self, self.signalManager, self.graph,
        #                          orngVizRank.SCATTERPLOT, "ScatterPlot")
        # self.optimizationDlg = self.vizrank

    # def settingsFromWidgetCallback(self, handler, context):
    #     context.selectionPolygons = []
    #     for curve in self.graph.selectionCurveList:
    #         xs = [curve.x(i) for i in range(curve.dataSize())]
    #         ys = [curve.y(i) for i in range(curve.dataSize())]
    #         context.selectionPolygons.append((xs, ys))

    # def settingsToWidgetCallback(self, handler, context):
    #     selections = getattr(context, "selectionPolygons", [])
    #     for (xs, ys) in selections:
    #         c = SelectionCurve("")
    #         c.setData(xs,ys)
    #         c.attach(self.graph)
    #         self.graph.selectionCurveList.append(c)

    def reset_graph_data(self, *_):
        self.graph.rescale_data()
        self.major_graph_update()

    def set_data(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        if data is not None and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return

        self.closeContext()
        same_domain = \
            self.data and data and \
            data.domain.checksum() == self.data.domain.checksum()
        self.data = data

        # TODO: adapt scatter plot to work on SqlTables (avoid use of X and Y)
        if isinstance(self.data, SqlTable):
            self.data.download_data()

        # self.vizrank.clearResults()
        if not same_domain:
            self.init_attr_values()
        self.openContext(self.data)

    def set_subset_data(self, subset_data):
        self.subset_data = subset_data
        # self.vizrank.clearArguments()

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        self.graph.new_data(self.data, self.subset_data)
        # self.vizrank.resetDialog()
        if self.attribute_selection_list and \
                all(attr.name in self.graph.attribute_name_index
                    for attr in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0].name
            self.attr_y = self.attribute_selection_list[1].name
        self.attribute_selection_list = None
        self.update_graph()
        self.unconditional_commit()

    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
        else:
            self.attribute_selection_list = None

    # Callback from VizRank dialog
    def show_selected_attributes(self):
        val = self.vizrank.get_selected_projection()
        if not val:
            return
        if self.data.domain.class_var:
            self.graph.attr_color = self.data.domain.class_var.name
        self.major_graph_update(val[3])

    def get_shown_attributes(self):
        return self.attr_x, self.attr_y

    def init_attr_values(self):
        self.cb_attr_x.clear()
        self.cb_attr_y.clear()
        self.cb_attr_color.clear()
        self.cb_attr_color.addItem("(Same color)")
        self.cb_attr_label.clear()
        self.cb_attr_label.addItem("(No labels)")
        self.cb_attr_shape.clear()
        self.cb_attr_shape.addItem("(Same shape)")
        self.cb_attr_size.clear()
        self.cb_attr_size.addItem("(Same size)")
        if not self.data:
            return

        for var in self.data.domain.metas:
            self.cb_attr_label.addItem(self.icons[var], var.name)
        for attr in self.data.domain.variables:
            self.cb_attr_x.addItem(self.icons[attr], attr.name)
            self.cb_attr_y.addItem(self.icons[attr], attr.name)
            self.cb_attr_color.addItem(self.icons[attr], attr.name)
            if attr.is_discrete:
                self.cb_attr_shape.addItem(self.icons[attr], attr.name)
            else:
                self.cb_attr_size.addItem(self.icons[attr], attr.name)
            self.cb_attr_label.addItem(self.icons[attr], attr.name)

        self.attr_x = self.cb_attr_x.itemText(0)
        if self.cb_attr_y.count() > 1:
            self.attr_y = self.cb_attr_y.itemText(1)
        else:
            self.attr_y = self.cb_attr_y.itemText(0)

        if self.data.domain.class_var:
            self.graph.attr_color = self.data.domain.class_var.name
        else:
            self.graph.attr_color = ""
        self.graph.attr_shape = ""
        self.graph.attr_size = ""
        self.graph.attr_label = ""

    def major_graph_update(self, attributes=None, inside_colors=None, **args):
        self.update_graph(attributes, inside_colors, **args)

    def update_graph(self, attributes=None, inside_colors=None, **_):
        self.graph.zoomStack = []
        if not self.graph.have_data:
            return
        if attributes and len(attributes) == 2:
            self.attr_x, self.attr_y = attributes
        self.graph.update_data(self.attr_x, self.attr_y)

    def saveSettings(self):
        OWWidget.saveSettings(self)
        # self.vizrank.saveSettings()

    def selection_changed(self):
        self.commit()

    def commit(self):
        selected = unselected = None
        # TODO: Implement selection for sql data
        if isinstance(self.data, SqlTable):
            selected = unselected = self.data
        elif self.data is not None:
            selection = self.graph.get_selection()
            selected = self.data[selection]
            unselection = np.full(len(self.data), True, dtype=bool)
            unselection[selection] = False
            unselected = self.data[unselection]
        self.send("Selected Data", selected)
        self.send("Other Data", unselected)

    def set_colors(self):
        dlg = self.create_color_dialog()
        if dlg.exec_():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.graph.continuous_palette = dlg.getContinuousPalette("contPalette")
            self.graph.discrete_palette = dlg.getDiscretePalette("discPalette")
            self.update_graph()

    def create_color_dialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c

    def closeEvent(self, ce):
        # self.vizrank.close()
        super().closeEvent(ce)

    def sendReport(self):
        self.startReport(
            "%s [%s - %s]" % (self.windowTitle(), self.attr_x, self.attr_y))
        self.reportSettings(
            "Visualized attributes",
            [("X", self.attr_x),
             ("Y", self.attr_y),
             self.graph.attr_color and ("Color", self.graph.attr_color),
             self.graph.attr_label and ("Label", self.graph.attr_label),
             self.graph.attr_shape and ("Shape", self.graph.attr_shape),
             self.graph.attr_size and ("Size", self.graph.attr_size)])
        self.reportSettings(
            "Settings",
            [("Symbol size", self.graph.point_width),
             ("Opacity", self.graph.alpha_value),
             ("Jittering", self.graph.jitter_size),
             ("Jitter continuous attributes",
              gui.YesNo[self.graph.jitter_continuous])])
        self.reportSection("Graph")
        self.reportImage(self.graph.save_to_file, QSize(400, 400))


def test_main():
    import sip
    a = QApplication(sys.argv)
    ow = OWScatterPlot()
    ow.show()
    ow.raise_()
    data = Orange.data.Table(r"iris.tab")
    ow.set_data(data)
    ow.set_subset_data(data[:30])
    #ow.setData(orange.ExampleTable("wine.tab"))
    ow.handleNewSignals()
    a.exec()
    #save settings
    ow.saveSettings()
    sip.delete(ow)

if __name__ == "__main__":
    test_main()
