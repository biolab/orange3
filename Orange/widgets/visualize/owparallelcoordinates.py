import sys

from PyQt4.QtGui import QApplication

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.data import Table
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.widgets.gui import attributeIconDict
from Orange.widgets.settings import DomainContextHandler, Setting, SettingProvider
from Orange.widgets.utils.colorpalette import ColorPaletteDlg, ColorPaletteGenerator
from Orange.widgets.utils.plot import xBottom, OWPalette
from Orange.widgets.utils.scaling import checksum
from Orange.widgets.utils.toolbar import ZoomSelectToolbar, ZOOM, PAN, SPACE, REMOVE_ALL, SEND_SELECTION
from Orange.widgets.visualize.owparallelgraph import OWParallelGraph
from Orange.widgets.visualize.owviswidget import OWVisWidget
from Orange.widgets.widget import AttributeList
from Orange.widgets import gui, widget


CONTINUOUS_PALETTE = "contPalette"
DISCRETE_PALETTE = "discPalette"
CANVAS_COLOR = "Canvas"


class OWParallelCoordinates(OWVisWidget):
    name = "Parallel Coordinates"
    description = "Parallel coordinates display of multi-dimensional data."
    icon = "icons/ParallelCoordinates.svg"
    priority = 100
    author = "Gregor Leban, Anze Staric"
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data Subset", Orange.data.Table, 'set_subset_data'),
              ("Features", AttributeList, 'set_shown_attributes')]
    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               ("Other Data", Orange.data.Table),
               ("Features", AttributeList)]

    settingsHandler = DomainContextHandler()

    show_all_attributes = Setting(default=False)
    auto_send_selection = Setting(default=True)

    color_settings = Setting(default=None)
    selected_schema_index = Setting(default=0)

    jitterSizeNums = [0, 2, 5, 10, 15, 20, 30]

    graph = SettingProvider(OWParallelGraph)
    zoom_select_toolbar = SettingProvider(ZoomSelectToolbar)

    __ignore_updates = True

    def __init__(self):
        super().__init__()
        #add a graph widget
        self.graph = OWParallelGraph(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.data = None
        self.subset_data = None
        self.discrete_attribute_order = "Unordered"
        self.continuous_attribute_order = "Unordered"
        self.middle_labels = "Correlations"

        self.create_control_panel()
        self.color_picker = self.create_color_picker_dialog()
        self.graph.continuous_palette = self.color_picker.getContinuousPalette(CONTINUOUS_PALETTE)
        self.graph.discrete_palette = self.color_picker.getDiscretePalette(DISCRETE_PALETTE)
        self.graph.setCanvasBackground(self.color_picker.getColor(CANVAS_COLOR))

        self.resize(900, 700)

        self.__ignore_updates = False

    #noinspection PyAttributeOutsideInit
    def create_control_panel(self):
        self.control_tabs = gui.tabWidget(self.controlArea)
        self.general_tab = gui.createTabPage(self.control_tabs, "Main")
        self.settings_tab = gui.createTabPage(self.control_tabs, "Settings")

        self.add_attribute_selection_area(self.general_tab)
        self.add_zoom_select_toolbar(self.general_tab)

        self.add_visual_settings(self.settings_tab)
        #self.add_annotation_settings(self.settings_tab)
        self.add_color_settings(self.settings_tab)
        self.add_group_settings(self.settings_tab)

        self.settings_tab.layout().addStretch(100)
        self.icons = attributeIconDict

    def add_attribute_selection_area(self, parent):
        super().add_attribute_selection_area(parent)
        self.shown_attributes_listbox.itemDoubleClicked.connect(self.flip_attribute)

    #noinspection PyAttributeOutsideInit
    def add_zoom_select_toolbar(self, parent):
        buttons = (ZOOM, PAN, SPACE, REMOVE_ALL, SEND_SELECTION)
        self.zoom_select_toolbar = ZoomSelectToolbar(self, parent, self.graph, self.auto_send_selection,
                                                     buttons=buttons)
        self.zoom_select_toolbar.buttonSendSelections.clicked.connect(self.sendSelections)

    def add_visual_settings(self, parent):
        box = gui.widgetBox(parent, "Visual Settings")
        gui.checkBox(box, self, 'graph.show_attr_values', 'Show attribute values', callback=self.update_graph)
        gui.checkBox(box, self, 'graph.use_splines', 'Show splines', callback=self.update_graph,
                     tooltip="Show lines using splines")
        self.graph.gui.show_legend_check_box(box)

    def add_annotation_settings(self, parent):
        box = gui.widgetBox(parent, "Statistical Information")
        gui.comboBox(box, self, "graph.show_statistics", label="Statistics: ", orientation="horizontal", labelWidth=90,
                     items=["No statistics", "Means, deviations", "Median, quartiles"], callback=self.update_graph,
                     sendSelectedValue=False, valueType=int)
        gui.checkBox(box, self, 'graph.show_distributions', 'Show distributions', callback=self.update_graph,
                     tooltip="Show bars with distribution of class values (only for discrete attributes)")

    def add_color_settings(self, parent):
        box = gui.widgetBox(parent, "Colors", orientation="horizontal")
        gui.button(box, self, "Set colors", self.select_colors,
                   tooltip="Set the canvas background color and color palette for coloring continuous variables")

    def add_group_settings(self, parent):
        box = gui.widgetBox(parent, "Groups", orientation="vertical")
        box2 = gui.widgetBox(box, orientation="horizontal")
        gui.checkBox(box2, self, "graph.group_lines", "Group lines into", tooltip="Show clusters instead of lines",
                     callback=self.update_graph)
        gui.spin(box2, self, "graph.number_of_groups", 0, 30, callback=self.update_graph)
        gui.label(box2, self, "groups")
        box2 = gui.widgetBox(box, orientation="horizontal")
        gui.spin(box2, self, "graph.number_of_steps", 0, 100, label="In no more than", callback=self.update_graph)
        gui.label(box2, self, "steps")

    def flip_attribute(self, item):
        if self.graph.flip_attribute(str(item.text())):
            self.update_graph()
            self.information(0)
        else:
            self.information(0, "Didn't flip the attribute. To flip a continuous "
                                "attribute uncheck 'Global value scaling' checkbox.")

    def update_graph(self):
        self.graph.update_data(self.shown_attributes)

    def increase_axes_distance(self):
        m, M = self.graph.bounds_for_axis(xBottom)
        if (M - m) == 0:
            return # we have not yet updated the axes (self.graph.updateAxes())
        self.graph.setAxisScale(xBottom, m, M - (M - m) / 10., 1)
        self.graph.replot()

    def decrease_axes_distance(self):
        m, M = self.graph.bounds_for_axis(xBottom)
        if (M - m) == 0:
            return # we have not yet updated the axes (self.graph.updateAxes())

        self.graph.setAxisScale(xBottom, m, min(len(self.graph.attributes) - 1, M + (M - m) / 10.), 1)
        self.graph.replot()

    # ------------- SIGNALS --------------------------
    # receive new data and update all fields
    def set_data(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        if data and (not bool(data) or len(data.domain) == 0):
            data = None
        if checksum(data) == checksum(self.data):
            return # check if the new data set is the same as the old one

        self.__ignore_updates = True
        self.closeContext()
        same_domain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data

        if not same_domain:
            self.shown_attributes = None

        self.openContext(self.data)
        self.__ignore_updates = False

    def set_subset_data(self, subset_data):
        self.subset_data = subset_data

    # attribute selection signal - list of attributes to show
    def set_shown_attributes(self, shown_attributes):
        self.new_shown_attributes = shown_attributes

    new_shown_attributes = None

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.__ignore_updates = True
        self.graph.set_data(self.data, self.subset_data)
        if self.new_shown_attributes:
            self.shown_attributes = self.new_shown_attributes
            self.new_shown_attributes = None
        else:
            self.shown_attributes = self._shown_attributes
            # trust open context to take care of this?
            # self.shown_attributes = None
        self.update_graph()
        self.sendSelections()
        self.__ignore_updates = False

    def send_shown_attributes(self, attributes=None):
        if attributes is None:
            attributes = self.shown_attributes
        self.send("Features", attributes)

    def selectionChanged(self):
        self.zoom_select_toolbar.buttonSendSelections.setEnabled(not self.auto_send_selection)
        if self.auto_send_selection:
            self.sendSelections()

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        return

    # jittering options
    def setJitteringSize(self):
        self.graph.rescale_data()
        self.update_graph()

    def select_colors(self):
        dlg = self.color_picker
        if dlg.exec_():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.graph.continuous_palette = dlg.getContinuousPalette(CONTINUOUS_PALETTE)
            self.graph.discrete_palette = dlg.getDiscretePalette(DISCRETE_PALETTE)
            self.graph.setCanvasBackground(dlg.getColor(CANVAS_COLOR))
            self.update_graph()

    def create_color_picker_dialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette(DISCRETE_PALETTE, "Discrete Palette")
        c.createContinuousPalette(CONTINUOUS_PALETTE, "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, CANVAS_COLOR, "Canvas color", self.graph.color(OWPalette.Canvas))
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c

    def attributes_changed(self):
        if not self.__ignore_updates:
            self.graph.removeAllSelections()
            self.update_graph()

            self.send_shown_attributes()


#test widget appearance
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWParallelCoordinates()
    ow.show()
    ow.graph.discrete_palette = ColorPaletteGenerator(rgb_colors=[(127, 201, 127), (190, 174, 212), (253, 192, 134)])
    ow.graph.group_lines = True
    ow.graph.number_of_groups = 10
    ow.graph.number_of_steps = 30
    data = Orange.data.Table("iris")
    ow.set_data(data)
    ow.handleNewSignals()

    a.exec_()

    ow.saveSettings()
