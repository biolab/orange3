import sys

from AnyQt.QtCore import Qt

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.data import Table
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.widgets.gui import attributeIconDict
from Orange.widgets.settings import DomainContextHandler, Setting, SettingProvider
from Orange.widgets.utils.plot import xBottom
from Orange.widgets.utils import checksum
from Orange.widgets.utils.toolbar import ZoomSelectToolbar, ZOOM, PAN, SPACE, REMOVE_ALL, SEND_SELECTION
from Orange.widgets.visualize.owparallelgraph import OWParallelGraph
from Orange.widgets.visualize.owviswidget import OWVisWidget
from Orange.widgets.widget import AttributeList
from Orange.widgets import gui, widget


class OWParallelCoordinates(OWVisWidget):
    name = "Parallel Coordinates"
    description = "Parallel coordinates display of multi-dimensional data."
    icon = "icons/ParallelCoordinates.svg"
    priority = 900
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data Subset", Orange.data.Table, 'set_subset_data'),
              ("Features", AttributeList, 'set_shown_attributes')]
    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               ("Other Data", Orange.data.Table),
               ("Features", AttributeList)]

    settingsHandler = DomainContextHandler()

    show_all_attributes = Setting(default=False)
    auto_send_selection = Setting(default=True)

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

        self.resize(900, 700)

        self.__ignore_updates = False

    #noinspection PyAttributeOutsideInit
    def create_control_panel(self):
        self.add_attribute_selection_area(self.controlArea)
        self.add_visual_settings(self.controlArea)
        #self.add_annotation_settings(self.        controlArea)
        self.add_group_settings(self.controlArea)
        self.add_zoom_select_toolbar(self.controlArea)
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
        box = gui.vBox(parent, "Visual Settings")
        gui.checkBox(box, self, 'graph.show_attr_values', 'Show attribute values', callback=self.update_graph)
        gui.checkBox(box, self, 'graph.use_splines', 'Show splines', callback=self.update_graph,
                     tooltip="Show lines using splines")
        self.graph.gui.show_legend_check_box(box)

    def add_annotation_settings(self, parent):
        box = gui.vBox(parent, "Statistical Information")
        gui.comboBox(box, self, "graph.show_statistics", label="Statistics: ",
                     orientation=Qt.Horizontal, labelWidth=90,
                     items=["No statistics", "Means, deviations", "Median, quartiles"], callback=self.update_graph,
                     sendSelectedValue=False, valueType=int)
        gui.checkBox(box, self, 'graph.show_distributions', 'Show distributions', callback=self.update_graph,
                     tooltip="Show bars with distribution of class values (only for discrete attributes)")

    def add_group_settings(self, parent):
        box = gui.vBox(parent, "Groups")
        box2 = gui.hBox(box)
        gui.checkBox(box2, self, "graph.group_lines", "Group lines into", tooltip="Show clusters instead of lines",
                     callback=self.update_graph)
        gui.spin(box2, self, "graph.number_of_groups", 0, 30, callback=self.update_graph)
        gui.label(box2, self, "groups")
        box2 = gui.hBox(box)
        gui.spin(box2, self, "graph.number_of_steps", 0, 100, label="In no more than", callback=self.update_graph)
        gui.label(box2, self, "steps")

    def flip_attribute(self, item):
        if self.graph.flip_attribute(str(item.text())):
            self.update_graph()
            self.information()
        else:
            self.information("To flip a numeric feature, disable"
                             "'Global value scaling'")

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

    def attributes_changed(self):
        if not self.__ignore_updates:
            self.graph.removeAllSelections()
            self.update_graph()

            self.send_shown_attributes()

    def send_report(self):
        self.report_plot(self.graph)


#test widget appearance
if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWParallelCoordinates()
    ow.show()
    ow.graph.group_lines = True
    ow.graph.number_of_groups = 10
    ow.graph.number_of_steps = 30
    data = Orange.data.Table("iris")
    ow.set_data(data)
    ow.handleNewSignals()

    a.exec_()

    ow.saveSettings()
