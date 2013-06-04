import os
import random

from PyQt4 import QtCore
from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.plot import owplot, owconstants, owpoint
from Orange.data.table import Table
from Orange.data.domain import Domain
from Orange.data.variable import DiscreteVariable, ContinuousVariable
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
icon_magnet = os.path.join(dir_path, "icons/paintdata/magnet.svg")
icon_jitter = os.path.join(dir_path, "icons/paintdata/jitter.svg")
icon_brush = os.path.join(dir_path, "icons/paintdata/brush.svg")
icon_put = os.path.join(dir_path, "icons/paintdata/put.svg")
icon_select = os.path.join(dir_path, "icons/paintdata/select-transparent_42px.png")
icon_lasso = os.path.join(dir_path, "icons/paintdata/lasso-transparent_42px.png")

class PaintDataPlot(owplot.OWPlot):
    def __init__(self, parent=None,  name="None",  show_legend = 1, axes=[owconstants.xBottom, owconstants.yLeft], widget=None):
        super().__init__(parent, name, show_legend, axes, widget)
        self.state = owconstants.SELECT
        self.graph_margin = 10
        self.y_axis_extra_margin = 10

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        position = event.pos()
        print(position)
        maptoscene = self.mapToScene(position)
        print(maptoscene)
        print(self.map_from_graph(maptoscene))



class DataTool(QtCore.QObject):
    """ A base class for data tools that operate on PaintDataPlot
    widget by installing itself as its event filter.

    """
    cursor = QtCore.Qt.ArrowCursor
    class optionsWidget(QtGui.QFrame):
        """ An options (parameters) widget for the tool (this will
        be put in the "Options" box in the main OWPaintData widget
        when this tool is selected.

        """
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool

    def __init__(self, graph, parent=None):
        QtCore.QObject.__init__(self, parent)
        self.setGraph(graph)

    def setGraph(self, graph):
        """ Install this tool to operate on ``graph``. If another tool
        is already operating on the graph it will first be removed.

        """
        self.graph = graph
        if graph:
            installed = getattr(graph, "_data_tool_event_filter", None)
            if installed:
                self.graph.removeEventFilter(installed)
                installed.removed()
            self.graph.canvas().setMouseTracking(True)
            self.graph.canvas().installEventFilter(self)
            self.graph._data_tool_event_filter = self
            self.graph._tool_pixmap = None
            self.graph.setCursor(self.cursor)
            self.graph.replot()
            self.installed()

    def paintEvent(self, event):
        return False

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def enterEvent(self, event):
        return False

    def leaveEvent(self, event):
        return False

    def keyPressEvent(self, event):
        return False

    def attributes(self):
        return self.graph.attr1, self.graph.attr2

    def dataTransform(self, *args):
        pass

class BrushTool(DataTool):
    brushRadius = 20
    density = 5
    cursor = QtCore.Qt.CrossCursor

    class optionsWidget(QtGui.QFrame):
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool
            layout = QtGui.QFormLayout()
            self.radiusSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.radiusSlider.pyqtConfigure(minimum=10, maximum=30, value=self.tool.brushRadius)
            self.densitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.densitySlider.pyqtConfigure(minimum=3, maximum=10, value=self.tool.density)

            layout.addRow("Radius", self.radiusSlider)
            layout.addRow("Density", self.densitySlider)
            self.setLayout(layout)

            self.connect(self.radiusSlider, QtCore.SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "brushRadius", value))

            self.connect(self.densitySlider, QtCore.SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "density", value))

    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.brushState = -20, -20, 0, 0

    def mousePressEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & QtCore.Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
            self.emit(QtCore.SIGNAL("editing()"))
        self.graph.replot()
        return True

    def mouseMoveEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & QtCore.Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
            self.emit(QtCore.SIGNAL("editing()"))
        self.graph.replot()
        return True

    def mouseReleaseEvent(self, event):
        self.graph.replot()
        if event.button() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editingFinished()"))
        return True

    def leaveEvent(self, event):
        self.graph._tool_pixmap = None
        self.graph.replot()
        return False


    # def brushGeometry(self, point):
    #     coord = self.invTransform(point)
    #     dcoord = self.invTransform(QPoint(point.x() + self.brushRadius, point.y() + self.brushRadius))
    #     x, y = coord.x(), coord.y()
    #     rx, ry = dcoord.x() - x, -(dcoord.y() - y)
    #     return x, y, rx, ry

    # def dataTransform(self, attr1, x, rx, attr2, y, ry):
    #     import random
    #     new = []
    #     for i in range(self.density):
    #         ex = orange.Example(self.graph.data.domain)
    #         ex[attr1] = random.normalvariate(x, rx)
    #         ex[attr2] = random.normalvariate(y, ry)
    #         ex.setclass(self.graph.data.domain.classVar(self.graph.data.domain.classVar.baseValue))
    #         new.append(ex)
    #     self.graph.data.extend(new)
    #     self.graph.updateGraph(dataInterval=(-len(new), sys.maxint))



class OWPaintData(widget.OWWidget):
    TOOLS = [("Brush", "Create multiple instances", BrushTool, icon_brush),
             ("Put", "Put individual instances", None, icon_put),
             ("Select", "Select and move instances", None, icon_select),
             ("Lasso", "Select and move instances", None, icon_lasso),
             ("Jitter", "Jitter instances", None, icon_jitter),
             ("Magnet", "Move (drag) multiple instances", None, icon_magnet),
             ]
    _name = "Paint Data"
    _description = """
    Creates the data by painting on the graph."""
    _long_description = """
    """
    _icon = "icons/PaintData.svg"
    _author = "Martin Frlin"
    _priority = 10
    _category = "Data"
    _keywords = ["data", "paint", "create"]
    outputs = [("Data", Table)]

    commit_on_change = Setting(False)

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.undoStack = QtGui.QUndoStack(self)

        self.plot = PaintDataPlot(self.mainArea, "Painted Plot", widget=self)
        self.classValuesModel = itemmodels.PyListModel(
            ["class-1", "class-2"], self,
            flags=QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        self.connect(self.classValuesModel,
            QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
            self.updateData)

        self.data = Table(
            Domain([ContinuousVariable("attribute-1"), ContinuousVariable("attribute-2")],
            DiscreteVariable("Class label", values=self.classValuesModel)))

        self.toolsStackCache = {}

        self.initUI()
        self.initPlot()

    def initUI(self):
        undoRedoBox = gui.widgetBox(self.controlArea, "", addSpace=True)
        undo = QtGui.QAction("Undo", self)
        undo.pyqtConfigure(toolTip="Undo Action (Ctrl+Z)")
        undo.setShortcut("Ctrl+Z")
        self.connect(undo, QtCore.SIGNAL("triggered()"), self.testMethod)
        redo = QtGui.QAction("Redo", self)
        redo.pyqtConfigure(toolTip="Redo Action (Ctrl+Shift+Z)")
        undo.setShortcut("Ctrl+Shift+Z")
        self.connect(redo, QtCore.SIGNAL("triggered()"), self.undoStack.redo)
        undoRedoActionsWidget = itemmodels.ModelActionsWidget([undo, redo], self)
        undoRedoActionsWidget.layout().addStretch(10)
        undoRedoActionsWidget.layout().setSpacing(1)
        undoRedoBox.layout().addWidget(undoRedoActionsWidget)

        classesBox = gui.widgetBox(self.controlArea, "Class Labels")
        self.classListView = listView = QtGui.QListView()
        listView.setSelectionMode(QtGui.QListView.SingleSelection)
        listView.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Maximum)
        listView.setModel(self.classValuesModel)
        listView.selectionModel().select(self.classValuesModel.index(0), QtGui.QItemSelectionModel.ClearAndSelect)
        self.connect(listView.selectionModel(),
                     QtCore.SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     lambda x, y: self.testMethod(x, y))
        classesBox.layout().addWidget(listView)

        addClassLabel = QtGui.QAction("+", self)
        addClassLabel.pyqtConfigure(toolTip="Add class label")
        self.connect(addClassLabel, QtCore.SIGNAL("triggered()"), self.addNewClassLabel)
        removeClassLabel = QtGui.QAction("-", self)
        removeClassLabel.pyqtConfigure(toolTip="Remove class label")
        self.connect(removeClassLabel, QtCore.SIGNAL("triggered()"), self.removeSelectedClassLabel)
        actionsWidget = itemmodels.ModelActionsWidget([addClassLabel, removeClassLabel], self)
        actionsWidget.layout().addStretch(10)
        actionsWidget.layout().setSpacing(1)
        classesBox.layout().addWidget(actionsWidget)

        toolsBox = gui.widgetBox(self.controlArea, "Tools", orientation=QtGui.QGridLayout(), addSpace=True)
        self.toolActions = QtGui.QActionGroup(self)
        self.toolActions.setExclusive(True)

        for i, (name, tooltip, tool, icon) in enumerate(self.TOOLS):
            action = QtGui.QAction(name, self)
            action.setToolTip(tooltip)
            action.setCheckable(True)
            action.setIcon(QtGui.QIcon(icon))
            self.connect(action, QtCore.SIGNAL("triggered()"), lambda tool=tool: self.setCurrentTool(tool))
            button = QtGui.QToolButton()
            button.setDefaultAction(action)
            button.setIconSize(QtCore.QSize(24, 24))
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            button.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
            toolsBox.layout().addWidget(button, i / 3, i % 3)
            self.toolActions.addAction(action)

        for column in range(3):
            toolsBox.layout().setColumnMinimumWidth(column, 10)
            toolsBox.layout().setColumnStretch(column, 1)

        self.optionsLayout = QtGui.QStackedLayout()

        optionsBox = gui.widgetBox(self.controlArea, "Options", addSpace=True, orientation=self.optionsLayout)

        commitBox = gui.widgetBox(self.controlArea, "Commit")
        gui.checkBox(commitBox, self, "commit_on_change", "Commit on change")
        gui.button(commitBox, self, "Commit", callback=self.sendData)

        # main area GUI
        self.mainArea.layout().addWidget(self.plot)

    def updateData(self):
        self.data = Table(
            Domain([ContinuousVariable("attribute-1"), ContinuousVariable("attribute-2")],
            DiscreteVariable("Class label", values=self.classValuesModel)))
        self.updatePlot()

    def initPlot(self):
        self.plot.set_axis_title(owconstants.xBottom, self.data.domain[0].name)
        self.plot.set_show_axis_title(owconstants.xBottom, True)
        self.plot.set_axis_title(owconstants.yLeft, self.data.domain[1].name)
        self.plot.set_show_axis_title(owconstants.yLeft, True)
        self.plot.set_axis_scale(owconstants.xBottom, 0, 1, 0.1)
        self.plot.set_axis_scale(owconstants.yLeft, 0, 1, 0.1)
        self.updatePlot()

    def updatePlot(self):
        self.plot.legend().clear()
        for i, value in enumerate(self.data.domain[2].values):
            rgb = self.plot.discrete_palette.getRGB(i)
            self.plot.legend().add_item(
                self.data.domain[2].name, value,
                owpoint.OWPoint(owpoint.OWPoint.Diamond, QtGui.QColor(*rgb), 5))

    def addNewClassLabel(self):
        self.classValuesModel.append("class-%d" % (len(self.classValuesModel)+1))
        self.updateData()

    def removeSelectedClassLabel(self):
        index = self.selectedClassLabelIndex()
        if index is not None:
            self.classValuesModel.pop(index)
            self.updateData()

    def selectedClassLabelIndex(self):
        rows = [i.row() for i in self.classListView.selectionModel().selectedRows()]
        if rows:
            return rows[0]
        else:
            return None

    def onClassLabelSelection(self, selected, unselected):
        index = self.selectedClassLabelIndex()
        if index is not None:
            self.classVariable.baseValue = index

    def setCurrentTool(self, tool):
        if tool not in self.toolsStackCache:
            newtool = tool(None, self)
            option = newtool.optionsWidget(newtool, self)
            self.optionsLayout.addWidget(option)
            self.connect(newtool, QtCore.SIGNAL("editing()"), self.onDataChanged)
            self.connect(newtool, QtCore.SIGNAL("editingFinished()"), self.commitIf)
            self.toolsStackCache[tool] = (newtool, option)

        self.currentTool, self.currentOptionsWidget = tool, option = self.toolsStackCache[tool]
        self.optionsLayout.setCurrentWidget(option)

    def sendData(self):
        self.send("Data", self.data)

    def testMethod(self, x="lol", y="troll"):
        print(x)
        print(y)

    def sizeHint(self):
        return QtCore.QSize(800, 400)