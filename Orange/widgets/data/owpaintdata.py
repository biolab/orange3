import os
import random

from PyQt4 import QtCore
from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.plot import owplot, owconstants, owpoint
from Orange.data.domain import Domain
from Orange.data.instance import Instance
from Orange.data.table import Table
from Orange.data.variable import DiscreteVariable, ContinuousVariable


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
        self.state = owconstants.NOTHING
        self.graph_margin = 10
        self.y_axis_extra_margin = 10
        self.animate_plot = False
        self.animate_points = True
        self.tool = None

    def mousePressEvent(self, event):
        if self.state == owconstants.NOTHING and self.tool:
            self.tool.mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.state == owconstants.NOTHING and self.tool:
            self.tool.mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)




class DataTool(QtCore.QObject):
    """
    A base class for data tools that operate on PaintDataPlot
    widget by installing itself as its event filter.

    """
    cursor = QtCore.Qt.ArrowCursor
    class optionsWidget(QtGui.QFrame):
        """
        An options (parameters) widget for the tool (this will
        be put in the "Options" box in the main OWPaintData widget
        when this tool is selected.

        """
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool

    def __init__(self, parent):
        QtCore.QObject.__init__(self, parent)
        #self.setGraph(graph)
        self.widget = parent
        self.widget.plot.setCursor(self.cursor)

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def toDataPoint(self, point):
        """
        Converts mouse position point to data point as its represented on the graph.
        """
        # first we convert it from widget point to scene point
        scenePoint = self.widget.plot.mapToScene(point)
        # and then from scene to data point
        return self.widget.plot.map_from_graph(scenePoint, zoom=True)


class PutInstanceTool(DataTool):
    cursor = QtCore.Qt.CrossCursor

    def mousePressEvent(self, event):
        dataPoint = self.toDataPoint(event.pos())
        print(dataPoint)
        if event.buttons() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editing()"))
            self.widget.addDataPoints([dataPoint])
            self.emit(QtCore.SIGNAL("editingFinished()"))
        return True


class BrushTool(DataTool):
    brushRadius = 70
    density = 5
    cursor = QtCore.Qt.CrossCursor

    class optionsWidget(QtGui.QFrame):
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool
            layout = QtGui.QFormLayout()
            self.radiusSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.radiusSlider.pyqtConfigure(minimum=50, maximum=100, value=self.tool.brushRadius)
            self.densitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.densitySlider.pyqtConfigure(minimum=3, maximum=10, value=self.tool.density)

            layout.addRow("Radius", self.radiusSlider)
            layout.addRow("Density", self.densitySlider)
            self.setLayout(layout)

            self.connect(self.radiusSlider, QtCore.SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "brushRadius", value))

            self.connect(self.densitySlider, QtCore.SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "density", value))


    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editing()"))
            dataPoint = self.toDataPoint(event.pos())
            self.previousDataPoint = dataPoint
            self.widget.addDataPoints(self.createPoints(dataPoint))
        return True

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editing()"))
            dataPoint = self.toDataPoint(event.pos())
            if abs(dataPoint[0] - self.previousDataPoint[0]) > self.brushRadius/2000 or abs(dataPoint[1] - self.previousDataPoint[1]) > self.brushRadius/2000:
                self.widget.addDataPoints(self.createPoints(dataPoint))
                self.previousDataPoint = dataPoint
        return True

    def mouseReleaseEvent(self, event):
        if event.button() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editingFinished()"))
        return True

    def createPoints(self, point):
        """
        Creates random points around the point that is passed in.
        """
        points = []
        x, y = point
        radius = self.brushRadius/1000
        for i in range(self.density):
            rndX = random.random()*radius
            rndY = random.random()*radius
            points.append((x+(radius/2)-rndX, y+(radius/2)-rndY))
        return points

class MagnetTool(DataTool):
    brushRadius = 70
    cursor = QtCore.Qt.CrossCursor

    class optionsWidget(QtGui.QFrame):
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool
            layout = QtGui.QFormLayout()
            self.radiusSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
            self.radiusSlider.pyqtConfigure(minimum=50, maximum=100, value=self.tool.brushRadius)
            layout.addRow("Radius", self.radiusSlider)
            self.setLayout(layout)

            self.connect(self.radiusSlider, QtCore.SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "brushRadius", value))


    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.emit(QtCore.SIGNAL("editing()"))
            dataPoint = self.toDataPoint(event.pos())
            self.previousDataPoint = dataPoint
            self.widget.addDataPoints(self.createPoints(dataPoint))
        return True

class SelectTool(DataTool):
    cursor = QtCore.Qt.ArrowCursor

    class optionsWidget(QtGui.QFrame):
        def __init__(self, tool, parent=None):
            QtGui.QFrame.__init__(self, parent)
            self.tool = tool
            layout = QtGui.QHBoxLayout()
            delete = QtGui.QToolButton(self)
            delete.pyqtConfigure(text="Delete", toolTip="Delete selected instances")
            self.connect(delete, QtCore.SIGNAL("clicked()"), self.tool.printSelected)

            layout.addWidget(delete)
            layout.addStretch(10)
            self.setLayout(layout)


    def __init__(self, parent):
        super(SelectTool, self).__init__(parent)
        self.widget.plot.activate_selection()

    def printSelected(self):
        [print(curve) for curve in self.widget.plot.selected_points()]

    def deleteSelected(self, *args):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        path = self.selection.path()
        selected = [i for i, ex in enumerate(data) if path.contains(QtGui.QPointF(float(ex[attr1]) , float(ex[attr2])))]
        for i in reversed(selected):
            del data[i]
        self.graph.updateGraph()
        if selected:
            self.emit(QtCore.SIGNAL("editing()"))
            self.emit(QtCore.SIGNAL("editingFinished()"))


class CommandAddData(QtGui.QUndoCommand):
    def __init__(self, data, points, classLabel, widget, description):
        super(CommandAddData, self).__init__(description)
        self.data = data
        self.points = points
        self.row = len(self.data)
        print(self.row)
        self.classLabel = classLabel
        self.widget = widget

    def redo(self):
        instances = [Instance(self.data.domain,
                              [x, y, self.classLabel]) for x, y in self.points if 0 <= x <= 1 and 0 <= y <= 1]
        self.data.extend(instances)
        self.widget.updatePlot()

    def undo(self):
        del self.data[self.row:self.row+len(self.points)]
        self.widget.updatePlot()


class CommandDelData(QtGui.QUndoCommand):
    def __init__(self, data, selectedPoints, description):
        super(CommandDelData, self).__init__(description)
        self.data = data
        self.points = selectedPoints

    def redo(self):
        pass

    def undo(self):
        pass


class CommandAddClassLabel(QtGui.QUndoCommand):
    def __init__(self, data, domain, classValues, widget, description):
        super(CommandAddClassLabel, self).__init__(description)
        self.data = data
        self.domain = domain
        self.oldDomain = data.domain
        self.classValues = classValues
        self.widget = widget

    def redo(self):
        self.data = Table().from_table(self.domain, self.data)
        self.widget.updatePlot()
        print(self.classValuesModel)
        print(self.domain)

    def undo(self):
        self.data = Table().from_table(self.oldDomain, self.data)
        self.classValues.pop()
        self.widget.updatePlot()
        print(self.classValuesModel)
        print(self.oldDomain)


class CommandRemoveClassLabel(QtGui.QUndoCommand):
    def __init__(self, data, classValues, index, description):
        super(CommandDelData, self).__init__(description)
        self.data = data
        self.classValues = classValues
        self.index = index

    def redo(self):
        pass

    def undo(self):
        pass


class OWPaintData(widget.OWWidget):
    TOOLS = [("Brush", "Create multiple instances", BrushTool, icon_brush),
             ("Put", "Put individual instances", PutInstanceTool, icon_put),
             ("Select", "Select and move instances", SelectTool, icon_select),
             #("Jitter", "Jitter instances", None, icon_jitter),
             #("Magnet", "Move (drag) multiple instances", None, icon_magnet),
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

        self.data = None

        self.undoStack = QtGui.QUndoStack(self)

        self.plot = PaintDataPlot(self.mainArea, "Painted Plot", widget=self)
        self.classValuesModel = itemmodels.PyListModel(
            ["class-1", "class-2"], self,
            flags=QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        self.connect(self.classValuesModel, QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.classNameChange)
        self.attr1 = "attribute-1"
        self.attr2 = "attribute-2"
        self.data = Table(
            Domain([ContinuousVariable(self.attr1), ContinuousVariable(self.attr2)],
            DiscreteVariable("Class label", values=self.classValuesModel)))

        self.toolsStackCache = {}

        self.initUI()
        self.initPlot()
        self.updatePlot()

    def initUI(self):
        undoRedoBox = gui.widgetBox(self.controlArea, "", addSpace=True)
        undo = QtGui.QAction("Undo", self)
        undo.pyqtConfigure(toolTip="Undo Action (Ctrl+Z)")
        undo.setShortcut("Ctrl+Z")
        self.connect(undo, QtCore.SIGNAL("triggered()"), self.undoStack.undo)
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

    def initPlot(self):
        self.plot.set_axis_title(owconstants.xBottom, self.data.domain[0].name)
        #self.plot.set_show_axis_title(owconstants.xBottom, True)
        self.plot.set_axis_title(owconstants.yLeft, self.data.domain[1].name)
        #self.plot.set_show_axis_title(owconstants.yLeft, True)
        self.plot.set_axis_scale(owconstants.xBottom, 0, 1, 0.1)
        self.plot.set_axis_scale(owconstants.yLeft, 0, 1, 0.1)
        self.updatePlot()

    def updatePlot(self):
        self.plot.legend().clear()
        colorDict = {}
        for i, value in enumerate(self.data.domain[2].values):
            rgb = self.plot.discrete_palette.getRGB(i)
            color = QtGui.QColor(*rgb)
            colorDict[i] = color
            self.plot.legend().add_item(
                self.data.domain[2].name, value,
                owpoint.OWPoint(owpoint.OWPoint.Diamond, color, 5))
        c_data = [colorDict[int(value)] for value in self.data.Y[:,0]]
        s_data = [5]*len(self.data.Y)
        self.plot.set_main_curve_data(list(self.data.X[:,0]), list(self.data.X[:,1]), color_data=c_data, label_data=[], size_data=s_data, shape_data=[owpoint.OWPoint.Diamond])
        self.plot.replot()

    def addNewClassLabel(self):
        self.classValuesModel.append("class-%d" % (len(self.classValuesModel)+1))
        newdomain = Domain([ContinuousVariable(self.attr1), ContinuousVariable(self.attr2)],
            DiscreteVariable("Class label", values=self.classValuesModel))
        command = CommandAddClassLabel(self.data, newdomain, self.classValuesModel, self, "Add Label")
        self.undoStack.push(command)
        print(self.classValuesModel)
        print(newdomain)

    def removeSelectedClassLabel(self):
        index = self.selectedClassLabelIndex()
        if index is not None:
            self.classValuesModel.pop(index)

    def classNameChange(self):
        pass

    def selectedClassLabelIndex(self):
        rows = [i.row() for i in self.classListView.selectionModel().selectedRows()]
        if rows:
            return rows[0]
        else:
            return None

    def setCurrentTool(self, tool):
        if tool not in self.toolsStackCache:
            newtool = tool(self)
            option = newtool.optionsWidget(newtool, self)
            self.optionsLayout.addWidget(option)
            # self.connect(newtool, QtCore.SIGNAL("editing()"), self.onDataChanged)
            # self.connect(newtool, QtCore.SIGNAL("editingFinished()"), self.commitIf)
            self.toolsStackCache[tool] = (newtool, option)

        self.currentTool, self.currentOptionsWidget = tool, option = self.toolsStackCache[tool]
        self.plot.tool = tool
        self.optionsLayout.setCurrentWidget(option)

    def addDataPoints(self, points):
        command = CommandAddData(self.data, points, self.classValuesModel[self.selectedClassLabelIndex()], self, "Add Data")
        self.undoStack.push(command)
        self.updatePlot()

    def sendData(self):
        self.send("Data", self.data)

    def testMethod(self, x="lol", y="troll"):
        print(x)
        print(y)

    def sizeHint(self):
        return QtCore.QSize(1200, 800)
