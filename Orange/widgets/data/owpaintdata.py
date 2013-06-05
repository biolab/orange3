from PyQt4 import QtCore
from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table

class OWPaintData(widget.OWWidget):
    _name = "Paint Data"
    _description = """
    Creates the data by painting on the graph."""
    _long_description = """
    """
    _icon = "icons/PaintData.svg"
    _author = "Martin Frlin"
    _maintainer_email = "janez.demsar(@at@)fri.uni-lj.si"
    _priority = 10
    _category = "Data"
    _keywords = ["data", "paint", "create"]
    outputs = [("Data", Table)]

    want_main_area = True

    commit_on_change = Setting(False)

    def __init__(self):
        self.data = None
        self.undoStack = QtGui.QUndoStack(self)

        self.initUI()

    def initUI(self):
        undoRedoBox = gui.widgetBox(self.controlArea, "Undo / Redo:", addSpace=True, orientation='horizontal')
        undoButton = gui.button(undoRedoBox, self, "Undo", callback=self.undoStack.undo)
        undoButton.setShortcut("Ctrl+Z")
        redoButton = gui.button(undoRedoBox, self, "Redo", callback=self.undoStack.redo)
        redoButton.setShortcut("Ctrl+Shift+Z")

        classesBox = gui.widgetBox(self.controlArea, "Classes:", addSpace=True)

        toolsBox = gui.widgetBox(self.controlArea, "Tools", orientation=QtGui.QGridLayout(), addSpace=True)

        optionsBox = gui.widgetBox(self.controlArea, "Options:", addSpace=True)

        commitBox = gui.widgetBox(self.controlArea, "Commit")
        gui.checkBox(commitBox, self, "commit_on_change", "Commit on change")
        gui.button(commitBox, self, "Commit", callback=self.sendData)

    def sendData(self):
        self.send("Data", self.data)

    def testMethod(self):
        print("it works!")


