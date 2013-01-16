"""
Test for canvas toolbox.
"""

from PyQt4.QtGui import QWidget, QToolBar, QTextEdit, QSplitter
from PyQt4.QtCore import Qt

from ...registry import global_registry
from ...registry.qt import QtWidgetRegistry
from ...gui.dock import CollapsibleDockWidget

from ..canvastooldock import WidgetToolBox, CanvasToolDock, SplitterResizer, \
                             QuickCategoryToolbar

from ...gui import test


class TestCanvasDockWidget(test.QAppTestCase):
    def test_dock(self):
        reg = global_registry()
        reg = QtWidgetRegistry(reg)

        toolbox = WidgetToolBox()
        toolbox.setObjectName("widgets-toolbox")
        toolbox.setModel(reg.model())
        text = QTextEdit()
        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)

        splitter.addWidget(toolbox)
        splitter.addWidget(text)

        dock = CollapsibleDockWidget()
        dock.setExpandedWidget(splitter)

        toolbar = QToolBar()
        toolbar.addAction("1")
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        dock.setCollapsedWidget(toolbar)

        dock.show()
        self.app.exec_()

    def test_canvas_tool_dock(self):
        reg = global_registry()
        reg = QtWidgetRegistry(reg)

        dock = CanvasToolDock()
        dock.toolbox.setModel(reg.model())

        dock.show()
        self.app.exec_()

    def test_splitter_resizer(self):
        w = QSplitter(orientation=Qt.Vertical)
        w.addWidget(QWidget())
        text = QTextEdit()
        w.addWidget(text)
        resizer = SplitterResizer(w)
        resizer.setSplitterAndWidget(w, text)

        def toogle():
            if resizer.size() == 0:
                resizer.open()
            else:
                resizer.close()
            self.singleShot(1000, toogle)

        w.show()
        self.singleShot(0, toogle)
        self.app.exec_()

    def test_category_toolbar(self):
        reg = global_registry()
        reg = QtWidgetRegistry(reg)

        w = QuickCategoryToolbar()
        w.setModel(reg.model())
        w.show()

        def p(action):
            print action.text()

        self.app.exec_()
