"""
Tests for the DockWidget.

"""

from PyQt4.QtGui import QWidget, QMainWindow, QListView, QTextEdit, \
                        QToolButton, QStringListModel, QHBoxLayout, QLabel

from PyQt4.QtCore import Qt

from .. import test
from ..dock import CollapsibleDockWidget


class TestDock(test.QAppTestCase):
    def test_dock_standalone(self):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        layout.addStretch(1)
        widget.show()

        dock = CollapsibleDockWidget()
        layout.addWidget(dock)
        list_view = QListView()
        list_view.setModel(QStringListModel(["a", "b"], list_view))

        label = QLabel("A label. ")
        label.setWordWrap(True)

        dock.setExpandedWidget(label)
        dock.setCollapsedWidget(list_view)
        dock.setExpanded(True)

        self.app.processEvents()

        def toogle():
            dock.setExpanded(not dock.expanded())
            self.singleShot(2000, toogle)

        toogle()

        self.app.exec_()

    def test_dock_mainwinow(self):
        mw = QMainWindow()
        dock = CollapsibleDockWidget()
        w1 = QTextEdit()

        w2 = QToolButton()
        w2.setFixedSize(38, 200)

        dock.setExpandedWidget(w1)
        dock.setCollapsedWidget(w2)

        mw.addDockWidget(Qt.LeftDockWidgetArea, dock)
        mw.setCentralWidget(QTextEdit())
        mw.show()

        def toogle():
            dock.setExpanded(not dock.expanded())
            self.singleShot(2000, toogle)

        toogle()

        self.app.exec_()
