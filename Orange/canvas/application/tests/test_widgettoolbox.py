"""
Tests for WidgetsToolBox.

"""
from PyQt4.QtGui import QWidget, QHBoxLayout
from PyQt4.QtCore import QSize

from ...registry import global_registry
from ...registry.qt import QtWidgetRegistry


from ..widgettoolbox import WidgetToolBox, WidgetToolGrid, ToolGrid

from ...gui import test


class TestWidgetToolBox(test.QAppTestCase):
    def test_widgettoolgrid(self):
        w = QWidget()
        layout = QHBoxLayout()

        reg = global_registry()
        qt_reg = QtWidgetRegistry(reg)

        triggered_actions1 = []
        triggered_actions2 = []

        model = qt_reg.model()
        data_descriptions = qt_reg.widgets("Data")

        file_action = qt_reg.action_for_widget(
            "Orange.widgets.data.owfile.OWFile"
        )

        actions = list(map(qt_reg.action_for_widget, data_descriptions))

        grid = ToolGrid(w)
        grid.setActions(actions)
        grid.actionTriggered.connect(triggered_actions1.append)
        layout.addWidget(grid)

        grid = WidgetToolGrid(w)

        # First category ("Data")
        grid.setModel(model, rootIndex=model.index(0, 0))

        self.assertIs(model, grid.model())

        # Test order of buttons
        grid_layout = grid.layout()
        for i in range(len(actions)):
            button = grid_layout.itemAtPosition(i / 4, i % 4).widget()
            self.assertIs(button.defaultAction(), actions[i])

        grid.actionTriggered.connect(triggered_actions2.append)

        layout.addWidget(grid)

        w.setLayout(layout)
        w.show()
        file_action.trigger()

        self.app.exec_()

    def test_toolbox(self):

        w = QWidget()
        layout = QHBoxLayout()

        reg = global_registry()
        qt_reg = QtWidgetRegistry(reg)

        triggered_actions = []

        model = qt_reg.model()

        file_action = qt_reg.action_for_widget(
            "Orange.widgets.data.owfile.OWFile"
        )

        box = WidgetToolBox()
        box.setModel(model)
        box.triggered.connect(triggered_actions.append)
        layout.addWidget(box)

        box.setButtonSize(QSize(50, 80))

        w.setLayout(layout)
        w.show()

        file_action.trigger()

        box.setButtonSize(QSize(60, 80))
        box.setIconSize(QSize(35, 35))
        box.setTabButtonHeight(40)
        box.setTabIconSize(QSize(30, 30))

        self.app.exec_()
