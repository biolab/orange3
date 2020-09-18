from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont, QColor

from orangewidget.tests.base import GuiTest
from Orange.widgets.utils.combobox import ItemStyledComboBox


class TestItemStyledComboBox(GuiTest):
    def test_combobox(self):
        cb = ItemStyledComboBox()
        cb.setPlaceholderText("...")
        self.assertEqual(cb.placeholderText(), "...")
        cb.grab()
        cb.addItems(["1"])
        cb.setCurrentIndex(0)
        model = cb.model()
        model.setItemData(model.index(0, 0), {
            Qt.ForegroundRole: QColor(Qt.blue),
            Qt.FontRole: QFont("Windings")
        })
        cb.grab()
