# pylint: disable=all
from AnyQt.QtGui import QStandardItemModel, QIcon, QColor
from AnyQt.QtCore import Qt, QItemSelectionModel, QPoint
from AnyQt.QtWidgets import QStyleOptionHeader, QStyle
from AnyQt.QtTest import QTest


from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.headerview import HeaderView
from Orange.widgets.utils.textimport import StampIconEngine


class TestHeaderView(GuiTest):
    def test_header(self):
        model = QStandardItemModel()

        hheader = HeaderView(Qt.Horizontal)
        vheader = HeaderView(Qt.Vertical)
        hheader.setSortIndicatorShown(True)

        # paint with no model.
        vheader.grab()
        hheader.grab()

        hheader.setModel(model)
        vheader.setModel(model)

        hheader.adjustSize()
        vheader.adjustSize()
        # paint with an empty model
        vheader.grab()
        hheader.grab()

        model.setRowCount(1)
        model.setColumnCount(1)
        icon = QIcon(StampIconEngine("A", Qt.red))
        model.setHeaderData(0, Qt.Horizontal, icon, Qt.DecorationRole)
        model.setHeaderData(0, Qt.Vertical, icon, Qt.DecorationRole)
        model.setHeaderData(0, Qt.Horizontal, QColor(Qt.blue), Qt.ForegroundRole)
        model.setHeaderData(0, Qt.Vertical, QColor(Qt.blue), Qt.ForegroundRole)
        model.setHeaderData(0, Qt.Horizontal, QColor(Qt.white), Qt.BackgroundRole)
        model.setHeaderData(0, Qt.Vertical, QColor(Qt.white), Qt.BackgroundRole)

        # paint with single col/row model
        vheader.grab()
        hheader.grab()

        model.setRowCount(3)
        model.setColumnCount(3)

        hheader.adjustSize()
        vheader.adjustSize()

        # paint with single col/row model
        vheader.grab()
        hheader.grab()

        hheader.setSortIndicator(0, Qt.AscendingOrder)
        vheader.setHighlightSections(True)
        hheader.setHighlightSections(True)

        vheader.grab()
        hheader.grab()

        vheader.setSectionsClickable(True)
        hheader.setSectionsClickable(True)

        vheader.grab()
        hheader.grab()

        vheader.setTextElideMode(Qt.ElideRight)
        hheader.setTextElideMode(Qt.ElideRight)

        selmodel = QItemSelectionModel(model, model)

        vheader.setSelectionModel(selmodel)
        hheader.setSelectionModel(selmodel)

        selmodel.select(model.index(1, 1), QItemSelectionModel.Rows | QItemSelectionModel.Select)
        selmodel.select(model.index(1, 1), QItemSelectionModel.Columns | QItemSelectionModel.Select)

        vheader.grab()
        vheader.grab()

    def test_header_view_clickable(self):
        model = QStandardItemModel()
        model.setColumnCount(3)
        header = HeaderView(Qt.Horizontal)
        header.setModel(model)
        header.setSectionsClickable(True)
        header.adjustSize()
        pos = header.sectionViewportPosition(0)
        size = header.sectionSize(0)
        # center of first section
        point = QPoint(pos + size // 2, header.viewport().height() // 2)
        QTest.mousePress(header.viewport(), Qt.LeftButton, Qt.NoModifier, point)

        opt = QStyleOptionHeader()
        header.initStyleOptionForIndex(opt, 0)
        self.assertTrue(opt.state & QStyle.State_Sunken)

        QTest.mouseRelease(header.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        opt = QStyleOptionHeader()
        header.initStyleOptionForIndex(opt, 0)
        self.assertFalse(opt.state & QStyle.State_Sunken)
