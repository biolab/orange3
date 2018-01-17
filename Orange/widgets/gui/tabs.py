from AnyQt.QtWidgets import QTabWidget, QScrollArea
from AnyQt.QtCore import Qt

from .boxes import vBox


def tabWidget(widget):
    w = QTabWidget(widget)
    if widget.layout() is not None:
        widget.layout().addWidget(w)
    return w


def createTabPage(tab_widget, name, widgetToAdd=None, canScroll=False):
    if widgetToAdd is None:
        widgetToAdd = vBox(tab_widget, addToLayout=0, margin=4)
    if canScroll:
        scrollArea = QScrollArea()
        tab_widget.addTab(scrollArea, name)
        scrollArea.setWidget(widgetToAdd)
        scrollArea.setWidgetResizable(1)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    else:
        tab_widget.addTab(widgetToAdd, name)
    return widgetToAdd
