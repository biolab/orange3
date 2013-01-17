# a simple module used in orange to see whether debugging is enabled, where to write the output and how much output do you want to see

orngDebuggingEnabled = 0                            # do we want to enable debugging
orngDebuggingFileName = "debuggingOutput.txt"       # where do we want to write output to
orngVerbosity = 1                                   # what's the level of verbosity

import weakref

from collections import defaultdict

import PyQt4.QtGui
from PyQt4 import QtCore, QtGui
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import random
class debug(object):
    elements_list = defaultdict(list)
    random = random

    @classmethod
    def registerQItemView(cls, widget, view):
        cls.elements_list[widget].append(view)

    @classmethod
    def regiseterQGraphicsView(cls, widget, view):
        cls.elements_list[widget].append(view)

    @classmethod
    def registerQwtPlot(cls, widget, graph):
        cls.elements_list[widget].append(graph)

    @classmethod
    def register(cls, widget, element):
        from PyQt4.QtGui import QGraphicsView, QAbstractItemView
        from OWGraph import OWGraph
        if isinstance(element, QGraphicsView):
            cls.regiseterQGraphicsView(widget, element)
        elif isinstance(element, QAbstractItemView):
            cls.registerQItemView(widget, element)
        elif isinstance(element, OWGraph):
            cls.registerOWGraph(widget, element)
        else:
            print("Unhandled type", element)

    @classmethod
    def scrollAreaInteract(cls, area):
    #        print "scrollAreaInteract", area
        from PyQt4.QtTest import QTest
        geom = area.geometry()
        randpos = lambda: geom.topLeft() + QtCore.QPoint(geom.width() * random.random(), geom.height() * random.random())
        QTest.mouseMove(area, randpos(), 2)
        QTest.mouseClick(area, Qt.LeftButton, pos=randpos(), delay=2)
        QTest.mouseDClick(area, Qt.LeftButton, pos=randpos(), delay=2)
        QTest.mousePress(area, Qt.LeftButton, pos=randpos(), delay=2)
        QTest.mouseRelease(area, Qt.LeftButton, pos=randpos(), delay=2)

    #        area.scrollContentsBy(random.randint(-10, 10), random.randint(-10, 10))


    @classmethod
    def itemViewInteract(cls, view):
        cls.scrollAreaInteract(view)

    @classmethod
    def graphicsViewInteract(cls, view):
        cls.scrollAreaInteract(view)

    @classmethod
    def graphInteract(cls, view):
        cls.scrollAreaInteract(view)

    @classmethod
    def interact(cls, widget):
        from PyQt4.QtGui import QGraphicsView, QAbstractItemView
        from OWGraph import OWGraph
        if isinstance(widget, QGraphicsView):
            cls.graphicsViewInteract(widget)
        elif isinstance(widget, QAbstractItemView):
            cls.itemViewInteract(widget)
        elif isinstance(widget, OWGraph):
            cls.graphInteract(widget)
        else:
            print("Unhandled widget interaction", widget)

    @classmethod
    def interactWithOWWidget(cls, widget):
        views = cls.candidateDebugWidgets(widget)
        for view in views:
            if view.isEnabled() and getattr(view, "debuggingEnabled", True):
                cls.interact(view)

    @classmethod
    def candidateDebugWidgets(cls, widget):
        from PyQt4.QtGui import QGraphicsView, QAbstractItemView
        from OWGraph import QwtPlot

        gviews_list = widget.findChildren(QGraphicsView)
        iviews_list = widget.findChildren(QAbstractItemView)
        pviews_list = widget.findChildren(QwtPlot)

        return gviews_list + iviews_list + pviews_list

#    @classmethod
#    def candidateDebugWidgets(cls, widget, cls_list=None):
#        if cls_list is None:
#            from PyQt4.QtGui import QGraphicsView, QAbstractItemView
#            from OWGraph import QwtPlot
#            cls_list = [QGraphicsView, QAbstractItemView, QwtPlot]
#        candidates = []
#        for child in widget.children():
#            print [(child, isinstance(child, db_cls)) for db_cls in cls_list]
#            if any(isinstance(child, db_cls) for db_cls in cls_list):
#                if getattr(child, "debuggingEnabled", True):
#                    candidates.append(child)
#                    candidates.extend(cls.candidateWidgets(child, cls_list))
#            else:
#                candidates.extend(cls.candidateWidgets(child, cls_list))
#        return candidates



