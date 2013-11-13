import os.path

from PyQt4.QtCore import SIGNAL, Qt
from PyQt4.QtGui import QToolButton, QGroupBox, QIcon, QHBoxLayout, QWidget, QVBoxLayout

from Orange.canvas.utils import environ
from Orange.widgets.gui import widgetBox


icons = os.path.join(environ.widget_install_dir, "icons")
dlg_zoom = os.path.join(icons, "Dlg_zoom.png")
dlg_zoom_selection = os.path.join(icons, "Dlg_zoom_selection.png")
dlg_pan = os.path.join(icons, "Dlg_pan_hand.png")
dlg_select = os.path.join(icons, "Dlg_arrow.png")
dlg_rect = os.path.join(icons, "Dlg_rect.png")
dlg_poly = os.path.join(icons, "Dlg_poly.png")
dlg_zoom_extent = os.path.join(icons, "Dlg_zoom_extent.png")
dlg_undo = os.path.join(icons, "Dlg_undo.png")
dlg_clear = os.path.join(icons, "Dlg_clear.png")
dlg_send = os.path.join(icons, "Dlg_send.png")
dlg_browseRectangle = os.path.join(icons, "Dlg_browseRectangle.png")
dlg_browseCircle = os.path.join(icons, "Dlg_browseCircle.png")


def createButton(parent, text, action=None, icon=None, toggle=0):
    btn = QToolButton(parent)
    btn.setMinimumSize(30, 30)
    if parent.layout() is not None:
        parent.layout().addWidget(btn)
    btn.setCheckable(toggle)
    if action:
        parent.connect(btn, SIGNAL("clicked()"), action)
    if icon:
        btn.setIcon(icon)
    btn.setToolTip(text)
    return btn


class ZoomSelectToolbar(QGroupBox):
#                (tooltip, attribute containing the button, callback function, button icon, button cursor, toggle)
    IconSpace, IconZoom, IconPan, IconSelect, IconRectangle, IconPolygon, \
    IconRemoveLast, IconRemoveAll, IconSendSelection, IconZoomExtent, \
    IconZoomSelection = range(11)

    DefaultButtons = 1, 4, 5, 0, 6, 7, 8
    SelectButtons = 3, 4, 5, 0, 6, 7, 8
    NavigateButtons = 1, 9, 10, 0, 2

    def __init__(self, widget, parent, graph, autoSend=0, buttons=(1, 4, 5, 0, 6, 7, 8), name="Zoom / Select",
                 exclusiveList="__toolbars"):
        if not hasattr(ZoomSelectToolbar, "builtinFunctions"):
            ZoomSelectToolbar.builtinFunctions = \
                (None,
                 ("Zooming", "buttonZoom", "activate_zooming", QIcon(dlg_zoom), Qt.ArrowCursor, 1),
                 ("Panning", "buttonPan", "activate_panning", QIcon(dlg_pan), Qt.OpenHandCursor, 1),
                 ("Selection", "buttonSelect", "activate_selection", QIcon(dlg_select), Qt.CrossCursor, 1),
                 ("Rectangle selection", "buttonSelectRect", "activate_rectangle_selection", QIcon(dlg_rect),
                  Qt.CrossCursor, 1),
                 ("Polygon selection", "buttonSelectPoly", "activate_polygon_selection", QIcon(dlg_poly), Qt.CrossCursor,
                  1),
                 (
                     "Remove last selection", "buttonRemoveLastSelection", "removeLastSelection", QIcon(dlg_undo), None,
                     0),
                 ("Remove all selections", "buttonRemoveAllSelections", "removeAllSelections", QIcon(dlg_clear), None,
                  0),
                 ("Send selections", "buttonSendSelections", "sendData", QIcon(dlg_send), None, 0),
                 ("Zoom to extent", "buttonZoomExtent", "zoomExtent", QIcon(dlg_zoom_extent), None, 0),
                 ("Zoom selection", "buttonZoomSelection", "zoomSelection", QIcon(dlg_zoom_selection), None, 0)
                )

        QGroupBox.__init__(self, name, parent)
        self.setLayout(QHBoxLayout())
        self.layout().setMargin(6)
        self.layout().setSpacing(4)
        if parent.layout() is not None:
            parent.layout().addWidget(self)

        self.graph = graph # save graph. used to send signals
        self.exclusiveList = exclusiveList

        self.widget = None
        self.functions = [type(f) == int and self.builtinFunctions[f] or f for f in buttons]
        for b, f in enumerate(self.functions):
            if not f:
                self.layout().addSpacing(10)
            else:
                button = createButton(self, f[0], lambda x=b: self.action(x), f[3], toggle=f[5])
                setattr(self, f[1], button)
                if f[1] == "buttonSendSelections":
                    button.setEnabled(not autoSend)

        if not hasattr(widget, exclusiveList):
            setattr(widget, exclusiveList, [self])
        else:
            getattr(widget, exclusiveList).append(self)

        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection
        self.action(0)


    def action(self, b):
        f = self.functions[b]
        if not f:
            return

        if f[5]:
            if hasattr(self.widget, "toolbarSelection"):
                self.widget.toolbarSelection = b
            for tbar in getattr(self.widget, self.exclusiveList):
                for fi, ff in enumerate(tbar.functions):
                    if ff and ff[5]:
                        getattr(tbar, ff[1]).setChecked(self == tbar and fi == b)
        getattr(self.graph, f[2])()

        cursor = f[4]
        if not cursor is None:
            self.graph.setCursor(cursor)


    # for backward compatibility with a previous version of this class
    def actionZooming(self):
        self.action(0)

    def actionRectangleSelection(self):
        self.action(3)

    def actionPolygonSelection(self):
        self.action(4)


class NavigateSelectToolbar(QWidget):
#                (tooltip, attribute containing the button, callback function, button icon, button cursor, toggle)

    IconSpace, IconZoom, IconPan, IconSelect, IconRectangle, IconPolygon, IconRemoveLast, IconRemoveAll, IconSendSelection, IconZoomExtent, IconZoomSelection = list(
        range(11))

    def __init__(self, widget, parent, graph, autoSend=0, buttons=(1, 4, 5, 0, 6, 7, 8)):
        if not hasattr(NavigateSelectToolbar, "builtinFunctions"):
            NavigateSelectToolbar.builtinFunctions = (None,
                                                      ("Zooming", "buttonZoom", "activateZooming", QIcon(dlg_zoom),
                                                       Qt.CrossCursor, 1, "navigate"),
                                                      ("Panning", "buttonPan", "activatePanning", QIcon(dlg_pan),
                                                       Qt.OpenHandCursor, 1, "navigate"),
                                                      ("Selection", "buttonSelect", "activateSelection",
                                                       QIcon(dlg_select), Qt.ArrowCursor, 1, "select"),
                                                      ("Selection", "buttonSelectRect", "activateRectangleSelection",
                                                       QIcon(dlg_select), Qt.ArrowCursor, 1, "select"),
                                                      ("Polygon selection", "buttonSelectPoly",
                                                       "activatePolygonSelection", QIcon(dlg_poly), Qt.ArrowCursor, 1,
                                                       "select"),
                                                      ("Remove last selection", "buttonRemoveLastSelection",
                                                       "removeLastSelection", QIcon(dlg_undo), None, 0, "select"),
                                                      ("Remove all selections", "buttonRemoveAllSelections",
                                                       "removeAllSelections", QIcon(dlg_clear), None, 0, "select"),
                                                      ("Send selections", "buttonSendSelections", "sendData",
                                                       QIcon(dlg_send), None, 0, "select"),
                                                      ("Zoom to extent", "buttonZoomExtent", "zoomExtent",
                                                       QIcon(dlg_zoom_extent), None, 0, "navigate"),
                                                      ("Zoom selection", "buttonZoomSelection", "zoomSelection",
                                                       QIcon(dlg_zoom_selection), None, 0, "navigate")
            )

        QWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 25, 0, 0)

        if parent.layout() is not None:
            parent.layout().addWidget(self)

        self.navigate = widgetBox(self, 0, orientation="vertical", margin=2)
        self.select = widgetBox(self, "", orientation="vertical")

        self.graph = graph # save graph. used to send signals
        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection

        self.functions = [type(f) == int and self.builtinFunctions[f] or f for f in buttons]
        for b, f in enumerate(self.functions):
            if not f:
                #self.layout().addSpacing(10)
                pass
            elif f[0] == "" or f[1] == "" or f[2] == "":
                self.navigate.layout().addSpacing(10)
            else:
                button = createButton(self.navigate, f[0], lambda x=b: self.action(x), f[3], toggle=f[5])
                setattr(self.navigate, f[1], button)
                if f[1] == "buttonSendSelections":
                    button.setEnabled(not autoSend)

        if hasattr(self.widget, "toolbarSelection"):
            self.action(self.widget.toolbarSelection)
        else:
            self.action(0)

    def action(self, b):
        f = self.functions[b]
        if not f:
            return

        if f[5]:
            if hasattr(self.widget, "toolbarSelection"):
                self.widget.toolbarSelection = b
            for fi, ff in enumerate(self.functions):
                if ff and ff[5]:
                    #if ff[6] == "navigate":
                    getattr(self.navigate, ff[1]).setChecked(fi == b)
                    #if ff[6] == "select":
                    #    getattr(self.select, ff[1]).setChecked(fi == b)

        try:
            getattr(self.graph, f[2])()
        except:
            getattr(self.widget, f[2])()

        cursor = f[4]
        if not cursor is None:
            self.graph.setCursor(cursor)


    # for backward compatibility with a previous version of this class
    def actionZooming(self):
        self.action(0)

    def actionRectangleSelection(self):
        self.action(3)

    def actionPolygonSelection(self):
        self.action(4)
