import os.path

from PyQt4.QtCore import SIGNAL, Qt
from PyQt4.QtGui import QToolButton, QGroupBox, QIcon, QHBoxLayout

from Orange.canvas.utils import environ
from Orange.widgets.settings import Setting

SPACE = 0
ZOOM = 1
PAN = 2
SELECT = 3
RECTANGLE = 4
POLYGON = 5
REMOVE_LAST = 6
REMOVE_ALL = 7
SEND_SELECTION = 8
ZOOM_EXTENT = 9
ZOOM_SELECTION = 10

# attr name used to store toolbars on a widget
TOOLBARS_STORE = "__toolbars"

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


class ToolbarButton:
    def __init__(self, text, attr_name, ext_attr_name,
                 icon=None, cursor=None, selectable=False):
        self.text = text
        self.attr_name = attr_name
        self.ext_attr_name = ext_attr_name
        self.icon = icon
        self.cursor = cursor
        self.selectable = selectable


class ZoomSelectToolbar(QGroupBox):
    DefaultButtons = ZOOM, RECTANGLE, POLYGON, SPACE, REMOVE_LAST, REMOVE_ALL, SEND_SELECTION
    SelectButtons = SELECT, RECTANGLE, POLYGON, SPACE, REMOVE_LAST, REMOVE_ALL, SEND_SELECTION
    NavigateButtons = ZOOM, ZOOM_EXTENT, ZOOM_SELECTION, SPACE, PAN

    selected_button = Setting(0)

    @property
    def builtin_functions(self):
        if ZoomSelectToolbar._builtin_functions is None:
            ZoomSelectToolbar._builtin_functions = (
                None,
                ToolbarButton("Zooming", "buttonZoom", "activate_zooming",
                              QIcon(dlg_zoom), Qt.ArrowCursor, True),
                ToolbarButton("Panning", "buttonPan", "activate_panning",
                              QIcon(dlg_pan), Qt.OpenHandCursor, True),
                ToolbarButton("Selection", "buttonSelect", "activate_selection",
                              QIcon(dlg_select), Qt.CrossCursor, True),
                ToolbarButton("Rectangle selection", "buttonSelectRect", "activate_rectangle_selection",
                              QIcon(dlg_rect), Qt.CrossCursor, True),
                ToolbarButton("Polygon selection", "buttonSelectPoly", "activate_polygon_selection",
                              QIcon(dlg_poly), Qt.CrossCursor, True),
                ToolbarButton("Remove last selection", "buttonRemoveLastSelection", "removeLastSelection",
                              QIcon(dlg_undo)),
                ToolbarButton("Remove all selections", "buttonRemoveAllSelections", "removeAllSelections",
                              QIcon(dlg_clear)),
                ToolbarButton("Send selections", "buttonSendSelections", "sendData",
                              QIcon(dlg_send)),
                ToolbarButton("Zoom to extent", "buttonZoomExtent", "zoomExtent",
                              QIcon(dlg_zoom_extent)),
                ToolbarButton("Zoom selection", "buttonZoomSelection", "zoomSelection",
                              QIcon(dlg_zoom_selection))
            )
        return ZoomSelectToolbar._builtin_functions
    _builtin_functions = None

    def __init__(self, widget, parent, graph, auto_send=False, buttons=DefaultButtons, name="Zoom / Select"):
        widget.settingsHandler.initialize(self)
        QGroupBox.__init__(self, name, parent)

        self.widget_toolbars = self.register_toolbar(widget)

        self.widget = widget
        self.graph = graph
        self.auto_send = auto_send

        self.setup_toolbar(parent)
        self.buttons = self.add_buttons(buttons)

        self.action(self.selected_button)

    def register_toolbar(self, widget):
        if hasattr(widget, TOOLBARS_STORE):
            getattr(widget, TOOLBARS_STORE).append(self)
        else:
            setattr(widget, TOOLBARS_STORE, [self])
        return getattr(widget, TOOLBARS_STORE)

    def setup_toolbar(self, parent):
        self.setLayout(QHBoxLayout())
        self.layout().setMargin(6)
        self.layout().setSpacing(4)
        if parent.layout() is not None:
            parent.layout().addWidget(self)

    def add_buttons(self, buttons):
        buttons = [self.builtin_functions[f] if isinstance(f, int) else f for f in buttons]
        for i, button in enumerate(buttons):
            if not button:
                self.add_spacing()
            else:
                self.add_button(button, action=lambda x=i: self.action(x))
        return buttons

    def add_spacing(self):
        self.layout().addSpacing(10)

    def add_button(self, button: ToolbarButton, action=None):
        btn = QToolButton(self)
        btn.setMinimumSize(30, 30)
        if self.layout() is not None:
            self.layout().addWidget(btn)
        btn.setCheckable(button.selectable)
        if action:
            self.connect(btn, SIGNAL("clicked()"), action)
        if button.icon:
            btn.setIcon(button.icon)
        btn.setToolTip(button.text)

        setattr(self, button.attr_name, btn)
        if button.attr_name == "buttonSendSelections":
            btn.setEnabled(not self.auto_send)

        return btn

    def action(self, button_idx):
        button = self.buttons[button_idx]
        if not isinstance(button, ToolbarButton):
            return

        if button.selectable:
            self.selected_button = button_idx
            for toolbar in self.widget_toolbars:
                for ti, tbutton in enumerate(toolbar.buttons):
                    if isinstance(tbutton, ToolbarButton) and tbutton.selectable:
                        getattr(toolbar, tbutton.attr_name).setChecked(self == toolbar and ti == button_idx)
        getattr(self.graph, button.ext_attr_name)()

        if button.cursor is not None:
            self.graph.setCursor(button.cursor)
