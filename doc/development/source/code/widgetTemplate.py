from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWWidgetName(widget.OWWidget):
    name = "Widget Name"
    id = "orange.widgets.widget_category.widget_name"
    description = ""
    icon = "icons/Unknown.svg"
    priority = 10
    category = ""
    keywords = ["list", "of", "keywords"]
    want_main_area = False
    outputs = [("Name", type)]
    inputs = [("Name", type, "handler")]

    foo = Setting(True)

    def __init__(self):
        super().__init__()

        # controls
        gui.rubber(self.controlArea)
