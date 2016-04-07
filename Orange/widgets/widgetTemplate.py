from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting

class OWWidgetName(widget.OWWidget):
    name = "Widget Name"
    id = "orange.widgets.widget_category.widget_name"
    description = ""
    icon = "icons/Unknown.svg"
    author = ""
    maintainer_email = ""
    priority = 10
    category = ""
    keywords = ["list", "of", "keywords"]
    outputs = [("Name", type)]
    inputs = [("Name", type, "handler")]

    want_main_area = False

    foo = Setting(True)

    def __init__(self):
        super().__init__()

        # controls
        gui.rubber(self.controlArea)

    def handler(self, obj):
        pass
