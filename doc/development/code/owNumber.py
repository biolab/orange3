from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from PyQt4.QtGui import QIntValidator

class OWWidgetNumber(widget.OWWidget):
    name = "Number"
    id = "orange.widgets.data.number"
    description = "Lets the user input a number"
    icon = "icons/Unknown.svg"
    author = ""
    maintainer_email = ""
    priority = 10
    category = ""
    keywords = ["list", "of", "keywords"]
    outputs = [{"name": "Number",
                "type": int,
                "doc": "The number entered by the user"}]

    want_main_area = False

    number = Setting(42)

    def __init__(self):
        super().__init__()

        gui.lineEdit(self.controlArea, self, "number", "Enter a number",
                     orientation="horizontal", box="Number",
                     callback=self.number_changed,
                     valueType=int, validator=QIntValidator())
        self.number_changed()

    def number_changed(self):
        self.send("Number", self.number)