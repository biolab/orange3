from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Output
from Orange.widgets.settings import Setting
from PyQt4.QtGui import QIntValidator


class OWWidgetNumber(OWWidget):
    name = "Number"
    description = "Lets the user input a number"
    icon = "icons/Unknown.svg"
    priority = 10
    category = ""

    class Outputs:
        number = Output("Number", int)

    want_main_area = False

    number = Setting(42)

    def __init__(self):
        super().__init__()

        gui.lineEdit(
            self.controlArea,
            self,
            "number",
            "Enter a number",
            box="Number",
            callback=self.number_changed,
            valueType=int,
            validator=QIntValidator(),
        )
        self.number_changed()

    def number_changed(self):
        self.Outputs.number.send(self.number)
