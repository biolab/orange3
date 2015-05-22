from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting

class OWMultiplier(widget.OWWidget):
    name = "Multiplier"
    id = "orange.widgets.data.multiplier"
    description = ""
    icon = "icons/Unknown.svg"
    author = ""
    maintainer_email = ""
    priority = 10
    category = ""
    keywords = ["list", "of", "keywords"]
    outputs = [("Product", int)]
    inputs = [("A number", int, "get_a_number")]

    want_main_area = False

    factor = Setting(True)

    def __init__(self):
        super().__init__()

        self.n = None
        self.product = 0

        gui.radioButtonsInBox(self.controlArea, self, "factor",
            ("None", "Double", "Triple", "Quadruple"),
            box="Multiply", callback=self.do_multiply)

        self.result = gui.label(self.controlArea, self,
                                "The product is %(product)i",
                                box="Result")
        self.result.hide()

        gui.rubber(self.controlArea)

    def get_a_number(self, n):
        self.n = n
        self.do_multiply()

    def do_multiply(self):
        if self.n is None:
            self.result.hide()
            self.send("Product", None)
        else:
            self.result.show()
            self.product = self.n * (self.factor + 1)
            self.send("Product", self.product)
