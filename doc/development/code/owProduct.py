from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting

class OWProduct(widget.OWWidget):
    name = "Product"
    id = "orange.widgets.data.multiplier"
    description = ""
    icon = "icons/Unknown.svg"
    author = ""
    maintainer_email = ""
    priority = 10
    category = ""
    keywords = ["list", "of", "keywords"]
    outputs = [{"name": "Product",
                "type": int,
                "doc": ""}]

    inputs = [{"name": "First factor", "type": int, "handler": "get_first"},
              {"name": "Second factor", "type": int, "handler": "get_second"}]

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.first = self.second = None
        self.product = None

        self.result = gui.label(self.controlArea, self,
                                "%(first)s times %(second)s is %(product)s",
                                box="Result")
        self.result.hide()

    def get_first(self, n):
        self.first = n
        self.do_multiply()

    def get_second(self, n):
        self.second = n
        self.do_multiply()

    def do_multiply(self):
        if self.first and self.second is None:
            self.result.hide()
            self.send("Product", None)
        else:
            self.result.show()
            self.product = self.first * self.second
            self.send("Product", self.product)
