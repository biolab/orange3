import inspect

def indent(level, line):
    """ Indents line 4*level spaces """
    return (level * " ") + line

def lines_for(in_str):
    return in_str.split("\n")

def str_for(in_lines):
    return in_lines.join("\n")

class CodeGenerator(object):
    """
    Framework to generate static Python code that performs
    the core functionality of a widget.

    """
    def __init__(self):
        self.name = "unnamed_widget"
        self.orig_widget = None
        self.imports = []
        self.externs = []
        self.inits = {}
        self.attrs = {}
        self.body = []
        self.inputs = {}
        self.outputs = {}
        self.null_lines = []

    def set_widget(self, widget):
        """
        Provides a copy of the original widget for internal use
        by the code generator.

        """
        self.orig_widget = widget

    def add_import(self, extern):
        """
        Adds an import for the supplied exernal variable to the start of
        the exported script.

        """
        if type(extern) == list:
            for elem in extern:
                self.imports.append(extern)
        else:
            self.imports.append(extern)

    def add_attr(self, **kwargs):
        """
        Adds an attribute to the code generator object.  If just
        name= is supplied, will assume it's an attribute of the widget.
        If name= and var= are suppplied, will use value of var.
        name= can also supply a list

        """
        def _add_attr(widget, name, **kwargs):
            if "value" not in kwargs:
                value = getattr(self.orig_widget, name)
            else:
                value = kwargs["value"]
            self.attrs[name] = value

        if "var" in kwargs:
            self.attrs[kwargs["name"]] = kwargs["var"]
        else:
            if type(kwargs["name"]) == list:
                for name in kwargs["name"]:
                    widget_attr = getattr(self.orig_widget, name)
                    self.attrs[name] = widget_attr
            else:
                widget_attr = getattr(self.orig_widget, kwargs["name"])
                self.attrs[name] = value

    def add_init(self, name, value):
        """ Adds a declaration to the __init__ of the generated class """



    def add_extern(self, extern):
        """ Adds a function not inside of class """
        self.externs.append(extern)

    def add_input(self, input_name, input_val):
        """ Adds a named input to the list of inputs for the widget. """
        self.inputs[input_name] = input_val

    def add_body(self, body_func):
        """ Adds a function to the body of the generated code """
        self.body.append(body_func)

    def send(self, channel, data):
        """ Sets output """
        self.outputs[channel] = data

    def null_ln(self, nullstr):
        """ Removes every line that contains `nullstr` in output """
        if type(nullstr) == list:
            self.null_lines.extend(nullstr)
        else:
            self.null_lines.append(nullstr)

    def generate(self):
        """
        Creates a piece of code representative of this class.  The
        code will get input from the defined named inputs and
        set its outputs to the result.

        Returns
        -------
        (preamble_lines, body_code,)

        """
        preamble = ""
        for dependency in self.imports:
            try:
                preamble += "from " + dependency.__module__
                preamble += " import " + dependency.__name__ + "\n"
            except:
                pass

        body = ""
        for extern in self.externs:
            body += inspect.getsource(extern) + "\n"
        body += "\n"

        body += "class " + self.name + "():\n"
        for attrName, attrValue in self.attrs.items():
            print("Generating source for " + attrName)
            try:
                lines = lines_for(inspect.getsource(attrValue))
                for line in lines:
                    body += indent(1, line) + "\n"
                body += "\n"
            except:
                print("failed!")

        # TODO: Body gen

        # TODO: Null lines

        return preamble, body
