import inspect

def indent(level, line): # TODO: Detect existing whitespace of first line and intelligently indent
    """ Indents line 4*level spaces """
    return ((4*level) * " ") + line

def lines_for(in_str):
    return in_str.split("\n")

def str_for(in_lines):
    return ("\n").join(in_lines)

def gen_declaration(declarName, declarValue, iscode=False):
    """ Convertes a name + value into a variable declaration """
    good_types = [int, float, tuple]
    if type(declarValue) in good_types:
        declaration = declarName + " = "
        declaration += str(declarValue)
        return declaration + "\n"
    elif type(declarValue) == str:
        declaration = declarName + " = "
        if not(iscode):
            declaration += "\""
        declaration += declarValue
        if not(iscode):
            declaration += "\""
        return declaration + "\n"
    else:
        print("Unable to declare " + declarName + " of " + str(type(declarValue)))
        return False

def trim_declaration(code):
    """
    Takes a snipped of Python code for a function and snips off
    all lines through e.g. `def foo():`

    """
    code_lines = lines_for(code)
    i = 0
    while i < len(code_lines):
        if ("def " not in code_lines[i]) and \
            "class " not in code_lines[i]:
            del code_lines[i]
            i -= 1
        else:
            del code_lines[i]
            break
        i += 1
    return str_for(code_lines)

class CodeGenerator(object):
    """
    Framework to generate static Python code that performs
    the core functionality of a widget.

    """
    def __init__(self):
        self.name = "unnamed_widget"
        self.orig_widget = None
        self.imports = set()
        self.externs = []
        self.inits = []
        self.code_inits = []
        self.attrs = {}
        self.main_func = None
        self.inputs = {}
        self.outputs = {}
        self.null_lines = []
        self.replacements = []

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
            self.imports |= set(extern)
        else:
            self.imports.append(extern)

    def add_init(self, name, value, iscode=False):
        """ Adds a declaration to the __init__ of the generated class """
        if iscode:
            self.code_inits.append(name)
        self.inits = [(name, value,)] + self.inits

    def set_main_func(self, func):
        """ Adds a function to the body of the generated code """
        self.main_func = func

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

    def add_extern(self, extern):
        """ Adds a function not inside of class """
        self.externs.append(extern)

    def add_input(self, input_name, input_val):
        """ Adds a named input to the list of inputs for the widget. """
        self.inputs[input_name] = input_val

    ###Template functions for copying into output###
    def send(self, channel, data):
        """ Sets output """
        self.outputs[channel] = data

    class info():
        def setText(data):
            print(data)

    def null_ln(self, nullstr):
        """ Removes every line that contains `nullstr` in output """
        if type(nullstr) == list:
            self.null_lines.extend(nullstr)
        else:
            self.null_lines.append(nullstr)

    def add_repl_map(self, replacements):
        """ replaces instances of a string in generated code """
        for repl in replacements:
            self.replacements.append(repl)

    def set_name(self, name):
        self.name = name

    def generate(self):
        """
        Creates a piece of code representative of this class.  The
        code will get input from the defined named inputs and
        set its outputs to the result.

        Returns
        -------
        (preamble_lines, body_code,)

        """
        preamble = set()
        body = ""

        # Imports generation
        for dependency in self.imports:
            try:
                importString = "from " + dependency.__module__
                preamble.add(importString +
                    " import " + dependency.__name__)
            except:
                preamble.add("import " + dependency.__name__)
        body += "\n"

        # External function generation
        for extern in self.externs:
            body += inspect.getsource(extern) + "\n"
        body += "\n"

        # Class generation
        body += "class " + self.name + "():\n"

        # __init__ generation
        body += indent(1, "def __init__(self):\n")
        # Manually construct outputs dict
        body += indent(2, "self.outputs = {}\n")
        if len(self.inits) == 0:
            body += indent(2, "# No code generator defined for this widget\n")
            body += indent(2, "pass")
        for init_pair in reversed(self.inits):
            initName, initValue = init_pair
            iscode = initName in self.code_inits
            declaration = gen_declaration(initName, initValue, iscode=iscode)
            if declaration:
                body += indent(2, "self." + declaration)
        body += "\n"

        # Create send() method
        body += inspect.getsource(self.send) + "\n"

        # Create custom info object.setText() that print()s
        body += inspect.getsource(self.info) + "\n"

        # class attributes generation
        for attrName, attrValue in self.attrs.items():
            # If it's a code-based object, insert code
            try:
                lines = lines_for(inspect.getsource(attrValue))
                for line in lines:
                    body += indent(0, line) + "\n"
                body += "\n"
            # Non-code object
            except:
                declaration = gen_declaration(attrName, attrValue)
                if declaration:
                    body += indent(1, declaration)
                else:
                    print("Unable to add attribute " + attrName)
        body += "\n"

        main_code = inspect.getsource(self.main_func)
        main_code = trim_declaration(main_code)
        body += indent(1, "def get_output(self):\n")
        body += main_code + "\n"

        body_lines = lines_for(body)
        for i, line in enumerate(body_lines):
            # line deletion
            for match in self.null_lines:
                if match in line:
                    # Replace line with deleting placeholder to
                    # preserve indexes during iteration
                    body_lines[i] = "#***TODEL***"
            # line replacement
            for match in self.replacements:
                if match[0] in line:
                    body_lines[i] = line.replace(match[0], match[1])
        body = str_for(body_lines)

        return preamble, body.replace("#***TODEL***\n", "")
