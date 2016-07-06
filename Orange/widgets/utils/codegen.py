import inspect

def indent(level, line, strip=False):
    """ Indents line 4*level spaces """
    if strip:
        line = line.lstrip()
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

def strip_declaration(code):
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

    def add_init(self, name, value, scrape=False, iscode=False):
        """
        Sets a variable equal to a value in the code segment for a widget.
        When scrape=True, all matching lines from the source code starting from
        `name` and ending when the indent returns to the same level as the start
        line are inserted into the output.
        When iscode=True, `value` is treated like code and inserted directly.

        """
        def get_indent(line):
            return len(line) - len(line.lstrip())

        if iscode:
            self.code_inits.append(name)
        elif scrape:
            res = []
            source = inspect.getsourcelines(widget.__init__)
            for i, line in enumerate(source):
                if name in line:
                    ix = i
                    break
            startline = source[ix]
            res.append(startline)
            ix += 1
            while get_indent(source[ix]) > get_indent(startline):
                line = source[ix]
                res.append(line)
                indent = get_indent(line)
                ix += 1
            value = "\n".join(reverse(res))
        self.inits = [(name, value,)] + self.inits

    def set_main(self, func):
        """ Sets the main funciton that's executed to produce output """
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
        """ Adds a function to the output code """
        self.externs.append(extern)

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

        # Main code block generation
        body += "#\n#" + self.name + "\n#\n"

        # initial declarations
        for init_pair in reversed(self.inits):
            initName, initValue = init_pair
            iscode = initName in self.code_inits
            declaration = gen_declaration(initName, initValue, iscode=iscode)
            if declaration:
                body += declaration

        # Create send() method
        #body += inspect.getsource(self.send) + "\n"

        # Create custom info object.setText() that print()s
        #body += inspect.getsource(self.info) + "\n"

        # class attributes generation
        for attrName, attrValue in self.attrs.items():
            # If it's a code-based object, insert code
            try:
                lines = lines_for(inspect.getsource(attrValue))
                for line in lines:
                    body += line + "\n"
            # Non-code object
            except:
                declaration = gen_declaration(attrName, attrValue)
                if declaration:
                    body += indent(1, declaration)
                else:
                    print("Unable to add attribute " + attrName)

        # main function
        body += indent(0, strip_declaration(inspect.getsource(self.main_func)), strip=True)

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
