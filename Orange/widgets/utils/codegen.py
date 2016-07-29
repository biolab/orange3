import inspect

def indent(level, line, strip=False):
    """ Indents line 4*level spaces """
    if strip:
        line = line.lstrip()
    return ((4*level) * " ") + line

def zerodent(lines, n):
    """
    Removes n characters from the beginning of every line execpt
    lines containing "def".

    """
    for i, line in enumerate(lines):
        if "def" not in line:
            # Strip excess indentation
            lines[i] = line[n:]
    return lines

def unindent(level, line):
    """ Unindents line 4*level spaces """
    return line[level:]

def lines_for(in_str):
    return in_str.split("\n")

def str_for(in_lines):
    return ("\n").join(in_lines)

def gen_declaration(declarName, declarValue, iscode=False):
    """ Convertes a name + value into a variable declaration """
    good_types = [int, float, bool, tuple]
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
    def __init__(self, loadsettings=False):
        self.loadsettings = loadsettings
        self.name = "unnamed_widget"
        self.orig_widget = None
        self.imports = set()
        self.preambles = []
        self.externs = []
        self.inits = []
        self.code_inits = []
        self.attrs = {}
        self.attrs_code = []
        self.inputs = []
        self.outputs = []
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
            self.imports |= set([extern])

    def add_preamble(self, func):
        """
        Inserts `func` before all widget code.
        `func` can either be a function or string.

        """
        if type(func) == str:
            self.preambles.append("def x():\n" + (12 * " ") + func)
        else:
            self.preambles.append(inspect.getsource(func))

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
        Adds a declaration to the code generator object after the main
        function.  If just name= is supplied, will assume it's an
        attribute of the widget. If name= and var= are suppplied,
        will use value of var.  name= can also supply a list

        """
        if "var" in kwargs:
            self.attrs[kwargs["name"]] = kwargs["var"]
            if "iscode" in kwargs and kwargs["iscode"]:
                self.attrs_code.append(kwargs["name"])
        else:
            if type(kwargs["name"]) == list:
                for name in kwargs["name"]:
                    widget_attr = getattr(self.orig_widget, name)
                    self.attrs[name] = widget_attr
            else:
                widget_attr = getattr(self.orig_widget, kwargs["name"])
                self.attrs[kwargs["name"]] = widget_attr

    def add_input(self, source_node, source_channel, sink_channel):
        self.inputs.append((source_node, source_channel, sink_channel))

    def add_output(self, channel, value, iscode=False):
        """ Adds an output statement with the given channel """
        self.outputs[channel.lower()] = (value, iscode,)

    def add_extern(self, extern):
        """ Adds a function to the output code """
        self.externs.append(extern)

    def null_ln(self, nullstr):
        """ Removes every line that contains `nullstr` in output """
        if type(nullstr) == list:
            self.null_lines.extend(nullstr)
        else:
            self.null_lines.append(nullstr)

    def add_repl_map(self, replacements):
        """ replaces instances of a string in generated code """
        if type(replacements) == list:
            for repl in replacements:
                self.replacements.append(repl)
        else:
            self.replacements.append(replacements)

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
            # Try importing it as a function/submodule
            try:
                importString = "from " + dependency.__module__
                preamble.add(importString +
                    " import " + dependency.__name__)
            except:
                # Import it as a module
                preamble.add("import " + dependency.__name__)
        body += "\n"

        # External function generation
        for extern in self.externs:
            body_lines = inspect.getsourcelines(extern)[0]
            _indent = len(body_lines[0]) - len(body_lines[0].lstrip())
            for line in body_lines:
                body += unindent(_indent, line)
            body += "\n"
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
        body += "\n"

        def format_constant(constant):
            """
            Converts a constant into a format that can ne inserted
            into the output code.

            """
            if type(constant) in (float, int, tuple, bool,):
                return str(constant)
            elif(type(constant) == str):
                return repr(constant)
            else:
                raise TypeError("unparsableType")

        # Creates a statement to load settings into a widget object
        # Assumes a widget named `ow` has been initialized
        bad_settings = ["auto_apply", "savedWidgetGeometry"]
        if self.loadsettings:
            settings = self.orig_widget.settingsHandler.pack_data(self.orig_widget)
            body += "ow.settingsHandler.initialize(ow, data={\n"
            for sName, sVal in settings.items():
                if sName not in bad_settings:
                    try:
                        body += indent(1, "\"" + sName + "\": " +
                                format_constant(sVal)) + ",\n"
                    except: # Setting is of an unstringifiable type
                        print("Could not load setting; unstringifiable type")
            body += "})\n\n"

        # input declaration
        for inpt in self.inputs:
            body += "input_" + inpt[2] + " = "
            body += inpt[0].lower() + "_"
            body += inpt[1] + "\n"

        # main function
        if self.main_func is not None:
            main_lines = inspect.getsourcelines(self.main_func)[0]
            main_lines = zerodent(main_lines, 12)
            main_code = "".join(main_lines)
            body += strip_declaration(main_code)

        # attributes generation
        for attrName, attrValue in self.attrs.items():
            # If it's a code-based object, insert code
            try:
                lines = lines_for(inspect.getsource(attrValue))
                for line in lines:
                    body += unindent(4, line) + "\n"
            # Non-code object
            except:
                declaration = gen_declaration(attrName, attrValue, iscode=attrName in self.attrs_code)
                if declaration:
                    body += declaration
                else:
                    print("Unable to add attribute " + attrName)

        # output generation
        for name, value in self.outputs.items():
            declaration = gen_declaration(self.name + "_" + name, value[0], iscode=value[1])
            body += declaration

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
