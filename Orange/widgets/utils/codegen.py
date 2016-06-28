import inspect

class CodeGenerator(object):
    """
    This is the default version of the function that replicates the
    results of the output of a widget.  It is used with the Python
    script exportation pipeline to generate a static script to accomplish
    the actions carried out in the widget statically.

    """
    def gen_preamble(self):
        """
        A list of expressions to be tacked on the the top of the
        script; duplicate expressions will be ignored.

        """
        pass

    def gen_declarations(self):
        """
        Variable name conversions to be inserted before the start
        of the main module code

        """
        pass

    def gen_body(self):
        """
        A function that returns the code to generate the output of
        the widget in a static way.

        """
        pass

    def generate(self):
        """
        Returns
        -------
        (preamble_string, declar_string, body_string,)

        """
        preamble_string = inspect.getsourcelines(self.gen_preamble)
        declar_string = inspect.getsourcelines(self.gen_declarations)
        body_string = inspect.getsourcelines(self.gen_body)

        return (preamble_string, declar_string, body_string,)
