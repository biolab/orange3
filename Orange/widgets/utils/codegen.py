import inspect

class CodeGenerator(object):
    """
    This is the default version of the function that replicates the
    results of the output of a widget.  It is used with the Python
    script exportation pipeline to generate a static script to accomplish
    the actions carried out in the widget statically.

    """
    def __call__(self):
        return self

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

    def set_preamble_gen(self, gen):
        self.gen_preamble = gen

    def set_header_gen(self, gen):
        self.gen_declarations = gen

    def set_body_gen(self, gen):
        self.gen_body = gen

    @classmethod
    def set_gens(cls, _preamble_gen, _header_gen, _body_gen):
        cls.gen_preamble = _preamble_gen
        cls.gen_declarations = _preamble_gen
        cls.gen_body = _body_gen
