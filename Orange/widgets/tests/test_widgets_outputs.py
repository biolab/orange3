import re
import unittest
import importlib.util

from Orange.canvas.registry import global_registry


class TestWidgetOutputs(unittest.TestCase):
    def test_outputs(self):
        re_send = re.compile('\\n\s+self.send\("([^"]*)"')
        registry = global_registry()
        errors = []
        for desc in registry.widgets():
            signal_names = {output.name for output in desc.outputs}
            module_name, class_name = desc.qualified_name.rsplit(".", 1)
            fname = importlib.util.find_spec(module_name).origin
            with open(fname, encoding='utf-8') as f:
                widget_code = f.read()
            used = set(re_send.findall(widget_code))
            undeclared = used - signal_names
            if undeclared:
                errors.append("- {} ({})".
                              format(desc.name, ", ".join(undeclared)))
        if errors:
            self.fail("Some widgets send to undeclared outputs:\n"+"\n".
                      join(errors))
