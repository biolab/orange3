# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import os
import pickle
from tempfile import mkstemp

from Orange.classification.majority import ConstantModel
from Orange.widgets.model.owloadmodel import OWLoadModel
from Orange.widgets.tests.base import WidgetTest


class TestOWLoadModel(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWLoadModel)

    def test_show_error(self):
        self.widget.load("no-such-file.pckls")
        self.assertTrue(self.widget.Error.load_error.is_shown())

        clsf = ConstantModel([1, 1, 1])
        fd, fname = mkstemp(suffix='.pkcls')
        os.close(fd)
        try:
            with open(fname, 'wb') as f:
                pickle.dump(clsf, f)
            self.widget.load(fname)
            self.assertFalse(self.widget.Error.load_error.is_shown())

            with open(fname, "w") as f:
                f.write("X")
            self.widget.load(fname)
            self.assertTrue(self.widget.Error.load_error.is_shown())
        finally:
            os.remove(fname)

