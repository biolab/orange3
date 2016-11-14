# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import os
import pickle
from tempfile import mkstemp

from Orange.classification.majority import ConstantModel
from Orange.widgets.classify.owloadclassifier import OWLoadClassifier
from Orange.widgets.tests.base import WidgetTest


class TestOWMajority(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWLoadClassifier)

    def test_show_error(self):
        self.widget.load("no-such-file.pckls")
        self.assertTrue(self.widget.Error.load_error.is_shown())

        clsf = ConstantModel([1, 1, 1])
        fd, fname = mkstemp(suffix='.pkcls')
        try:
            pickle.dump(clsf, open(fname, "wb"))
            self.widget.load(fname)
            self.assertFalse(self.widget.Error.load_error.is_shown())

            open(fname, "w").write("X")
            self.widget.load(fname)
            self.assertTrue(self.widget.Error.load_error.is_shown())
        finally:
            os.remove(fname)

