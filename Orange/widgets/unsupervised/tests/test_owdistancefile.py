import os
import unittest

import numpy as np

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owdistancefile \
    import OWDistanceFile, OWDistanceFileDropHandler

import Orange.tests


class TestOWDistanceFile(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDistanceFile)

    def open_file(self, filename):
        filename = os.path.join(os.path.split(Orange.tests.__file__)[0],
                                filename)
        self.widget.add_path(filename)
        self.widget.open_file()

    def test_non_square(self):
        self.open_file("xlsx_files/distances_nonsquare.xlsx")
        self.assertIsNone(self.get_output(self.widget.Outputs.distances))
        self.assertTrue(self.widget.Error.non_square_matrix.is_shown())
        self.open_file("xlsx_files/distances_with_nans.xlsx")
        self.assertFalse(self.widget.Error.non_square_matrix.is_shown())

    def test_nan_to_num(self):
        self.open_file("xlsx_files/distances_with_nans.xlsx")
        dist = self.get_output(self.widget.Outputs.distances)
        np.testing.assert_equal(dist, [[1, 2, 3], [4, 5, 0], [7, 0, 9]])


class TestOWDistanceFileDropHandler(unittest.TestCase):
    def test_canDropFile(self):
        handler = OWDistanceFileDropHandler()
        self.assertTrue(handler.canDropFile("test.dst"))
        self.assertTrue(handler.canDropFile("test.xlsx"))
        self.assertFalse(handler.canDropFile("test.bin"))

    def test_parametersFromFile(self):
        handler = OWDistanceFileDropHandler()
        r = handler.parametersFromFile("test.dst")
        self.assertEqual(r["recent_paths"][0].basename, "test.dst")
