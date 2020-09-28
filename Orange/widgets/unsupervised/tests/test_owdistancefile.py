import unittest

from Orange.widgets.unsupervised.owdistancefile import OWDistanceFileDropHandler


class TestOWDistanceFileDropHandler(unittest.TestCase):
    def test_canDropFile(self):
        handler = OWDistanceFileDropHandler()
        self.assertTrue(handler.canDropFile("test.dst"))
        self.assertFalse(handler.canDropFile("test.bin"))

    def test_parametersFromFile(self):
        handler = OWDistanceFileDropHandler()
        r = handler.parametersFromFile("test.dst")
        self.assertEqual(r["recent_paths"][0].basename, "test.dst")
