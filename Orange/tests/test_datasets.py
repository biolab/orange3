import unittest

import os

import Orange


class TestDatasets(unittest.TestCase):
    def test_access(self):
        d1 = Orange.datasets.anneal
        fname = Orange.datasets.anneal['location']
        d2 = Orange.datasets['anneal']
        self.assertNotEqual(len(d1), 0)
        self.assertEqual(len(d1), len(d2))

    def test_filter(self):
        discr = [info for info in Orange.datasets.values()
                 if info['features']['continuous'] == 0]
        for info in discr:
            data = Orange.data.Table(info['location'])
            self.assertFalse(data.domain.has_continuous_attributes())

    def test_have_all(self):
        datasets_folder = os.path.join(os.path.dirname(__file__),
                                       '../datasets')
        for fname in os.listdir(datasets_folder):
            if not os.path.isfile(os.path.join(datasets_folder, fname)):
                continue
            name, ext = os.path.splitext(fname)
            if ext != '.tab':
                continue
            self.assertIn(name, Orange.datasets)
