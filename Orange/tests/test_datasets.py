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

    def test_have_all(self):
        datasets_folder = '../datasets'
        for fname in os.listdir(datasets_folder):
            if not os.path.isfile(os.path.join(datasets_folder, fname)):
                continue
            name, ext = os.path.splitext(fname)
            if ext != '.tab':
                continue
            self.assertIn(name, Orange.datasets)
