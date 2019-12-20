# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import os

from Orange.data import Table, Variable
from Orange import datasets


class TestDatasets(unittest.TestCase):
    def test_access(self):
        d1 = datasets.iris
        fname = datasets.iris['location']
        d2 = datasets['iris']
        self.assertNotEqual(len(d1), 0)
        self.assertEqual(len(d1), len(d2))

    def test_filter(self):
        for info in datasets.values():
            if info['features']['continuous'] == 0:
                data = Table(info['location'])
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
            self.assertIn(name, datasets)

    def test_datasets_info_features(self):
        for dataset, info in datasets.items():

            if info['location'].startswith('http'): continue  # Tested elsewhere

            table = Table(dataset)
            domain = table.domain

            # Test features
            self.assertEqual(table.X.shape[0], info['rows'], dataset)
            self.assertEqual(table.has_missing(), info['missing'], dataset)
            self.assertEqual(len(domain.metas), info['features']['meta'], dataset)
            self.assertEqual(sum(i.is_discrete for i in domain.attributes),
                             info['features']['discrete'],
                             dataset)
            self.assertEqual(sum(i.is_continuous for i in domain.attributes),
                             info['features']['continuous'],
                             dataset)

            # Test class vars
            if len(domain.class_vars) > 1:
                self.assertEqual(['discrete' if i.is_discrete else 'continuous'
                                  for i in domain.class_vars],
                                 info['target']['type'],
                                 dataset)
            elif len(domain.class_vars) == 1:
                cls = domain.class_var
                self.assertEqual('discrete' if cls.is_discrete else 'continuous',
                                 info['target']['type'],
                                 dataset)
                if cls.is_discrete:
                    self.assertEqual(len(cls.values), info['target']['values'], dataset)
            else:
                self.assertEqual(False, info['target']['type'], dataset)
