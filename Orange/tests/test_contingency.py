import unittest

import numpy as np
import scipy.sparse as sp
from Orange.statistics import contingency
from Orange import data


class Discrete_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_discrete(self):
        d = data.Table("zoo")
        cont = contingency.Discrete(d, 0)
        print(cont)

class Continuous_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_continuous(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, 0)
        print(cont)