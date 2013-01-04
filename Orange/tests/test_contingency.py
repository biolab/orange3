import unittest

import numpy as np
import scipy.sparse as sp
from Orange.statistics import contingency
from Orange import data


class Contingency_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_discrete(self):
        d = data.Table("zoo")
        cont = contingency.Contingency(d, 0)
        print(cont)