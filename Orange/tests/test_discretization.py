from unittest import TestCase

import numpy as np

import Orange


class TestEqualFreq(TestCase):
    def test_equifreq(self):
        X = np.random.randint(0, 2, (100, 1))
        data = Orange.data.Table(X)
        disc = Orange.feature.discretization.EqualFreq(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
