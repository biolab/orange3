# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.preprocess import Continuize, Randomize
from Orange.projection import LDA
from Orange.data import Table


class TestLDA(unittest.TestCase):
    def test_lda(self):
        iris = Table('iris')
        n_components = 2
        lda = LDA(solver="eigen", n_components=n_components)
        model = lda(iris)
        transformed = model(iris)
        self.assertEqual(transformed.X.shape, (len(iris), n_components))
        self.assertEqual(transformed.Y.shape, (len(iris),))

    def test_transform_changed_domain(self):
        """
        1. Open data, apply some preprocessor, splits the data into two parts,
        use LDA on the first part, and then transform the second part.

        2. Open data, split into two parts, apply the same preprocessor and
        LDA only on the first part, and then transform the second part.

        The transformed second part in (1) and (2) has to be the same.
        """
        data = Table("iris")
        data = Randomize()(data)
        preprocessor = Continuize()
        lda = LDA()

        # normalize all
        ndata = preprocessor(data)

        model = lda(ndata[:75])
        result_1 = model(ndata[75:])

        # normalize only the "training" part
        ndata = preprocessor(data[:75])
        model = lda(ndata)
        result_2 = model(data[75:])

        np.testing.assert_almost_equal(result_1.X, result_2.X)
