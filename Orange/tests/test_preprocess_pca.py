# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.preprocess import ProjectPCA


class TestPCAProjector(unittest.TestCase):

    def test_project_pca_default(self):
        data = Table("ionosphere")
        for kwargs, value in (({}, data.X.shape[1]),
                              ({'n_components': 5}, 5)):
            projector = ProjectPCA(**kwargs)
            data_pc = projector(data)
            self.assertEqual(data_pc.X.shape[1], value)
            self.assertTrue((data.metas == data_pc.metas).all())
            self.assertTrue((data.Y == data_pc.Y).any())
