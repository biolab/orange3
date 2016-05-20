# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.preprocess import ProjectCUR


class TestCURProjector(unittest.TestCase):
    def test_project_cur(self):
        data = Table("ionosphere")
        for kwargs in ({}, {'rank': 3, 'max_error': 1}):
            projector = ProjectCUR(**kwargs)
            data_cur = projector(data)
            for i in range(data_cur.X.shape[1]):
                sbtr = (data.X - data_cur.X[:, i][:, None]) == 0
                self.assertTrue((sbtr.sum(0) == data.X.shape[0]).any())
            self.assertLessEqual(data_cur.X.shape[1], data.X.shape[1])
            self.assertTrue((data.metas == data_cur.metas).all())
            self.assertTrue((data.Y == data_cur.Y).any())
