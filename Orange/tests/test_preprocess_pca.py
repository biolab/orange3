import unittest

from Orange.data import Table
from Orange.preprocess import ProjectPCA


class TestPCAProjector(unittest.TestCase):
    def test_project_pca_default(self):
        data = Table("ionosphere")
        projector = ProjectPCA()
        data_pc = projector(data)
        self.assertTrue(data_pc.X.shape[1] == data.X.shape[1])
        self.assertTrue((data.metas == data_pc.metas).all())
        self.assertTrue((data.Y == data_pc.Y).any())

    def test_project_pca(self):
        data = Table("ionosphere")
        projector = ProjectPCA(n_components=5)
        data_pc = projector(data)
        self.assertTrue(data_pc.X.shape[1] == 5)
        self.assertTrue((data.metas == data_pc.metas).all())
        self.assertTrue((data.Y == data_pc.Y).any())
