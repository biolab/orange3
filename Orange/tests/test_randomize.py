import unittest

import numpy as np
from Orange.data import Table
from Orange.preprocess import Randomize


class TestRandomizer(unittest.TestCase):
    def test_randomize_default(self):
        np.random.seed(42)
        data = Table("zoo")
        randomizer = Randomize()
        data_rand = randomizer(data)
        self.assertTrue((data.Y != data_rand.Y).any())

    def test_randomize_classes(self):
        np.random.seed(42)
        data = Table("test6.tab")
        randomizer = Randomize(rand_type=Randomize.RandomizeClasses)
        data_rand = randomizer(data)
        solutionX = [[1., 5., 11, 55],
                     [2., 6., 22, 66],
                     [3., 7., 33, 77],
                     [4., 8., 44, 88]]
        solutionY = np.array([[1, 9], [2, 10], [3, 11], [4, 12]])
        solutionM = np.array([['aa'], ['bb'], ['cc'], ['dd']])
        self.assertTrue((solutionX == data_rand.X).all())
        self.assertTrue((solutionM == data_rand.metas).all())
        self.assertTrue((solutionY != data_rand.Y).all())

    def test_randomize_attributes(self):
        np.random.seed(42)
        data = Table("test6.tab")
        randomizer = Randomize(rand_type=Randomize.RandomizeAttributes)
        data_rand = randomizer(data)
        solutionX = [[1., 5., 11, 55],
                     [2., 6., 22, 66],
                     [3., 7., 33, 77],
                     [4., 8., 44, 88]]
        solutionY = np.array([[1, 9], [2, 10], [3, 11], [4, 12]])
        solutionM = [['aa'], ['bb'], ['cc'], ['dd']]
        self.assertTrue((solutionX != data_rand.X).all())
        self.assertTrue((solutionM == data_rand.metas).all())
        self.assertTrue((solutionY == data_rand.Y).all())

    def test_randomize_metas(self):
        np.random.seed(42)
        data = Table("test6.tab")
        randomizer = Randomize(rand_type=Randomize.RandomizeMetas)
        data_rand = randomizer(data)
        solutionX = [[1., 5., 11, 55],
                     [2., 6., 22, 66],
                     [3., 7., 33, 77],
                     [4., 8., 44, 88]]
        solutionY = np.array([[1, 9], [2, 10], [3, 11], [4, 12]])
        solutionM = [['aa'], ['bb'], ['cc'], ['dd']]
        self.assertTrue((solutionX == data_rand.X).all())
        self.assertTrue((solutionM != data_rand.metas).all())
        self.assertTrue((solutionY == data_rand.Y).all())
