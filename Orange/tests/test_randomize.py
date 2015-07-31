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
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.Y != data_rand.Y).any())
        self.assertTrue((np.sort(data.Y, axis=0) == np.sort(
            data_rand.Y, axis=0)).all())


    def test_randomize_classes(self):
        np.random.seed(42)
        data = Table("zoo")
        randomizer = Randomize(rand_type=Randomize.RandomizeClasses)
        data_rand = randomizer(data)
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.Y != data_rand.Y).any())
        self.assertTrue((np.sort(data.Y, axis=0) == np.sort(
            data_rand.Y, axis=0)).all())

    def test_randomize_attributes(self):
        np.random.seed(42)
        data = Table("zoo")
        randomizer = Randomize(rand_type=Randomize.RandomizeAttributes)
        data_rand = randomizer(data)
        self.assertTrue((data.Y == data_rand.Y).all())
        self.assertTrue((data.metas == data_rand.metas).all())
        self.assertTrue((data.X != data_rand.X).any())
        self.assertTrue((np.sort(data.X, axis=0) == np.sort(
            data_rand.X, axis=0)).all())

    def test_randomize_metas(self):
        np.random.seed(42)
        data = Table("zoo")
        randomizer = Randomize(rand_type=Randomize.RandomizeMetas)
        data_rand = randomizer(data)
        self.assertTrue((data.X == data_rand.X).all())
        self.assertTrue((data.Y == data_rand.Y).all())
        self.assertTrue((data.metas != data_rand.metas).any())
        self.assertTrue((np.sort(data.metas, axis=0) == np.sort(
            data_rand.metas, axis=0)).all())

    def test_randomize_keep_original_data(self):
        data_orig = Table("zoo")
        data = Table("zoo")
        _ = Randomize(rand_type=Randomize.RandomizeClasses)(data)
        _ = Randomize(rand_type=Randomize.RandomizeAttributes)(data)
        _ = Randomize(rand_type=Randomize.RandomizeMetas)(data)
        self.assertTrue((data.X == data_orig.X).all())
        self.assertTrue((data.metas == data_orig.metas).all())
        self.assertTrue((data.Y == data_orig.Y).all())
