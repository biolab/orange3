import numpy as np

from Orange.data import Table
from .preprocess import Randomize

__all__ = ["Randomizer"]


class Randomizer:
    def __init__(self, rand_type=Randomize.RandomizeClasses):
        self.rand_type = rand_type

    def __call__(self, data):
        domain = data.domain
        X = data.X.copy()
        Y = data.Y.copy()
        metas = data.metas.copy()

        if self.rand_type == Randomize.RandomizeClasses:
            Y = self.randomize(Y)
        elif self.rand_type == Randomize.RandomizeAttributes:
            X = self.randomize(X)
        elif self.rand_type == Randomize.RandomizeMetas:
            metas = self.randomize(metas)
        else:
            raise TypeError('Unsuported type')

        return Table(domain, X, Y, metas)

    def randomize(self, table):
        if len(table.shape) > 1:
            for i in range(table.shape[1]):
                np.random.shuffle(table[:,i])
            return table
        else:
            np.random.shuffle(table)
            return table
