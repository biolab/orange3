from unittest import TestCase

from math import *
import numpy as np
from numpy.random import random, randint, uniform

from PyQt4.QtGui import QColor

from Orange.widgets.visualize.owscatterplotgraph import compute_density, grid_sample


class TestScatterplotDensity(TestCase):

    def random_data(self, n_grid, n_colors, n_data):
        mx, Mx = 200, 2000
        my, My = 300, 3000
        mr, Mr = 10, 500

        x_grid = sorted(uniform(mx, Mx, n_grid))
        y_grid = sorted(uniform(my, My, n_grid))

        colors = [QColor(randint(256), randint(256), randint(256), randint(256)) for i in range(n_colors)]
        cx = uniform(mx, Mx, n_colors)
        cy = uniform(my, My, n_colors)
        cr = uniform(mr, Mr, n_colors)

        x_data, y_data, rgb_data = [], [], []
        for i in range(n_data):
            c = randint(n_colors)
            r = uniform(1, cr[c])
            a = random()*2*pi
            x_data.append(cx[c]+r*cos(a))
            y_data.append(cy[c]+r*sin(a))
            rgb_data.append(colors[c].getRgb()[:3])

        return x_grid, y_grid, x_data, y_data, rgb_data

    def test_random(self):
        x_grid, y_grid, x_data, y_data, rgb_data = self.random_data(n_grid=50, n_colors=5, n_data=121)
        img = compute_density(x_grid, y_grid, x_data, y_data, rgb_data)
        self.assertTrue(img.shape == (50, 50, 4))
        self.assertTrue(np.all(0 <= img) and np.all(img < 256))

    def test_single_class(self):
        x_grid, y_grid, x_data, y_data, rgb_data = self.random_data(n_grid=50, n_colors=1, n_data=100)
        img = compute_density(x_grid, y_grid, x_data, y_data, rgb_data)
        self.assertTrue(np.all(img[:, :, 3] == 128))

    def test_sampling(self):
        x_data = [4, 1] + list(uniform(10, 20, 1000))
        y_data = [95, 3] + list(uniform(15, 20, 1000))
        sample = grid_sample(x_data, y_data, k=30, g=10)
        self.assertIn(0, sample)
        self.assertIn(1, sample)
