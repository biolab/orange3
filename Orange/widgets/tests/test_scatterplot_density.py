from unittest import TestCase

from collections import Counter
from math import *
import numpy as np
from numpy.random import random, randint, shuffle, uniform

from PyQt4.QtGui import QColor
from sklearn.neighbors import NearestNeighbors

from Orange.widgets.visualize.owscatterplotgraph import compute_density as compute_density_cpp


class TestScatterplotDensity(TestCase):

    # reference Python implementation
    def compute_density_py(self, x_grid, y_grid, x_data, y_data, rgb_data):
        k = int(len(x_data)**0.5)
        distinct_colors = len(set(rgb_data))
        lo, hi = ceil(k/distinct_colors), k
        # find nearest neighbours of all grid points
        grid = [[x, y] for x in x_grid for y in y_grid]
        clf = NearestNeighbors()
        clf.fit(np.column_stack((x_data, y_data)))
        dist, ind = clf.kneighbors(grid, k)
        # combine colors of found neighbours
        colors = []
        for neigh in ind:
            cnt = Counter(rgb_data[i] for i in neigh)
            main_color, color_count = cnt.most_common(1)[0]
            a = int(128*((color_count-lo)/(hi-lo))) if lo != hi else 128
            colors += [(main_color[0], main_color[1], main_color[2], a)]
        return np.array(colors).reshape((len(x_grid), len(y_grid), 4))

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
        img_py = self.compute_density_py(x_grid, y_grid, x_data, y_data, rgb_data)
        img_cpp = compute_density_cpp(x_grid, y_grid, x_data, y_data, rgb_data)
        self.assertGreater(np.sum(img_py == img_cpp)/img_py.size, 0.9)

    def test_few_colors(self):
        for c in [1, 2]:
            x_grid, y_grid, x_data, y_data, rgb_data = self.random_data(n_grid=50, n_colors=c, n_data=121)
            img_py = self.compute_density_py(x_grid, y_grid, x_data, y_data, rgb_data)
            img_cpp = compute_density_cpp(x_grid, y_grid, x_data, y_data, rgb_data)
            self.assertTrue(np.all(img_py == img_cpp))

    def test_grid_data(self):
        x_coord = uniform(-1, 1, 13)
        y_coord = uniform(-1, 1, 13)
        xy = [(x,y) for x in x_coord for y in y_coord]
        xy = xy*3
        shuffle(xy)
        x_data, y_data = zip(*xy)
        rgb_data = [(255,0,0) if x < y else (0,0,255) for x, y in xy]

        x_grid = sorted(uniform(-2, 2, 31))
        y_grid = sorted(uniform(-2, 2, 31))

        img_py = self.compute_density_py(x_grid, y_grid, x_data, y_data, rgb_data)
        img_cpp = compute_density_cpp(x_grid, y_grid, x_data, y_data, rgb_data)
        self.assertTrue(np.all(img_py == img_cpp))
