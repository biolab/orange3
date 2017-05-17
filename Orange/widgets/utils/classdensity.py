import ctypes
import numpy as np

from AnyQt.QtCore import QRectF
from pyqtgraph.graphicsItems.ImageItem import ImageItem


# load the C++ library; The _grid_density is build and distributed as a
# python extension but does not export any python objects (apart from PyInit),
from . import _grid_density
lib = ctypes.pydll.LoadLibrary(_grid_density.__file__)


# compute the color/class density image
def class_density_image(min_x, max_x, min_y, max_y, resolution, x_data, y_data, rgb_data):
    x_sz = (max_x-min_x)/(resolution-1)
    y_sz = (max_y-min_y)/(resolution-1)
    x_grid = [min_x+i*x_sz for i in range(resolution)]
    y_grid = [min_y+i*y_sz for i in range(resolution)]
    n_points = len(x_data)
    sample = range(n_points)
    if n_points > 1000:
        sample = grid_sample(x_data, y_data, 1000)
    x_data_norm = (np.array(x_data)-min_x)/(max_x-min_x)
    y_data_norm = (np.array(y_data)-min_y)/(max_y-min_y)
    x_grid_norm = (np.array(x_grid)-min_x)/(max_x-min_x)
    y_grid_norm = (np.array(y_grid)-min_y)/(max_y-min_y)
    img = compute_density(x_grid_norm, y_grid_norm,
                          x_data_norm[sample], y_data_norm[sample], np.array(rgb_data)[sample])
    density_img = ImageItem(img.astype(np.uint8), autoLevels=False)
    density_img.setRect(QRectF(min_x-x_sz/2, min_y-y_sz/2,
                                    max_x-min_x+x_sz, max_y-min_y+y_sz))
    density_img.setZValue(-1)
    return density_img

# call C++ implementation
def compute_density(x_grid, y_grid, x_data, y_data, rgb_data):
    fun = lib.compute_density
    fun.restype = None
    fun.argtypes = [ctypes.c_int,
                    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ctypes.c_int,
                    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    gx = np.ascontiguousarray(x_grid, dtype=np.float64)
    gy = np.ascontiguousarray(y_grid, dtype=np.float64)
    dx = np.ascontiguousarray(x_data, dtype=np.float64)
    dy = np.ascontiguousarray(y_data, dtype=np.float64)
    drgb = np.ascontiguousarray(rgb_data, dtype=np.int32)
    resolution = len(x_grid)
    n_points = len(x_data)
    img = np.ascontiguousarray(np.zeros((resolution, resolution, 4)), dtype=np.int32)
    fun(resolution, gx, gy, n_points, dx, dy, drgb, img)
    img = np.swapaxes(img, 0, 1)
    return img

# sample k data points from a uniformly spaced g*g grid of buckets
def grid_sample(x_data, y_data, k=1000, g=10):
    n = len(x_data)
    min_x, max_x = min(x_data), max(x_data)
    min_y, max_y = min(y_data), max(y_data)
    dx, dy = (max_x-min_x)/g, (max_y-min_y)/g
    grid = [[[] for j in range(g)] for i in range(g)]
    for i in range(n):
        y = int(min((y_data[i]-min_y)/dy, g-1))
        x = int(min((x_data[i]-min_x)/dx, g-1))
        grid[y][x].append(i)
    for y in range(g):
        for x in range(g):
            np.random.shuffle(grid[y][x])
    sample = []
    while len(sample) < k:
        for y in range(g):
            for x in range(g):
                if len(grid[y][x]) != 0:
                    sample.append(grid[y][x].pop())
    np.random.shuffle(sample)
    return sample[:k]