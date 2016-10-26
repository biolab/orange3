import sys
import time
import itertools
import heapq
import operator

from functools import reduce
from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QRectF, QPointF

import Orange.data
from Orange.data.sql.table import SqlTable
from Orange.statistics import contingency
from Orange.preprocess.discretize import EqualWidth, Discretizer

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette
from Orange.widgets.io import FileFormat
from Orange.canvas import report


def is_not_none(obj):
    return obj is not None


Tree = namedtuple(
    "Tree",
    ["xbins",          # bin edges on the first axis
     "ybins",          # bin edges on the second axis
     "contingencies",  # x/y contingency table/s
     "children",       # an (nbins, nbins) array of sub trees or None (if leaf)
     ]
)


class Tree(Tree):
    @property
    def is_leaf(self):
        """Is this node a leaf."""
        return self.children is None

    @property
    def is_empty(self):
        """Is this node empty, i.e. is it's contingency matrix empty."""
        return not np.any(self.contingencies)

    @property
    def brect(self):
        """The bounding rect `(x, y, width, height)` tuple of the node's bins.
        """
        return (self.xbins[0], self.ybins[0],
                self.xbins[-1] - self.xbins[0],
                self.ybins[-1] - self.ybins[0])

    @property
    def nbins(self):
        """Number of bins."""
        return self.xbins.size - 1

    def depth(self):
        """Return the tree depth."""
        return (1 if self.is_leaf
                  else max(ch.depth() + 1
                           for ch in filter(is_not_none, self.children.flat)))


def max_contingency(node):
    """Return the maximum contingency value from node."""
    if node.is_leaf:
        return node.contingencies.max()
    else:
        valid = np.nonzero(node.children)
        children = node.children[valid]
        mask = np.ones_like(node.children, dtype=bool)
        mask[valid] = False
        ctng = node.contingencies[mask]
        v = 0.0
        if len(children):
            v = max(max_contingency(ch) for ch in children)
        if len(ctng):
            v = max(ctng.max(), v)
        return v


def blockshaped(arr, rows, cols):
    """
    Return an array of (rows, cols) `arr` sub blocks.

    E.g. given an (N, M) array return a (N//rows, M//cols, rows, cols)
    array A, such that A[0, 0] contains the upper left sub-block of `arr`
    (`arr[0:rows, 0:cols]`), A[0, 1] the sub-block left to it
    (arr[0: rows, rows: 2 * rows]), ...

    Example
    -------

    >>> A = numpy.array(
    ...     [[1, 2, 3,  4,  5,  6],
    ...      [7, 8, 9, 10, 11, 12]]
    ... )

    >>> blockshaped(A, 2, 3)
    array([[[[ 1,  2,  3],
             [ 7,  8,  9]],
            [[ 4,  5,  6],
             [10, 11, 12]]]])

    >>> blockshaped(A, 1, 2)
    array([[[[ 1,  2]],
            [[ 3,  4]],
            [[ 5,  6]]],
           [[[ 7,  8]],
            [[ 9, 10]],
            [[11, 12]]]])

    """
    N, M = arr.shape[:2]
    rest = arr.shape[2:]
    assert N % rows == 0
    assert M % cols == 0
    return (arr.reshape((N // rows, rows, -1, cols) + rest)
               .swapaxes(1, 2)
               .reshape((N // rows, M // cols, rows, cols) + rest))


def lod_from_transform(T):
    """
    Return level of detail from a translation/scale only transform T.

    Level of detail is the geometric mean of a transformed unit
    rectangle's width and height (i.e. sqrt(area)).

    :type T: QTransform

    """
    r = T.mapRect(QRectF(0, 0, 1, 1))
    return np.sqrt(r.width() * r.height())

#: Density patch shapes
Rect, RoundRect, Circle = 0, 1, 2


class DensityPatch(pg.GraphicsObject):
    """
    A 2-dimentional (rectangular) bin-plot graphics item.

    Displays a contingency from a `Tree` instance, automatically
    re-sampling to adjust for level of detail.
    """
    #: Density patch shapes
    Rect, RoundRect, Circle = Rect, RoundRect, Circle
    #: Density patch color scale (linear, square root and logarithmic).
    Linear, Sqrt, Log = 1, 2, 3

    def __init__(self, root=None, cell_size=10, cell_shape=Rect,
                 color_scale=Sqrt, palette=None):
        super().__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption, True)
        self._root = root
        self._cache = {}
        self._cell_size = cell_size
        self._cell_shape = cell_shape
        self._color_scale = color_scale
        self._palette = palette

    def boundingRect(self):
        return self.rect()

    def rect(self):
        if self._root is not None:
            return QRectF(*self._root.brect)
        else:
            return QRectF()

    def set_root(self, root):
        """
        Set root `Tree` node.
        """
        self.prepareGeometryChange()
        self._root = root
        self._cache.clear()
        self.update()

    def set_cell_shape(self, shape):
        """
        Set the cell shape (Rect, RoundRect or Circle).
        """
        if self._cell_shape != shape:
            self._cell_shape = shape
            self.update()

    def cell_shape(self):
        return self._cell_shape

    def set_cell_size(self, size):
        """
        Set the (approximate preferred) cell size in pixels.
        """
        assert size >= 1
        if self._cell_size != size:
            self._cell_size = size
            self.update()

    def cell_size(self):
        return self._cell_size

    def set_color_scale(self, scale):
        if self._color_scale != scale:
            self._color_scale = scale
            self.update()

    def color_scale(self):
        return self._color_scale

    def paint(self, painter, option, widget):
        root = self._root
        if root is None:
            return

        cell_shape, cell_size = self._cell_shape, self._cell_size
        nbins = root.nbins
        T = painter.worldTransform()
        lod = lod_from_transform(T)
        rect = self.rect()
        # sqrt(area) of one cell in object coordinates.
        size1 = np.sqrt(rect.width() * rect.height()) / nbins
        scale = cell_size / (lod * size1)

        if np.isinf(scale):
            scale = np.finfo(float).max

        p = int(np.floor(np.log2(scale)))

        p = min(max(p, - int(np.log2(nbins ** (root.depth() - 1)))),
                int(np.log2(root.nbins)))

        if (p, cell_shape, cell_size) not in self._cache:
            rs_root = resample(root, 2 ** p)
            rs_max = max_contingency(rs_root)

            def log_scale(ctng):
                log_max = np.log(rs_max + 1)
                log_ctng = np.log(ctng + 1)
                return log_ctng / log_max

            def sqrt_scale(ctng):
                sqrt_max = np.sqrt(rs_max)
                sqrt_ctng = np.sqrt(ctng)
                return sqrt_ctng / (sqrt_max or 1)

            def lin_scale(ctng):
                return ctng / (rs_max or 1)

            scale = {self.Linear: lin_scale, self.Sqrt: sqrt_scale,
                     self.Log: log_scale}
            patch = Patch_create(rs_root, palette=self._palette,
                                 scale=scale[self._color_scale],
                                 shape=cell_shape)
            self._cache[p, cell_shape, cell_size] = patch
        else:
            patch = self._cache[p, cell_shape, cell_size]

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        for picture in picture_intersect(patch, option.exposedRect):
            picture.play(painter)

#: A visual patch of a Tree 'rendered' as a QPicture
Patch = namedtuple(
  "Patch",
  ["node",      # : Tree # source node (Tree)
   "picture",   # : () -> QPicture # combined full QPicture
   "children",  # : () -> Tuple[Patch] # all child subpatches
   ]
)


class Patch(Patch):
    @property
    def is_leaf(self):
        """Is this a leaf patch (has no children)."""
        return len(self.children()) == 0

    @property
    def rect(self):
        """Patch bounding rectangle as a QRectF."""
        return QRectF(*self.node.brect)

Some = namedtuple("Some", ["val"])


def once(f):
    cached = None

    def f_once():
        nonlocal cached
        if cached is None:
            cached = Some(f())
        return cached.val
    return f_once


def picture_intersect(patch, region):
    """
    Return a list of all QPictures in `patch` that intersect region.
    :type patch: Patch
    :type region: QRectF
    :rval: List[Patch]

    """
    if not region.intersects(patch.rect):
        return []
    elif region.contains(patch.rect) or patch.is_leaf:
        return [patch.picture()]
    else:
        accum = reduce(
            operator.iadd,
            (picture_intersect(child, region) for child in patch.children()),
            []
        )
        return accum


def Patch_create(node, palette=None, scale=None, shape=Rect):
    """
    Return a `Patch` for visualizing `node`.

    .. note::
        The patch (picture and children fields) are evaluated lazily.

    :type node: Tree
    :type palette: colorpalette.PaletteGenerator
    :type scale: nparray -> ndarray
    :type shape: int
    :rtype: Patch

    """
    if node.is_empty:
        return Patch(node, once(lambda: QtGui.QPicture()), once(lambda: ()))
    else:
        @once
        def picture_this_level():
            # Create a QPicture drawing the contribution from this
            # level only. This is all regions where the contingency is
            # not empty and does not have a computed sub-contingency
            # (i.e. the node does not have a child in that cell).

            pic = QtGui.QPicture()
            painter = QtGui.QPainter(pic)
            ctng = node.contingencies
            colors = create_image(ctng, palette, scale=scale)
            x, y, w, h = node.brect
            N, M = ctng.shape[:2]

            # Nonzero contingency mask
            any_mask = Node_mask(node)

            if node.is_leaf:
                skip = itertools.repeat(False)
            else:
                # Skip all None children they were already painted.
                skip = (ch is not None for ch in node.children.flat)

            painter.save()
            painter.translate(x, y)
            painter.scale(w / node.nbins, h / node.nbins)

            indices = itertools.product(range(N), range(M))
            for (i, j), skip, any_ in zip(indices, skip, any_mask.flat):
                if not skip and any_:
                    painter.setBrush(QtGui.QColor(*colors[i, j]))
                    if shape == Rect:
                        painter.drawRect(i, j, 1, 1)
                    elif shape == Circle:
                        painter.drawEllipse(i, j, 1, 1)
                    elif shape == RoundRect:
                        painter.drawRoundedRect(i, j, 1, 1, 25.0, 25.0,
                                                Qt.RelativeSize)
            painter.restore()
            painter.end()
            return pic

        @once
        def child_patches():
            # Return a tuple of all non empty child patches for this node.
            if node.is_leaf:
                children = []
            else:
                children = filter(is_not_none, node.children.flat)
            return tuple(Patch_create(child, palette, scale, shape)
                         for child in children) + \
                   (Patch(node, picture_this_level, once(lambda: ())),)

        @once
        def picture_children():
            # Return a QPicture displaying all children patches of this
            # node.
            pic = QtGui.QPicture()
            painter = QtGui.QPainter(pic)
            for ch in child_patches():
                painter.drawPicture(0, 0, ch.picture())
            painter.end()
            return pic

        return Patch(node, picture_children, child_patches)


def resample(node, samplewidth):
    """
    Resample/aggregate the node's contingency, joining `samplewidth` bins.

    `samplewidth` is the number of bins which should be joined and MUST
    be a power of 2.

    If `samplewidth` == 1 then return the node as is. If larger then
    1 then sum `samplewidth` neighboring contingency cells returning a
    new node with shape ``(nbins//samplewidth, nbins//samplewidth)``.
    If smaller then 1 (i.e. undersampled) recurse into node's
    subcontingencies with a samplewidth * nbins

    """
    assert 0 < samplewidth <= node.nbins
    assert int(np.log2(samplewidth)) == np.log2(samplewidth)

    if samplewidth == 1:
        return node._replace(children=None)
    elif samplewidth > 1:
        samplewidth = int(samplewidth)
        ctng = blockshaped(node.contingencies, samplewidth, samplewidth)
        ctng = ctng.sum(axis=(2, 3))
        assert ctng.shape[0] == node.nbins // samplewidth
        return Tree(node.xbins[::samplewidth],
                    node.ybins[::samplewidth],
                    ctng,
                    None)
    elif node.is_leaf:
        return Tree(*node)
    else:
        nbins = node.nbins
        children = [resample(ch, samplewidth * nbins)
                    if ch is not None else None
                    for ch in node.children.flat]

        children_ar = np.full(nbins ** 2, None, dtype=object)
        children_ar[:] = children
        return node._replace(children=children_ar.reshape((-1, nbins)))


class OWScatterMap(widget.OWWidget):
    name = "Scatter Map"
    description = "Draw a two dimensional rectangular bin density plot."
    icon = "icons/Scattermap.svg"
    priority = 500

    inputs = [("Data", Orange.data.Table, "set_data")]

    settingsHandler = settings.DomainContextHandler()

    x_var_index = settings.ContextSetting(0)
    y_var_index = settings.ContextSetting(1)
    z_var_index = settings.ContextSetting(0)
    selected_z_values = settings.ContextSetting([])

    color_scale = settings.Setting(1)
    sample_level = settings.Setting(0)

    sample_percentages = []
    sample_percentages_captions = []
    sample_times = [0.5, 3, 5, 20, 40, 80]
    sample_times_captions = ['1 s', '5 s', '10 s', '30 s', '1 min', '2 min']

    use_cache = settings.Setting(True)

    n_bins = 2 ** 4

    mouse_mode = 0

    graph_name = "plot.plotItem"

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.z_values = []

        self._root = None
        self._displayed_root = None
        self._item = None
        self._cache = {}

        self.colors = colorpalette.ColorPaletteGenerator(10)

        box = gui.vBox(self.controlArea, "Axes")
        self.x_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesX = gui.comboBox(
            box, self, value='x_var_index', callback=self.replot,
            contentsLength=12)
        self.comboBoxAttributesX.setModel(self.x_var_model)

        self.y_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesY = gui.comboBox(
            box, self, value='y_var_index', callback=self.replot,
            contentsLength=12)
        self.comboBoxAttributesY.setModel(self.y_var_model)

        box = gui.vBox(self.controlArea, "Color")
        self.z_var_model = itemmodels.VariableListModel()
        self.comboBoxClassvars = gui.comboBox(
            box, self, value='z_var_index',
            callback=self._on_z_var_changed,
            contentsLength=12)
        self.comboBoxClassvars.setModel(self.z_var_model)

        self.z_values_view = gui.listBox(
            box, self, "selected_z_values", "z_values",
            callback=self._on_z_values_selection_changed,
            selectionMode=QtGui.QListView.MultiSelection,
            addSpace=False
        )
        gui.comboBox(box, self, "color_scale", label="Scale: ",
                     orientation=Qt.Horizontal,
                     items=["Linear", "Square root", "Logarithmic"],
                     callback=self._on_color_scale_changed)

        self.sampling_box = gui.vBox(self.controlArea, "Sampling")
        sampling_options = (self.sample_times_captions +
                            self.sample_percentages_captions)
        self.sample_combo = gui.comboBox(
            self.sampling_box, self, 'sample_level', items=sampling_options,
            callback=self.update_sample)
        gui.button(self.sampling_box, self, "Sharpen", self.sharpen)

        gui.rubber(self.controlArea)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.plot.setMinimumSize(500, 500)

        def font_resize(font, factor, minsize=None, maxsize=None):
            font = QtGui.QFont(font)
            fontinfo = QtGui.QFontInfo(font)
            size = fontinfo.pointSizeF() * factor

            if minsize is not None:
                size = max(size, minsize)
            if maxsize is not None:
                size = min(size, maxsize)

            font.setPointSizeF(size)
            return font

        axisfont = font_resize(self.font(), 0.8, minsize=11)
        axispen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        axis = self.plot.getAxis("bottom")
        axis.setTickFont(axisfont)
        axis.setPen(axispen)
        axis = self.plot.getAxis("left")
        axis.setTickFont(axisfont)
        axis.setPen(axispen)

        self.plot.getViewBox().sigTransformChanged.connect(
            self._on_transform_changed)
        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, dataset):
        self.closeContext()
        self.clear()

        if isinstance(dataset, SqlTable):
            self.original_data = dataset
            self.sample_level = 0
            self.sample_combo.setEnabled(True)
            self.update_sample()
        else:
            self.dataset = dataset
            self.sample_combo.setCurrentIndex(-1)
            self.sample_combo.setEnabled(False)
            self.set_sampled_data(self.dataset)

    def update_sample(self):
        self.closeContext()
        self.clear()

        if self.sample_level < len(self.sample_times):
            sample_type = 'time'
            level = self.sample_times[self.sample_level]
        else:
            sample_type = 'percentage'
            level = self.sample_level - len(self.sample_times)
            level = self.sample_percentages[level]

        if sample_type == 'time':
            self.dataset = self.original_data.sample_time(level, no_cache=True)
        else:
            if 0 < level < 100:
                self.dataset = self.original_data.sample_percentage(
                    level, no_cache=True)
            if level >= 100:
                self.dataset = self.original_data
        self.set_sampled_data(self.dataset)

    def set_sampled_data(self, dataset):
        if dataset is None:
            return

        domain = dataset.domain
        cvars = [var for var in domain.variables if var.is_continuous]
        dvars = [var for var in domain.variables if var.is_discrete]

        self.x_var_model[:] = cvars
        self.y_var_model[:] = cvars
        self.z_var_model[:] = dvars

        nvars = len(cvars)
        self.x_var_index = min(max(0, self.x_var_index), nvars - 1)
        self.y_var_index = min(max(0, self.y_var_index), nvars - 1)
        self.z_var_index = min(max(0, self.z_var_index), len(dvars) - 1)

        if domain.has_discrete_class:
            self.z_var_index = dvars.index(domain.class_var)
        else:
            self.z_var_index = len(dvars) - 1

        self.openContext(dataset)

        if 0 <= self.z_var_index < len(self.z_var_model):
            self.z_values = self.z_var_model[self.z_var_index].values
            k = len(self.z_values)
            self.selected_z_values = range(k)
            self.colors = colorpalette.ColorPaletteGenerator(k)
            for i in range(k):
                item = self.z_values_view.item(i)
                item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        self.error("Data contains no continuous features", shown=not cvars)
        self.setup_plot()

    def clear(self):
        self.dataset = None
        self.x_var_model[:] = []
        self.y_var_model[:] = []
        self.z_var_model[:] = []
        self.z_values = []
        self._root = None
        self._displayed_root = None
        self._item = None
        self._cache = {}
        self.plot.clear()

    def _on_z_var_changed(self):
        if 0 <= self.z_var_index < len(self.z_var_model):
            self.z_values = self.z_var_model[self.z_var_index].values
            k = len(self.z_values)
            self.selected_z_values = range(k)

            self.colors = colorpalette.ColorPaletteGenerator(k)
            for i in range(k):
                item = self.z_values_view.item(i)
                item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

            self.replot()

    def _on_z_values_selection_changed(self):
        if self._displayed_root is not None:
            self.update_map(self._displayed_root)

    def _on_color_scale_changed(self):
        if self._displayed_root is not None:
            self.update_map(self._displayed_root)

    def setup_plot(self):
        """Setup the density map plot"""
        self.plot.clear()
        self.x_var_index = min(self.x_var_index, len(self.x_var_model) - 1)
        self.y_var_index = min(self.y_var_index, len(self.y_var_model) - 1)
        if not self.dataset or self.x_var_index == -1 or self.y_var_index == -1:
            return

        data = self.dataset
        xvar = self.x_var_model[self.x_var_index]
        yvar = self.y_var_model[self.y_var_index]
        if 0 <= self.z_var_index < len(self.z_var_model):
            zvar = self.z_var_model[self.z_var_index]
        else:
            zvar = None

        axis = self.plot.getAxis("bottom")
        axis.setLabel(xvar.name)

        axis = self.plot.getAxis("left")
        axis.setLabel(yvar.name)

        if (xvar, yvar, zvar) in self._cache:
            root = self._cache[xvar, yvar, zvar]
        else:
            root = self.get_root(data, xvar, yvar, zvar)
            self._cache[xvar, yvar, zvar] = root

        self._root = root

        self.update_map(root)

    def get_root(self, data, xvar, yvar, zvar=None):
        """Compute the root density map item"""
        assert self.n_bins > 2
        x_disc = EqualWidth(n=self.n_bins)(data, xvar)
        y_disc = EqualWidth(n=self.n_bins)(data, yvar)

        def bins(var):
            points = list(var.compute_value.points)
            assert points[0] <= points[1]
            width = points[1] - points[0]
            return np.array([points[0] - width] +
                            points +
                            [points[-1] + width])

        xbins = bins(x_disc)
        ybins = bins(y_disc)

        # Extend the lower/upper bin edges to infinity.
        # (the grid_bin function has an optimization for this case).
        xbins1 = np.r_[-np.inf, xbins[1:-1], np.inf]
        ybins1 = np.r_[-np.inf, ybins[1:-1], np.inf]

        t = grid_bin(data, xvar, yvar, xbins1, ybins1, zvar=zvar)
        return t._replace(xbins=xbins, ybins=ybins)

    def replot(self):
        self.setup_plot()

    def update_map(self, root):
        self.plot.clear()
        self._item = None

        self._displayed_root = root

        palette = self.colors
        contingencies = root.contingencies

        def Tree_take(node, indices, axis):
            """Take elements from the contingency matrices in node."""
            contig = np.take(node.contingencies, indices, axis)
            if node.is_leaf:
                return node._replace(contingencies=contig)
            else:
                children_ar = np.full(node.children.size, None, dtype=object)
                children_ar[:] = [
                    Tree_take(ch, indices, axis) if ch is not None else None
                    for ch in node.children.flat
                ]
                children = children_ar.reshape(node.children.shape)
                return node._replace(contingencies=contig, children=children)

        if contingencies.ndim == 3:
            if not self.selected_z_values:
                return

            _, _, k = contingencies.shape

            if self.selected_z_values != list(range(k)):
                palette = [palette[i] for i in self.selected_z_values]
                root = Tree_take(root, self.selected_z_values, 2)

        self._item = item = DensityPatch(
            root, cell_size=10,
            cell_shape=DensityPatch.Rect,
            color_scale=self.color_scale + 1,
            palette=palette
        )
        self.plot.addItem(item)

    def sharpen(self):
        viewb = self.plot.getViewBox()
        rect = viewb.boundingRect()
        p1 = viewb.mapToView(rect.topLeft())
        p2 = viewb.mapToView(rect.bottomRight())
        rect = QtCore.QRectF(p1, p2).normalized()

        self.sharpen_region(rect)

    def sharpen_root_region(self, region):
        data = self.dataset
        xvar = self.x_var_model[self.x_var_index]
        yvar = self.y_var_model[self.y_var_index]

        if 0 <= self.z_var_index < len(self.z_var_model):
            zvar = self.z_var_model[self.z_var_index]
        else:
            zvar = None

        root = self._root

        if not QRectF(*root.brect).intersects(region):
            return

        nbins = self.n_bins

        def bin_func(xbins, ybins):
            return grid_bin(data, xvar, yvar, xbins, ybins, zvar)

        last_node = root
        update_time = time.time()
        changed = False

        with self.progressBar(nbins ** 2) as progress_bar:
            for i, node in enumerate(
                    sharpen_region(self._root, region, nbins, bin_func)):
                tick = time.time() - update_time
                changed = changed or node is not last_node
                if changed and ((i % nbins == 0) or tick > 2.0):
                    self.update_map(node)
                    last_node = node
                    changed = False
                    update_time = time.time()
                    progress_bar.advance()

        self._root = last_node
        self._cache[xvar, yvar, zvar] = self._root
        self.update_map(self._root)

    def _sampling_width(self):
        if self._item is None:
            return 0

        item = self._item
        rect = item.rect()

        T = self.plot.transform() * item.sceneTransform()
#         lod = QtGui.QStyleOptionGraphicsItem.levelOfDetailFromTransform(T)
        lod = lod_from_transform(T)
        size1 = np.sqrt(rect.width() * rect.height()) / self.n_bins
        cell_size = 10
        scale = cell_size / (lod * size1)
        if np.isinf(scale):
            scale = np.finfo(float).max
        p = int(np.floor(np.log2(scale)))
        p = min(p, int(np.log2(self.n_bins)))
        return 2 ** int(p)

    def sharpen_region(self, region):
        data = self.dataset
        root = self._root
        nbins = self.n_bins
        xvar = yvar = zvar = None

        if 0 <= self.x_var_index < len(self.x_var_model):
            xvar = self.x_var_model[self.x_var_index]
        if 0 <= self.y_var_index < len(self.y_var_model):
            yvar = self.y_var_model[self.y_var_index]
        if 0 <= self.z_var_index < len(self.z_var_model):
            zvar = self.z_var_model[self.z_var_index]

        if data is None or xvar is None or yvar is None or root is None:
            return

        if not QRectF(*root.brect).intersects(region):
            return

        def bin_func(xbins, ybins):
            return grid_bin(data, xvar, yvar, xbins, ybins, zvar)

        def min_depth(node, region):
            if not region.intersects(QRectF(*node.brect)):
                return np.inf
            elif node.is_leaf:
                return 1
            elif node.is_empty:
                return 1
            else:
                xs, xe, ys, ye = bindices(node, region)
                children = node.children[xs: xe, ys: ye].ravel()
                contingency = node.contingencies[xs: xe, ys: ye]
                if contingency.ndim == 3:
                    contingency = contingency.reshape(-1, contingency.shape[2])

                if any(ch is None and np.any(val)
                       for ch, val in zip(children, contingency)):
                    return 1
                else:
                    ch_depth = [min_depth(ch, region) + 1
                                for ch in filter(is_not_none, children.flat)]
                    return min(ch_depth if ch_depth else [1])

        depth = min_depth(self._root, region)
        bw = self._sampling_width()
        nodes = self.select_nodes_to_sharpen(self._root, region, bw,
                                             depth + 1)

        def update_rects(node):
            scored = score_candidate_rects(node, region)
            ind1 = set(zip(*Node_nonzero(node)))
            ind2 = set(zip(*node.children.nonzero())) \
                   if not node.is_leaf else set()
            ind = ind1 - ind2
            return [(score, r) for score, i, j, r in scored if (i, j) in ind]

        scored_rects = reduce(operator.iadd, map(update_rects, nodes), [])
        scored_rects = sorted(scored_rects, reverse=True,
                              key=operator.itemgetter(0))
        root = self._root
        update_time = time.time()

        with self.progressBar(len(scored_rects)) as progress_bar:
            for i, (_, rect) in enumerate(scored_rects):
                root = sharpen_region_recur(
                    root, rect.intersected(region),
                    nbins, depth + 1, bin_func)
                tick = time.time() - update_time
                if tick > 2.0:
                    self.update_map(root)
                    update_time = time.time()
                progress_bar.advance()

        self._root = root

        self._cache[xvar, yvar, zvar] = self._root
        self.update_map(self._root)

    def select_nodes_to_sharpen(self, node, region, bw, depth):
        """
        :param node:
        :param bw: bandwidth (samplewidth)
        :param depth: maximum node depth to consider
        """

        if not QRectF(*node.brect).intersects(region):
            return []
        elif bw >= 1:
            return []
        elif depth == 1:
            return []
        elif node.is_empty:
            return []
        elif node.is_leaf:
            return [node]
        else:
            xs, xe, ys, ye = bindices(node, region)

            def intersect_indices(rows, cols):
                mask = (xs <= rows) & (rows < xe) & (ys <= cols) & (cols < ye)
                return rows[mask], cols[mask]

            indices1 = intersect_indices(*Node_nonzero(node))
            indices2 = intersect_indices(*node.children.nonzero())
            # If there are any non empty and non expanded cells in the
            # intersection return the node for sharpening, ...
            if np.any(np.array(indices1) != np.array(indices2)):
                return [node]

            children = node.children[indices2]
            # ... else run down the children in the intersection
            return reduce(operator.iadd,
                          (self.select_nodes_to_sharpen(
                               ch, region, bw * node.nbins, depth - 1)
                           for ch in children.flat),
                          [])

    def _on_transform_changed(self, *args):
        pass

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    def get_widget_name_extension(self):
        if self.dataset is None:
            return
        if self.x_var_index < 0 or self.y_var_index < 0:
            return
        return "{} vs {}".format(
            self.x_var_model[self.x_var_index],
            self.y_var_model[self.y_var_index])

    def send_report(self):
        if self.dataset is None:
            return
        caption = report.list_legend(self.z_values_view,
                                     self.selected_z_values)
        self.report_plot()
        self.report_caption(caption)


def grid_bin(data, xvar, yvar, xbins, ybins, zvar=None):
    x_disc = Discretizer.create_discretized_var(xvar, xbins[1:-1])
    y_disc = Discretizer.create_discretized_var(yvar, ybins[1:-1])

    x_min, x_max = xbins[0], xbins[-1]
    y_min, y_max = ybins[0], ybins[-1]

    querydomain = [x_disc, y_disc]
    if zvar is not None:
        querydomain = querydomain + [zvar]

    querydomain = Orange.data.Domain(querydomain)

    def interval_filter(var, low, high):
        return Orange.data.filter.Values(
            [Orange.data.filter.FilterContinuous(
                 var, max=high, min=low,
                 oper=Orange.data.filter.FilterContinuous.Between)]
        )

    def value_filter(var, val):
        return Orange.data.filter.Values(
            [Orange.data.filter.FilterDiscrete(var, [val])]
        )

    def filters_join(filters):
        return Orange.data.filter.Values(
            reduce(list.__iadd__, (f.conditions for f in filters), [])
        )

    inf_bounds = np.isinf([x_min, x_max, y_min, y_max])
    if not all(inf_bounds):
        # No need to filter the data
        range_filters = [interval_filter(xvar, x_min, x_max),
                         interval_filter(yvar, y_min, y_max)]
        range_filter = filters_join(range_filters)
        subset = range_filter(data)
    else:
        subset = data

    if zvar and zvar.is_discrete:
        filters = [value_filter(zvar, val) for val in zvar.values]
        contingencies = [
            contingency.get_contingency(
                filter_(subset.from_table(querydomain, subset)),
                col_variable=y_disc, row_variable=x_disc
            )
            for filter_ in filters
        ]
        contingencies = np.dstack(contingencies)
    else:
        contingencies = contingency.get_contingency(
            subset.from_table(querydomain, subset),
            col_variable=y_disc, row_variable=x_disc
        )

    contingencies = np.asarray(contingencies)
    return Tree(xbins, ybins, contingencies, None)


def sharpen_node_cell(node, i, j, nbins, gridbin_func):
    if node.is_leaf:
        children = np.full((nbins, nbins), None, dtype=object)
    else:
        children = np.array(node.children, dtype=None)

    xbins = np.linspace(node.xbins[i], node.xbins[i + 1], nbins + 1)
    ybins = np.linspace(node.ybins[j], node.ybins[j + 1], nbins + 1)

    if node.contingencies[i, j].any():
        t = gridbin_func(xbins, ybins)
        assert t.contingencies.shape[:2] == (nbins, nbins)
        children[i, j] = t
        return node._replace(children=children)
    else:
        return node


def sharpen_node_cell_range(node, xrange, yrange, nbins, gridbin_func):
    if node.is_leaf:
        children = np.full((nbins, nbins), None, dtype=object)
    else:
        children = np.array(node.children, dtype=None)

    xs, xe = xrange.start, xrange.stop
    ys, ye = yrange.start, yrange.stop

    xbins = np.linspace(node.xbins[xs], node.xbins[xe], (xe - xs) * nbins + 1)
    ybins = np.linspace(node.ybins[ys], node.ybins[ye], (ye - ys) * nbins + 1)

    if node.contingencies[xs: xe, ys: ye].any():
        t = gridbin_func(xbins, ybins)
        for i, j in itertools.product(range(xs, xe), range(ys, ye)):
            children[i, j] = Tree(t.xbins[i * nbins: (i + 1) * nbins + 1],
                                  t.ybins[j * nbins: (j + 1) * nbins + 1],
                                  t.contingencies[xs: xe, ys: ye])
            assert children[i, j].shape[:2] == (nbins, nbins)
        return node._replace(children=children)
    else:
        return node


def sharpen_region(node, region, nbins, gridbin_func):
    if not QRectF(*node.brect).intersects(region):
        raise ValueError()
#         return node

    xs, xe, ys, ye = bindices(node, region)
    ndim = node.contingencies.ndim

    if node.children is not None:
        children = np.array(node.children, dtype=object)
        assert children.ndim == 2
    else:
        children = np.full((nbins, nbins), None, dtype=object)

    if ndim == 3:
        # compute_chisqares expects classes in 1 dim
        c = node.contingencies
        chi_lr, chi_up = compute_chi_squares(
            c[xs: xe, ys: ye, :].swapaxes(1, 2).swapaxes(0, 1)
        )

        def max_chisq(i, j):
            def valid(i, j):
                return 0 <= i < chi_up.shape[0] and \
                       0 <= j < chi_lr.shape[1]

            return max(chi_lr[i, j] if valid(i, j) else 0,
                       chi_lr[i, j - 1] if valid(i, j - 1) else 0,
                       chi_up[i, j] if valid(i, j) else 0,
                       chi_up[i - 1, j] if valid(i - 1, j) else 0)

        heap = [(-max_chisq(i - xs, j - ys), (i, j))
                for i in range(xs, xe)
                for j in range(ys, ye)
                if children[i, j] is None]
    else:
        heap = list(enumerate((i, j)
                    for i in range(xs, xe)
                    for j in range(ys, ye)
                    if children[i, j] is None))

    heap = sorted(heap)
    update_node = node
    while heap:
        _, (i, j) = heapq.heappop(heap)

        xbins = np.linspace(node.xbins[i], node.xbins[i + 1], nbins + 1)
        ybins = np.linspace(node.ybins[j], node.ybins[j + 1], nbins + 1)

        if node.contingencies[i, j].any():
            t = gridbin_func(xbins, ybins)
            assert t.contingencies.shape[:2] == (nbins, nbins)
        else:
            t = None

        children[i, j] = t
        if t is None:
            yield update_node
        else:
            update_node = update_node._replace(
                children=np.array(children, dtype=object)
            )
            yield update_node


def Node_mask(node):
    if node.contingencies.ndim == 3:
        return node.contingencies.any(axis=2)
    else:
        return node.contingencies > 0


def Node_nonzero(node):
    return np.nonzero(Node_mask(node))


def sharpen_region_recur(node, region, nbins, depth, gridbin_func):
    if depth <= 1:
        return node
    elif not QRectF(*node.brect).intersects(region):
        return node
    elif node.is_empty:
        return node
    elif node.is_leaf:
        xs, xe, ys, ye = bindices(node, region)
        # indices in need of update
        indices = Node_nonzero(node)
        for i, j in zip(*indices):
            if xs <= i < xe and ys <= j < ye:
                node = sharpen_node_cell(node, i, j, nbins, gridbin_func)

        # if the exposed region is empty the node.is_leaf property
        # is preserved
        if node.is_leaf:
            return node

        return sharpen_region_recur(node, region, nbins, depth, gridbin_func)
    else:
        xs, xe, ys, ye = bindices(node, region)

        # indices is need of update
        indices1 = Node_nonzero(node)
        indices2 = node.children.nonzero()
        indices = sorted(set(list(zip(*indices1))) - set(list(zip(*indices2))))

        for i, j in indices:
            if xs <= i < xe and ys <= j < ye:
                node = sharpen_node_cell(node, i, j, nbins, gridbin_func)

        children = np.array(node.children, dtype=object)
        children[xs: xe, ys: ye] = [
            [sharpen_region_recur(ch, region, nbins, depth - 1, gridbin_func)
             if ch is not None else None
             for ch in row]
            for row in np.array(children[xs: xe, ys: ye])
        ]
        return node._replace(children=children)


def stack_tile_blocks(blocks):
    return np.vstack(list(map(np.hstack, blocks)))


def bindices(node, rect):
    assert rect.normalized() == rect
    assert not rect.intersected(QRectF(*node.brect)).isEmpty()

    xs = np.searchsorted(node.xbins, rect.left(), side="left") - 1
    xe = np.searchsorted(node.xbins, rect.right(), side="right")
    ys = np.searchsorted(node.ybins, rect.top(), side="left") - 1
    ye = np.searchsorted(node.ybins, rect.bottom(), side="right")

    return np.clip([xs, xe, ys, ye],
                   [0, 0, 0, 0],
                   [node.xbins.size - 2, node.xbins.size - 1,
                    node.ybins.size - 2, node.ybins.size - 1])


def create_image(contingencies, palette=None, scale=None):
#     import scipy.signal
#     import scipy.ndimage

    if scale is None:
        scale = lambda c: c / (contingencies.max() or 1)

    P = scale(contingencies)

#     if scale > 0:
#         P = contingencies / scale
#     else:
#         P = contingencies

#     nbins = node.xbins.shape[0] - 1
#     smoothing = 32
#     bandwidth = nbins / smoothing

    if P.ndim == 3:
        ncol = P.shape[-1]
        if palette is None:
            palette = colorpalette.ColorPaletteGenerator(ncol)
        colors = [palette[i] for i in range(ncol)]
        colors = np.array(
            [[c.red(), c.green(), c.blue()] for c in colors]
        )
#         P = scipy.ndimage.filters.gaussian_filter(
#             P, bandwidth, mode="constant")
#         P /= P.max()

        argmax = np.argmax(P, axis=2)
        irow, icol = np.indices(argmax.shape)
        P_max = P[irow, icol, argmax]
        positive = P_max > 0
        P_max = np.where(positive, P_max * 0.95 + 0.05, 0.0)

        colors = 255 - colors[argmax.ravel()]

        # XXX: Non linear intensity scaling
        colors = colors * P_max.ravel().reshape(-1, 1)
        colors = colors.reshape(P_max.shape + (3,))
        colors = 255 - colors
    elif P.ndim == 2:
        palette = colorpalette.ColorPaletteBW()
        mix = P
        positive = mix > 0
        mix = np.where(positive, mix * 0.99 + 0.01, 0.0)

#         mix = scipy.ndimage.filters.gaussian_filter(
#             mix, bandwidth, mode="constant")
#         mix /= mix.max() if total else 1.0

        colors = np.zeros((np.prod(mix.shape), 3)) + 255
        colors = colors - mix.ravel().reshape(-1, 1) * 255
        colors = colors.reshape(mix.shape + (3,))

    return colors.astype(int)


def score_candidate_rects(node, region):
    """
    Score candidate bin rects in node.

    Return a list of (score, i, j QRectF) list)

    """
    xs, xe, ys, ye = bindices(node, region)

    if node.contingencies.ndim == 3:
        c = node.contingencies
        # compute_chisqares expects classes in 1 dim
        chi_lr, chi_up = compute_chi_squares(
            c[xs: xe, ys: ye, :].swapaxes(1, 2).swapaxes(0, 1)
        )

        def max_chisq(i, j):
            def valid(i, j):
                return 0 <= i < chi_up.shape[0] and \
                       0 <= j < chi_lr.shape[1]

            return max(chi_lr[i, j] if valid(i, j) else 0,
                       chi_lr[i, j - 1] if valid(i, j - 1) else 0,
                       chi_up[i, j] if valid(i, j) else 0,
                       chi_up[i - 1, j] if valid(i - 1, j) else 0)

        return [(max_chisq(i - xs, j - ys), i, j,
                 QRectF(QPointF(node.xbins[i], node.ybins[j]),
                        QPointF(node.xbins[i + 1], node.ybins[j + 1])))
                 for i, j in itertools.product(range(xs, xe), range(ys, ye))]
    else:
        return [(1, i, j,
                 QRectF(QPointF(node.xbins[i], node.ybins[j]),
                        QPointF(node.xbins[i + 1], node.ybins[j + 1])))
                 for i, j in itertools.product(range(xs, xe), range(ys, ye))]


def compute_chi_squares(observes):
    """Compute chi2 scores of given observations.

    Assumes that data is generated by two independent distributions,
    one for rows and one for columns and estimate distribution parameters
    from data.

    Parameters
    ----------
    observes : numpy array with dimensions (N_CLASSES * N_ROWS * N_COLUMNS)
        Multiple contingencies containing observations for multiple classes.
    """
    CLASSES, COLS, ROWS = 0, 1, 2

    n = observes.sum((ROWS, COLS), keepdims=True)
    row_sums = observes.sum(ROWS, keepdims=True)
    col_sums = observes.sum(COLS, keepdims=True)
    estimates = row_sums * col_sums / n

    chi2 = np.nan_to_num(np.nansum((observes - estimates)**2 / estimates, axis=CLASSES))

    # compute chi squares for left-right neighbours
    chi2lr = chi2[:,:-1] + chi2[:,1:]
    chi2ud = chi2[:-1,:] + chi2[1:,:]

    return chi2lr, chi2ud


def main(argv=None):
    import sip
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)

    w = OWScatterMap()
    w.show()
    w.raise_()

    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "zoo"

    data = Orange.data.Table(filename)

    w.set_data(data)
    rval = app.exec_()

    w.set_data(None)
    w.onDeleteWidget()

    sip.delete(w)
    del w
    app.processEvents()
    return rval

if __name__ == "__main__":
    sys.exit(main())
