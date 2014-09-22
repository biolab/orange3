import time
import itertools
import heapq
from functools import reduce
from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QRectF

import Orange.data
from Orange.statistics import contingency
from Orange.feature.discretization import EqualWidth, _discretized_var

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


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

Tree.is_leaf = property(
    lambda self: self.children is None
)

Tree.is_empty = property(
    lambda self: not np.any(self.contingencies)
)

Tree.brect = property(
    lambda self:
        (self.xbins[0],
         self.ybins[0],
         self.xbins[-1] - self.xbins[0],
         self.ybins[-1] - self.ybins[0])
)

Tree.nbins = property(
    lambda self: self.xbins.size - 1
)


Tree.depth = (
    lambda self:
        1 if self.is_leaf
          else max(ch.depth() + 1
                   for ch in filter(is_not_none, self.children.flat))
)


def blockshaped(arr, rows, cols):
    N, M = arr.shape[:2]
    rest = arr.shape[2:]
    assert N % rows == 0
    assert M % cols == 0
    return (arr.reshape((N // rows, rows, -1, cols) + rest)
               .swapaxes(1, 2)
               .reshape((N // rows, M // cols, rows, cols) + rest))


Rect, RoundRect, Circle = 0, 1, 2


class DensityPatch(pg.GraphicsObject):
    Rect, RoundRect, Circle = Rect, RoundRect, Circle

    def __init__(self, root=None, cell_size=10, cell_shape=Rect, palette=None):
        super().__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption, True)
        self._root = root
        self._cache = {}
        self._cell_size = cell_size
        self._cell_shape = cell_shape
        self._palette = palette

    def boundingRect(self):
        return self.rect()

    def rect(self):
        if self._root is not None:
            return QRectF(*self._root.brect)
        else:
            return QRectF()

    def set_root(self, root):
        self.prepareGeometryChange()
        self._root = root
        self._cache.clear()
        self.update()

    def set_cell_shape(self, shape):
        if self._cell_shape != shape:
            self._cell_shape = shape
            self.update()

    def cell_shape(self):
        return self._cell_shape

    def set_cell_size(self, size):
        assert size >= 1
        if self._cell_size != size:
            self._cell_size = size
            self.update()

    def cell_size(self):
        return self._cell_size

    def paint(self, painter, option, widget):
        root = self._root
        if root is None:
            return

        cell_shape, cell_size = self._cell_shape, self._cell_size
        nbins = root.nbins
        T = painter.worldTransform()
        # level of detail is the geometric mean of a transformed
        # unit rectangle's sides (== sqrt(area)).
        lod = option.levelOfDetailFromTransform(T)
        rect = self.rect()
        # sqrt(area) of one cell
        size1 = np.sqrt(rect.width() * rect.height()) / nbins
        cell_size = cell_size
        scale = cell_size / (lod * size1)
        p = int(np.floor(np.log2(scale)))

        p = min(max(p, - int(np.log2(nbins ** (root.depth() - 1)))),
                int(np.log2(root.nbins)))

        if (p, cell_shape, cell_size) not in self._cache:
            self._cache[p, cell_shape, cell_size] = patch = \
                create_picture_patch(resample(root, 2 ** p),
                                     palette=self._palette,
                                     shape=cell_shape)
        else:
            patch = self._cache[p, cell_shape, cell_size]

        def intersect_patch(patch, rect):
            if not rect.intersects(patch.rect):
                return []
            elif rect.contains(patch.rect) or patch.is_leaf:
                return [patch.picture]

            else:
                accum = reduce(list.__iadd__,
                               map(lambda patch: intersect_patch(patch, rect),
                                   patch.subpatches),
                               [])

                if len(patch.subpatches) != patch.node.nbins ** 2:
                    accum.append(patch.patch)
                return accum

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        for picture in intersect_patch(patch, option.exposedRect):
            picture.play(painter)


Patch = namedtuple(
  "Patch",
  ["node",     # source node
   "picture",  # combined full picture
   "patch",    # contribution from this node alone
   "subpatches"  # a list of child patches
   ]
)

Patch.rect = property(
    lambda self: QRectF(*self.node.brect)
)

Patch.is_leaf = property(
    lambda self: len(self.subpatches) == 0
)


def create_picture_patch(node, palette=None, scale=None, shape=Rect):
    pic = QtGui.QPicture()
    subpatches = []
    if node.is_empty:
        return Patch(node, pic, pic, [])

    painter1 = QtGui.QPainter(pic)

    ctng = node.contingencies

    if not node.is_leaf:
        # First run down the non None children
        for ch in filter(is_not_none, node.children.flat):
            sub = create_picture_patch(ch, palette, scale, shape=shape)
            sub.picture.play(painter1)
            subpatches.append(sub)

    colors = create_image(ctng, palette, scale=scale)
    x, y, w, h = node.brect
    N, M = ctng.shape[:2]

    indices = itertools.product(range(N), range(M))

    ctng = ctng.reshape((-1,) + ctng.shape[2:])
    if ctng.ndim == 2:
        ctng_any = ctng.any(axis=1)
    else:
        ctng_any = ctng > 0

    if node.is_leaf:
        skip = itertools.repeat(False)
    else:
        # Skip all None children they were already painted.
        skip = (ch is not None for ch in node.children.flat)

    thispic = QtGui.QPicture()
    painter = QtGui.QPainter(thispic)
    painter.save()
    painter.translate(x, y)
    painter.scale(w / node.nbins, h / node.nbins)

    for (i, j), skip, any_ in zip(indices, skip, ctng_any):
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

    painter1.drawPicture(0, 0, thispic)
    painter1.end()

    return Patch(node, pic, thispic, subpatches)


def resample(node, samplewidth):
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


class OWHeatMap(widget.OWWidget):
    name = "Heat map"
    description = "Draw a two dimentional density plot."
    icon = "icons/Heatmap.svg"
    priority = 100

    inputs = [("Data", Orange.data.Table, "set_data")]

    settingsHandler = settings.DomainContextHandler()

    x_var_index = settings.Setting(0)
    y_var_index = settings.Setting(1)
    z_var_index = settings.Setting(0)
    selected_z_values = settings.Setting([])

    use_cache = settings.Setting(True)

    n_bins = 32
    mouse_mode = 0

    def __init__(self, parent=None):
        super().__init__(self, parent)

        self.dataset = None
        self.z_values = []

        self._root = None
        self._displayed_root = None
        self._item = None
        self._cache = {}

        self.colors = colorpalette.ColorPaletteGenerator(10)

        box = gui.widgetBox(self.controlArea, "Input")

        self.labelDataInput = gui.widgetLabel(box, 'No data on input')
        self.labelDataInput.setTextFormat(Qt.PlainText)
        self.labelOutput = gui.widgetLabel(box, '')

        self.x_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesX = gui.comboBox(
            self.controlArea, self, value='x_var_index', box='X Attribute',
            callback=self.replot)
        self.comboBoxAttributesX.setModel(self.x_var_model)

        self.y_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesY = gui.comboBox(
            self.controlArea, self, value='y_var_index', box='Y Attribute',
            callback=self.replot)
        self.comboBoxAttributesY.setModel(self.y_var_model)

        box = gui.widgetBox(self.controlArea, "Color by")
        self.z_var_model = itemmodels.VariableListModel()
        self.comboBoxClassvars = gui.comboBox(
            box, self, value='z_var_index',
            callback=self._on_z_var_changed)
        self.comboBoxClassvars.setModel(self.z_var_model)

        box = gui.widgetBox(box, 'Colors displayed')
        box.setFlat(True)

        self.z_values_view = gui.listBox(
            box, self, "selected_z_values", "z_values",
            callback=self._on_z_values_selection_changed,
            selectionMode=QtGui.QListView.MultiSelection
        )

        self.mouseBehaviourBox = gui.radioButtons(
            self.controlArea, self, value='mouse_mode',
            btnLabels=('Drag', 'Select'),
            box='Mouse left button behavior',
            callback=self._update_mouse_mode
        )

        box = gui.widgetBox(self.controlArea, box='Display')
        gui.button(box, self, "Sharpen", self.sharpen)

        gui.rubber(self.controlArea)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.plot.setMinimumSize(500, 500)
        self.plot.getViewBox().sigTransformChanged.connect(
            self._on_transform_changed)
        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, dataset):
        self.closeContext()
        self.clear()

        self.dataset = dataset

        if dataset is not None:
            domain = dataset.domain
            cvars = list(filter(is_continuous, domain.variables))
            dvars = list(filter(is_discrete, domain.variables))

            self.x_var_model[:] = cvars
            self.y_var_model[:] = cvars
            self.z_var_model[:] = dvars

            nvars = len(cvars)
            self.x_var_index = min(max(0, self.x_var_index), nvars - 1)
            self.y_var_index = min(max(0, self.y_var_index), nvars - 1)
            self.z_var_index = min(max(0, self.z_var_index), len(cvars) - 1)

            if is_discrete(domain.class_var):
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

            self.labelDataInput.setText(
                'Data set: %s\nInstances: %d'
                % (getattr(self.dataset, "name", "untitled"), len(dataset))
            )

            self.setup_plot()
        else:
            self.labelDataInput.setText('No data on input')
            self.send("Sampled data", None)

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
            self.update_map(self._displayed_root, )

    def setup_plot(self):
        """Setup the density map plot"""
        self.plot.clear()
        if self.dataset is None or self.x_var_index == -1 or \
                self.y_var_index == -1:
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
            points = list(var.get_value_from.points)
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
        self.plot.clear()
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

    def sharpen_region(self, region):
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

        xs, xe, ys, ye = bindices(root, region)
        ndim = root.contingencies.ndim
        nbins = self.n_bins

        if root.children is not None:
            children = np.array(root.children)
        else:
            children = np.full((self.n_bins, self.n_bins), None, dtype=object)

        if ndim == 3:
            # compute_chisqares expects classes in 1 dim
            c = root.contingencies
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
        niter = len(heap)
        self.progressBarInit()

        update_time = time.time()
        changed = False
        while heap:
            _, (i, j) = heapq.heappop(heap)

            xbins = np.linspace(root.xbins[i], root.xbins[i + 1],
                                self.n_bins + 1)
            ybins = np.linspace(root.ybins[j], root.ybins[j + 1],
                                self.n_bins + 1)

            if root.contingencies[i, j].any():
                t = grid_bin(data, xvar, yvar, xbins, ybins, zvar)
                changed = True
                assert t.contingencies.shape[:2] == (self.n_bins, self.n_bins)
            else:
                t = Tree(xbins, ybins,
                         np.zeros((self.n_bins, self.n_bins) +
                                  root.contingencies.shape[2:]),
                         None)

            children[i, j] = t

            self.progressBarSet(100.0 * (niter - len(heap)) / niter)
            tick = time.time() - update_time
            if changed and (len(heap) % nbins == 0 or tick > 1):
                update_time = time.time()
                changed = False
                self.update_map(root._replace(children=children))

        self._root = root._replace(children=children)
        self._cache[xvar, yvar, zvar] = self._root

        self.update_map(self._root)

        self.progressBarFinished()

    def _on_transform_changed(self, *args):
        if self._item is None:
            return

        item = self._item
        rect = item.rect()

        T = self.plot.transform() * item.sceneTransform()
        lod = QtGui.QStyleOptionGraphicsItem.levelOfDetailFromTransform(T)
        size1 = np.sqrt(rect.width() * rect.height()) / self.n_bins
        size = 10
        scale = lod * size1 / size
        p = np.floor(np.log10(1 / scale))
        bw = 2 ** int(p)
#         if bw < 1:
#             print("undersampled", lod, p)
#         else:
#             print("oversampled", lod, p)

        viewrect = self.plot.getViewBox().viewRect()

    def _update_mouse_mode(self):
        if self.mouse_mode == 0:
            mode = pg.ViewBox.PanMode
        else:
            mode = pg.ViewBox.RectMode
        self.plot.getViewBox().setMouseMode(mode)

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()


def grid_bin(data, xvar, yvar, xbins, ybins, zvar=None):
    x_disc = _discretized_var(data, xvar, xbins[1:-1])
    y_disc = _discretized_var(data, yvar, ybins[1:-1])

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

    if is_discrete(zvar):

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


def stack_tile_blocks(blocks):
    return np.vstack(list(map(np.hstack, blocks)))


def bins_join(bins):
    return np.hstack([b[:-1] for b in bins[:-1]] + [bins[-1]])


def flatten(node, nbins=None, preserve_max=False):
    if node.is_leaf:
        return node
    else:
        N, M = node.children.shape[:2]

        xind = {i: np.flatnonzero(node.children[i, :]) for i in range(N)}
        yind = {j: np.flatnonzero(node.children[:, j]) for j in range(M)}
        xind = {i: ind[0] for i, ind in xind.items() if ind.size}
        yind = {j: ind[0] for j, ind in yind.items() if ind.size}

        xbins = [node.children[i, xind[i]].xbins if i in xind
                 else np.linspace(node.xbins[i], node.xbins[i + 1], nbins + 1)
                 for i in range(N)]

        ybins = [node.children[yind[j], j].ybins if j in yind
                  else np.linspace(node.ybins[j], node.ybins[j + 1], nbins + 1)
                  for j in range(M)]

        xbins = bins_join(xbins)
        ybins = bins_join(ybins)

#         xbins = bins_join([c.xbins for c in node.children[:, 0]])
#         ybins = bins_join([c.ybins for c in node.children[0, :]])

        ndim = node.contingencies.ndim
        if ndim == 3:
            repeats = (nbins, nbins, 1)
        else:
            repeats = (nbins, nbins)

        def child_contingency(node, row, col):
            child = node.children[row, col]
            if child is None:
                return np.tile(node.contingencies[row, col], repeats)
            elif preserve_max:
                parent_max = np.max(node.contingencies[row, col])
                c_max = np.max(child.contingencies)
                if c_max > 0:
                    return child.contingencies * (parent_max / c_max)
                else:
                    return child.contingencies
            else:
                return child.contingencies

        contingencies = [[child_contingency(node, i, j)
                          for j in range(nbins)]
                         for i in range(nbins)]

        contingencies = stack_tile_blocks(contingencies)
        cnode = Tree(xbins, ybins, contingencies, None)
        assert node.brect == cnode.brect
        assert np.all(np.diff(cnode.xbins) > 0)
        assert np.all(np.diff(cnode.ybins) > 0)
        return cnode


def bindices(node, rect):
    assert rect.normalized() == rect
    assert not rect.intersect(QRectF(*node.brect)).isEmpty()

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
        scale = contingencies.max()

    if scale > 0:
        P = contingencies / scale
    else:
        P = contingencies

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
#         P_max /= P_max.max()
        colors = 255 - colors[argmax.ravel()]

        # XXX: Non linear intensity scaling
        colors *= P_max.ravel().reshape(-1, 1)
        colors = colors.reshape(P_max.shape + (3,))
        colors = 255 - colors
    elif P.ndim == 2:
        palette = colorpalette.ColorPaletteBW()
        mix = P

#         mix = scipy.ndimage.filters.gaussian_filter(
#             mix, bandwidth, mode="constant")
#         mix /= mix.max() if total else 1.0

        colors = np.zeros((np.prod(mix.shape), 3)) + 255
        colors -= mix.ravel().reshape(-1, 1) * 255
        colors = colors.reshape(mix.shape + (3,))

    return colors


def compute_chi_squares(observes):
    # compute chi squares for left-right neighbours

    def get_estimates(observes):
        estimates = []
        for obs in observes:
            n = obs.sum()
            sum_rows = obs.sum(1)
            sum_cols = obs.sum(0)
            prob_rows = sum_rows / n
            prob_cols = sum_cols / n
            rows, cols = np.indices(obs.shape)
            est = np.zeros(obs.shape)
            est[rows, cols] = n * prob_rows[rows] * prob_cols[cols]
            estimates.append(est)
        return np.nan_to_num(np.array(estimates))

    estimates = get_estimates(observes)

    depth, rows, coll = np.indices(( observes.shape[0], observes.shape[1], observes.shape[2]-1 ))
    colr = coll + 1
    obs_dblstack = np.array([ observes[depth, rows, coll], observes[depth, rows, colr] ])
    obs_pairs = np.zeros(( obs_dblstack.shape[1], obs_dblstack.shape[2], obs_dblstack.shape[3], obs_dblstack.shape[0] ))
    depth, rows, coll, pairs = np.indices(obs_pairs.shape)
    obs_pairs[depth, rows, coll, pairs] = obs_dblstack[pairs, depth, rows, coll]

    depth, rows, coll = np.indices(( estimates.shape[0], estimates.shape[1], estimates.shape[2]-1 ))
    colr = coll + 1
    est_dblstack = np.array([ estimates[depth, rows, coll], estimates[depth, rows, colr] ])
    est_pairs = np.zeros(( est_dblstack.shape[1], est_dblstack.shape[2], est_dblstack.shape[3], est_dblstack.shape[0] ))
    depth, rows, coll, pairs = np.indices(est_pairs.shape)
    est_pairs[depth, rows, coll, pairs] = est_dblstack[pairs, depth, rows, coll]

    oe2e = (obs_pairs - est_pairs)**2 / est_pairs
    chi_squares_lr = np.nan_to_num(np.nansum(np.nansum(oe2e, axis=3), axis=0))

    # compute chi squares for up-down neighbours
    depth, rowu, cols = np.indices(( observes.shape[0], observes.shape[1]-1, observes.shape[2] ))
    rowd = rowu + 1
    obs_dblstack = np.array([ observes[depth, rowu, cols], observes[depth, rowd, cols] ])
    obs_pairs = np.zeros(( obs_dblstack.shape[1], obs_dblstack.shape[2], obs_dblstack.shape[3], obs_dblstack.shape[0] ))
    depth, rowu, cols, pairs = np.indices(obs_pairs.shape)
    obs_pairs[depth, rowu, cols, pairs] = obs_dblstack[pairs, depth, rowu, cols]

    depth, rowu, cols = np.indices(( estimates.shape[0], estimates.shape[1]-1, estimates.shape[2] ))
    rowd = rowu + 1
    est_dblstack = np.array([ estimates[depth, rowu, cols], estimates[depth, rowd, cols] ])
    est_pairs = np.zeros(( est_dblstack.shape[1], est_dblstack.shape[2], est_dblstack.shape[3], est_dblstack.shape[0] ))
    depth, rowu, cols, pairs = np.indices(est_pairs.shape)
    est_pairs[depth, rowu, cols, pairs] = est_dblstack[pairs, depth, rowu, cols]

    oe2e = (obs_pairs - est_pairs)**2 / est_pairs
    chi_squares_ud = np.nan_to_num(np.nansum(np.nansum(oe2e, axis=3), axis=0))

    return (chi_squares_lr, chi_squares_ud)


def main():
    import sip
    app = QtGui.QApplication([])
    w = OWHeatMap()
    w.show()
    w.raise_()
#     data = Orange.data.Table('iris')
#     data = Orange.data.Table('housing')
    data = Orange.data.Table('adult')
    w.set_data(data)
    rval = app.exec_()
    w.onDeleteWidget()

    sip.delete(w)
    del w
    app.processEvents()
    return rval

if __name__ == "__main__":
    import sys
    sys.exit(main())
