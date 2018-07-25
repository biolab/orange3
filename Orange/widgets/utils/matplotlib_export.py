from itertools import chain

import numpy as np

from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPen, QBrush


def numpy_repr(a):
    """ A numpy repr without summarization """
    opts = np.get_printoptions()
    try:
        np.set_printoptions(threshold=10**10)
        return repr(a)
    finally:
        np.set_printoptions(**opts)


def compress_if_all_same(l):
    s = set(l)
    return s.pop() if len(s) == 1 else l


def is_sequence_not_string(a):
    if isinstance(a, str):
        return False
    try:
        iter(a)
        return True
    except TypeError:
        pass
    return False


def code_with_indices(data, data_name, indices, indices_name):
    if is_sequence_not_string(data) and indices is not None:
        return data_name + "[" + indices_name + "]"
    else:
        return data_name


def scatterplot_code(scatterplot_item):
    x = scatterplot_item.data['x']
    y = scatterplot_item.data['y']
    sizes = scatterplot_item.data["size"]

    code = []

    code.append("# data")
    code.append("x = {}".format(numpy_repr(x)))
    code.append("y = {}".format(numpy_repr(y)))

    code.append("# style")
    sizes = compress_if_all_same(sizes)
    if sizes == -1:
        sizes = None
    code.append("sizes = {}".format(numpy_repr(sizes)))

    def colortuple(pen):
        if isinstance(pen, (QPen, QBrush)):
            color = pen.color()
            return color.redF(), color.greenF(), color.blueF(), color.alphaF()
        return pen

    def width(pen):
        if isinstance(pen, QPen):
            return pen.widthF()
        return pen

    linewidths = np.array([width(a) for a in scatterplot_item.data["pen"]])

    def shown(a):
        if isinstance(a, (QPen, QBrush)):
            s = a.style()
            if s == Qt.NoPen or s == Qt.NoBrush or a.color().alpha() == 0:
                return False
        return True

    shown_edge = [shown(a) for a in scatterplot_item.data["pen"]]
    shown_brush = [shown(a) for a in scatterplot_item.data["brush"]]

    # return early if the scatterplot is all transparent
    if not any(shown_edge) and not any(shown_brush):
        return ""

    def do_colors(data_column, show):
        colors = [colortuple(a) for a in data_column]
        if all(a is None for a in colors):
            colors = None
        else:
            # replace None values with blue colors
            colors = np.array([((0, 0, 1, 1) if a is None else a)
                               for a in colors])
            # set alpha for hidden (Qt.NoPen, Qt.NoBrush) elements to zero
            colors[:, 3][np.array(show) == 0] = 0
        return colors

    edgecolors = do_colors(scatterplot_item.data["pen"], shown_edge)
    facecolors = do_colors(scatterplot_item.data["brush"], shown_brush)

    code.append("edgecolors = {}".format(numpy_repr(edgecolors)))
    code.append("facecolors = {}".format(numpy_repr(facecolors)))
    linewidths = compress_if_all_same(linewidths)
    code.append("linewidths = {}".format(numpy_repr(linewidths)))

    # possible_markers for scatterplot are in .graph.CurveSymbols
    def matplotlib_marker(m):
        if m == "t":
            return "^"
        elif m == "t2":
            return ">"
        elif m == "t3":
            return "<"
        elif m == "star":
            return "*"
        elif m == "+":
            return "P"
        elif m == "x":
            return "X"
        return m

    # TODO labels are missing

    # each marker requires one call to matplotlib's scatter!
    markers = np.array([matplotlib_marker(m) for m in scatterplot_item.data["symbol"]])
    for m in set(markers):
        indices = np.where(markers == m)[0]
        if np.all(indices == np.arange(x.shape[0])):
            indices = None
        if indices is not None:
            code.append("indices = {}".format(numpy_repr(indices)))

        def indexed(data, data_name, indices=indices):
            return code_with_indices(data, data_name, indices, "indices")

        code.append("plt.scatter(x={}, y={}, s={}, marker={},\n"
                    "            facecolors={}, edgecolors={},\n"
                    "            linewidths={})"
                    .format(indexed(x, "x"),
                            indexed(y, "y"),
                            (indexed(sizes, "sizes") + "**2/4") if sizes is not None else "sizes",
                            repr(m),
                            indexed(facecolors, "facecolors"),
                            indexed(edgecolors, "edgecolors"),
                            indexed(linewidths, "linewidths")
                           ))

    return "\n".join(code)


def scene_code(scene):

    code = []

    code.append("import matplotlib.pyplot as plt")
    code.append("from numpy import array")

    code.append("")
    code.append("plt.clf()")

    code.append("")

    for item in scene.items:
        if isinstance(item, ScatterPlotItem):
            code.append(scatterplot_code(item))

    # TODO currently does not work for graphs without axes and for multiple axes!
    for position, set_ticks, set_label in [("bottom", "plt.xticks", "plt.xlabel"),
                                           ("left", "plt.yticks", "plt.ylabel")]:
        axis = scene.getAxis(position)
        code.append("{}({})".format(set_label, repr(str(axis.labelText))))

        # textual tick labels
        if axis._tickLevels is not None:
            major_minor = list(chain(*axis._tickLevels))
            locs = [a[0] for a in major_minor]
            labels = [a[1] for a in major_minor]
            code.append("{}({}, {})".format(set_ticks, locs, repr(labels)))

    return "\n".join(code)
