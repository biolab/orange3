from itertools import chain

import numpy as np

from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from AnyQt.QtCore import Qt


def numpy_repr(a):
    """ A numpy repr without summarization """
    opts = np.get_printoptions()
    try:
        np.set_printoptions(threshold=10**10)
        return repr(a)
    finally:
        np.set_printoptions(**opts)


def scatterplot_code(scatterplot_item):
    x = scatterplot_item.data['x']
    y = scatterplot_item.data['y']
    sizes = scatterplot_item.data["size"]

    code = []

    code.append("# data")
    code.append("x = {}".format(numpy_repr(x)))
    code.append("y = {}".format(numpy_repr(y)))

    code.append("# style")
    code.append("sizes = {}".format(numpy_repr(sizes)))

    def colortuple(color):
        return color.redF(), color.greenF(), color.blueF(), color.alphaF()

    edgecolors = np.array([colortuple(a.color()) for a in scatterplot_item.data["pen"]])
    facecolors = np.array([colortuple(a.color()) for a in scatterplot_item.data["brush"]])
    linewidths = np.array([a.widthF() for a in scatterplot_item.data["pen"]])

    pen_style = [a.style() for a in scatterplot_item.data["pen"]]
    no_pen = [s == Qt.NoPen for s in pen_style]
    linewidths[np.nonzero(no_pen)[0]] = 0

    code.append("edgecolors = {}".format(numpy_repr(edgecolors)))
    code.append("facecolors = {}".format(numpy_repr(facecolors)))
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
            # indices are unused
            code.append("plt.scatter(x=x, y=y, s=sizes**2/4, marker={},".format(repr(m)))
            code.append("            facecolors=facecolors, edgecolors=edgecolors,")
            code.append("            linewidths=linewidths)")
        else:
            code.append("indices = {}".format(numpy_repr(indices)))
            code.append("plt.scatter(x=x[indices], y=y[indices], s=sizes[indices]**2/4, "
                        "marker={},".format(repr(m)))
            code.append("            facecolors=facecolors[indices], "
                        "edgecolors=edgecolors[indices],")
            code.append("            linewidths=linewidths[indices])")

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


