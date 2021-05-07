import numpy as np
from pyqtgraph import PlotWidget, mkPen, InfiniteLine, PlotCurveItem, \
    TextItem, Point
from AnyQt.QtGui import QColor
from AnyQt.QtCore import Qt


class SliderGraph(PlotWidget):
    """
    An widget graph element that shows a line plot with more sequences. It
    also plot a vertical line that can be moved left and right by a user. When
    the line is moved a callback function is called with selected value (on
    x axis).

    Attributes
    ----------
    x_axis_label : str
        A text label for x axis
    y_axis_label : str
        A text label for y axis
    callback : callable
        A function which is called when selection is changed.
    background : str, optional (default: "w")
        Plot background color
    """

    def __init__(self, x_axis_label, y_axis_label, callback):
        super().__init__(background="w")

        axis = self.getAxis("bottom")
        axis.setLabel(x_axis_label)
        axis = self.getAxis("left")
        axis.setLabel(y_axis_label)

        self.getViewBox().setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(False, False)
        self.showGrid(True, True, alpha=0.5)
        self.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))
        self.hideButtons()

        # tuples to store horisontal lines and labels
        self.plot_horlabel = []
        self.plot_horline = []
        self._line = None
        self.callback = callback

        # variables to store sequences
        self.sequences = None
        self.x = None
        self.selection_limit = None
        self.data_increasing = None  # true if data mainly increasing

    def update(self, x, y, colors, cutpoint_x=None, selection_limit=None,
               names=None):
        """
        Function replots a graph.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional array with X coordinates of the points
        y : array-like
            List of np.ndarrays that contains an array of Y values for each
            sequence.
        colors : array-like
            List of Qt colors (eg. Qt.red) for each sequence.
        cutpoint_x : int, optional
            A starting cutpoint - the location of the vertical line.
        selection_limit : tuple
            The tuple of two values that limit the range for selection.
        names : array-like
            The name of each sequence that shows in the legend, if None
            legend is not shown.
        legend_anchor : array-like
            The anchor of the legend in the graph
        """
        self.clear_plot()
        if names is None:
            names = [None] * len(y)

        self.sequences = y
        self.x = x
        self.selection_limit = selection_limit

        self.data_increasing = [np.sum(d[1:] - d[:-1]) > 0 for d in y]

        # plot sequence
        for s, c, n, inc in zip(y, colors, names, self.data_increasing):
            c = QColor(c)
            self.plot(x, s, pen=mkPen(c, width=2), antialias=True)

            if n is not None:
                label = TextItem(
                    text=n, anchor=(0, 1), color=QColor(0, 0, 0, 128))
                label.setPos(x[-1], s[-1])
                self._set_anchor(label, len(x) - 1, inc)
                self.addItem(label)

        self._plot_cutpoint(cutpoint_x)
        self.autoRange()

    def clear_plot(self):
        """
        This function clears the plot and removes data.
        """
        self.clear()
        self.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))
        self.plot_horlabel = []
        self.plot_horline = []
        self._line = None
        self.sequences = None

    def set_cut_point(self, x):
        """
        This function sets the cutpoint (selection line) at the specific
        location.

        Parameters
        ----------
        x : int
            Cutpoint location at the x axis.
        """
        self._plot_cutpoint(x)

    def _plot_cutpoint(self, x):
        """
        Function plots the cutpoint.

        Parameters
        ----------
        x : int
            Cutpoint location.
        """
        if x is None:
            self._line = None
            return
        if self._line is None:
            # plot interactive vertical line
            self._line = InfiniteLine(
                angle=90, pos=x, movable=True,
                bounds=self.selection_limit if self.selection_limit is not None
                else (self.x.min(), self.x.max())
            )
            self._line.setCursor(Qt.SizeHorCursor)
            self._line.setPen(mkPen(QColor(Qt.black), width=2))
            self._line.sigPositionChanged.connect(self._on_cut_changed)
            self.addItem(self._line)
        else:
            self._line.setValue(x)

        self._update_horizontal_lines()

    def _plot_horizontal_lines(self):
        """
        Function plots the vertical dashed lines that points to the selected
        sequence values at the y axis.
        """
        for _ in range(len(self.sequences)):
            self.plot_horline.append(PlotCurveItem(
                pen=mkPen(QColor(Qt.blue), style=Qt.DashLine)))
            self.plot_horlabel.append(TextItem(
                color=QColor(Qt.black), anchor=(0, 1)))
        for item in self.plot_horlabel + self.plot_horline:
            self.addItem(item)

    def _set_anchor(self, label, cutidx, inc):
        """
        This function set the location of the text label around the selected
        point at the curve. It place the text such that it is not plotted
        at the line.

        Parameters
        ----------
        label : TextItem
            Text item that needs to have location set.
        cutidx : int
            The index of the selected element in the list. If index in first
            part of the list we put label on the right side else on the left,
            such that it does not disappear at the graph edge.
        inc : bool
            This parameter tels whether the curve value is increasing or
            decreasing.
        """
        if inc:
            label.anchor = Point(0, 0) if cutidx < len(self.x) / 2 \
                else Point(1, 1)
        else:
            label.anchor = Point(0, 1) if cutidx < len(self.x) / 2 \
                else Point(1, 0)

    def _update_horizontal_lines(self):
        """
        This function update the horisontal lines when selection changes.
        If lines are present jet it calls the function to init them.
        """
        if not self.plot_horline:  # init horizontal lines
            self._plot_horizontal_lines()

        # in every case set their position
        location = int(round(self._line.value()))
        cutidx = np.searchsorted(self.x, location)
        minx = np.min(self.x)
        for s, curve, label, inc in zip(
                self.sequences, self.plot_horline, self.plot_horlabel,
                self.data_increasing):
            y = s[cutidx]
            curve.setData([minx, location], [y, y])
            self._set_anchor(label, cutidx, inc)
            label.setPos(location, y)
            label.setPlainText("{:.3f}".format(y))

    def _on_cut_changed(self, line):
        """
        This function is called when selection changes. It extract the selected
        value and calls the callback function.

        Parameters
        ----------
        line : InfiniteLine
            The cutpoint - selection line.
        """
        # cut changed by means of a cut line over the scree plot.
        value = int(round(line.value()))

        # vertical line can take only int positions
        self._line.setValue(value)

        self._update_horizontal_lines()
        self.callback(value)
