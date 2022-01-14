import numpy as np
from scipy import sparse as sp
from AnyQt.QtCore import Qt, QRectF, QSizeF, QPointF, QLineF
from AnyQt.QtGui import QColor, QBrush, QPen
from AnyQt.QtWidgets import (
    QGraphicsWidget,
    QGraphicsRectItem,
    QGraphicsLinearLayout,
    QSizePolicy,
    QGraphicsLineItem,
)

import Orange.statistics.util as ut
from Orange.data.util import one_hot


class BarItem(QGraphicsWidget):
    """A single bar in a histogram representing one single target value."""
    def __init__(self, width, height, color, parent=None):
        super().__init__(parent=parent)
        self.width = width
        self.height = height
        self.color = color
        if not isinstance(self.color, QColor):
            self.color = QColor(self.color)

        self.__rect = QGraphicsRectItem(0, 0, self.width, self.height, self)
        self.__rect.setPen(QPen(Qt.NoPen))
        self.__rect.setBrush(QBrush(self.color))

    def boundingRect(self):
        return self.__rect.boundingRect()

    def sizeHint(self, which, constraint):
        return self.boundingRect().size()

    def sizePolicy(self):
        return QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class ProportionalBarItem(QGraphicsLinearLayout):
    """A bar that fills draws ``'BarItem'`` objects given some proportions.

    Parameters
    ----------
    distribution : np.ndarray
        Contains the counts of individual target values that belong to the
        particular bin. This can have length 1 if there is no target class.
    colors : Optional[Iterable[QColor]]
        If colors are passed, they must match the shape of the distribution.
        The bars will be colored according to these values, where the indices
        in the distribution must match the color indices.
    bar_size : Union[int, float]
        The width of the bar.
    height : Union[int, float]
        The height of the bar.

    """

    def __init__(self, distribution, bar_size=10, height=100, colors=None):
        super().__init__()

        self.distribution = distribution

        assert not colors or len(distribution) is len(colors), \
            'If colors are provided, they must match the shape of distribution'
        self.colors = colors

        self.height = height
        self.setOrientation(Qt.Vertical)
        self._bar_size = bar_size

        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

        self._draw_bars()

    def _draw_bars(self):
        heights, dist_sum = self.distribution, self.distribution.sum()
        # If the number of instances within a column is not 0, divide by that
        # sum to get the proportional height, otherwise set the height to 0
        heights *= (dist_sum ** -1 if dist_sum != 0 else 0) * self.height

        for idx, height in enumerate(heights):
            color = self.colors[idx] if self.colors else QColor('#ccc')
            self.addItem(BarItem(width=self._bar_size, height=height, color=color))

    def sizeHint(self, which, constraint):
        return QSizeF(self._bar_size, self.height)

    def sizePolicy(self):
        return QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


# The price of flexibility is complexity...
# pylint: disable=too-many-instance-attributes
class Histogram(QGraphicsWidget):
    """A basic histogram widget.

    Parameters
    ----------
        data : Table
        variable : Union[int, str, Variable]
        parent : QObject
        height : Union[int, float]
        width : Union[int, float]
        side_padding : Union[int, float]
            Specify the padding between the edges of the histogram and the
            first and last bars.
        top_padding : Union[int, float]
            Specify the padding between the top of the histogram and the
            highest bar.
        bar_spacing : Union[int, float]
            Specify the amount of spacing to place between individual bars.
        border : Union[Tuple[Union[int, float]], int, float]
            Can be anything that can go into the ``'QColor'`` constructor.
            Draws a border around the entire histogram in a given color.
        border_color : Union[QColor, str]
        class_index : int
            The index of the target variable in ``'data'``.
        n_bins : int

    """

    def __init__(self, data, variable, parent=None, height=200,
                 width=300, side_padding=5, top_padding=20, bottom_padding=0,
                 bar_spacing=4,
                 border=0, border_color=None, color_attribute=None, n_bins=10):
        super().__init__(parent)
        self.height, self.width = height, width
        self.padding = side_padding
        self.bar_spacing = bar_spacing

        self.data = data
        self.attribute = data.domain[variable]

        self.x = data.get_column_view(self.attribute)[0].astype(np.float64)
        self.x_nans = np.isnan(self.x)
        self.x = self.x[~self.x_nans]

        if self.attribute.is_discrete:
            self.n_bins = len(self.attribute.values)
        elif self.attribute.is_continuous:
            # If the attribute is continuous but contains fewer values than the
            # bins, it is better to assign each their own bin. We will require
            # at least 2 bins so that the histogram still visually makes sense
            # except if there is only a single value, then we use 3 bins for
            # symmetry
            num_unique = ut.nanunique(self.x).shape[0]
            if num_unique == 1:
                self.n_bins = 3
            else:
                self.n_bins = min(max(2, num_unique), n_bins)

        # Handle target variable index
        self.color_attribute = color_attribute
        if self.color_attribute is not None:
            self.target_var = data.domain[color_attribute]
            self.y = data.get_column_view(color_attribute)[0]
            self.y = self.y[~self.x_nans]
            if not np.issubdtype(self.y.dtype, np.number):
                self.y = self.y.astype(np.float64)
        else:
            self.target_var, self.y = None, None

        # Borders
        self.border_color = border_color if border_color is not None else '#000'
        if isinstance(border, tuple):
            assert len(border) == 4, 'Border tuple must be of size 4.'
            self.border = border
        else:
            self.border = (border, border, border, border)
        t, r, b, l = self.border

        def _draw_border(point_1, point_2, border_width, parent):
            pen = QPen(QColor(self.border_color))
            pen.setCosmetic(True)
            pen.setWidth(border_width)
            line = QGraphicsLineItem(QLineF(point_1, point_2), parent)
            line.setPen(pen)
            return line

        top_left = QPointF(0, 0)
        bottom_left = QPointF(0, self.height)
        top_right = QPointF(self.width, 0)
        bottom_right = QPointF(self.width, self.height)

        self.border_top = _draw_border(top_left, top_right, t, self) if t else None
        self.border_bottom = _draw_border(bottom_left, bottom_right, b, self) if b else None
        self.border_left = _draw_border(top_left, bottom_left, l, self) if l else None
        self.border_right = _draw_border(top_right, bottom_right, r, self) if r else None

        # _plot_`dim` accounts for all the paddings and spacings
        self._plot_height = self.height
        self._plot_height -= top_padding + bottom_padding
        self._plot_height -= t / 4 + b / 4

        self._plot_width = self.width
        self._plot_width -= 2 * side_padding
        self._plot_width -= (self.n_bins - 2) * bar_spacing
        self._plot_width -= l / 4 + r / 4

        self.__layout = QGraphicsLinearLayout(Qt.Horizontal, self)
        self.__layout.setContentsMargins(
            side_padding + r / 2,
            top_padding + t / 2,
            side_padding + l / 2,
            bottom_padding + b / 2
        )
        self.__layout.setSpacing(bar_spacing)

        # If the data contains any non-NaN values, we can draw a histogram
        if self.x.size > 0:
            self.edges, self.distributions = self._histogram()
            self._draw_histogram()

    def _get_histogram_edges(self):
        """Get the edges in the histogram based on the attribute type.

        In case of a continuous variable, we split the variable range into
        n bins. In case of a discrete variable, bins don't make sense, so we
        just return the attribute values.

        This will return the staring and ending edge, not just the edges in
        between (in the case of a continuous variable).

        Returns
        -------
        np.ndarray

        """
        if self.attribute.is_discrete:
            return np.array([self.attribute.to_val(v) for v in self.attribute.values])
        else:
            edges = np.linspace(ut.nanmin(self.x), ut.nanmax(self.x), self.n_bins)
            edge_diff = edges[1] - edges[0]
            edges = np.hstack((edges, [edges[-1] + edge_diff]))

            # If the variable takes on a single value, we still need to spit
            # out some reasonable bin edges
            if np.all(edges == edges[0]):
                edges = np.array([edges[0] - 1, edges[0], edges[0] + 1])

            return edges

    def _get_bin_distributions(self, bin_indices):
        """Compute the distribution of instances within bins.

        Parameters
        ----------
        bin_indices : np.ndarray
            An array with same shape as `x` but containing the bin index of the
            instance.

        Returns
        -------
        np.ndarray
            A 2d array; the first dimension represents different bins, the
            second - the counts of different target values.

        """
        if self.target_var and self.target_var.is_discrete:
            y = self.y
            # TODO This probably also isn't the best handling of sparse data...
            if sp.issparse(y):
                y = np.squeeze(np.array(y.todense()))

            # Since y can contain missing values, we need to filter them out as
            # well as their corresponding `x` values
            y_nan_mask = np.isnan(y)
            y, bin_indices = y[~y_nan_mask], bin_indices[~y_nan_mask]
            y = one_hot(y, dim=len(self.target_var.values))

            bins = np.arange(self.n_bins)[:, np.newaxis]
            mask = bin_indices == bins
            distributions = np.zeros((self.n_bins, y.shape[1]))
            for bin_idx in range(self.n_bins):
                distributions[bin_idx] = y[mask[bin_idx]].sum(axis=0)
        else:
            distributions, _ = ut.bincount(bin_indices.astype(np.int64))
            # To keep things consistent across different variable types, we
            # want to return a 2d array where the first dim represent different
            # bins, and the second the distributions.
            distributions = distributions[:, np.newaxis]

        return distributions

    def _histogram(self):
        assert self.x.size > 0, 'Cannot calculate histogram on empty array'
        edges = self._get_histogram_edges()

        if self.attribute.is_discrete:
            bin_indices = self.x
            # TODO It probably isn't a very good idea to convert a sparse row
            # to a dense array... Converts sparse to 1d numpy array
            if sp.issparse(bin_indices):
                bin_indices = np.squeeze(np.asarray(
                    bin_indices.todense(), dtype=np.int64
                ))
        elif self.attribute.is_continuous:
            bin_indices = ut.digitize(self.x, bins=edges[1:-1]).flatten()

        distributions = self._get_bin_distributions(bin_indices)

        return edges, distributions

    def _draw_histogram(self):
        # In case the data for the variable were all NaNs, then the
        # distributions will be empty, and we don't need to display any bars
        if self.x.size == 0:
            return

        # In case we have a target var, but the values are all NaNs, then there
        # is no sense in displaying anything
        if self.target_var:
            y_nn = self.y[~np.isnan(self.y)]
            if y_nn.size == 0:
                return

        if self.distributions.ndim > 1:
            largest_bin_count = self.distributions.sum(axis=1).max()
        else:
            largest_bin_count = self.distributions.max()

        bar_size = self._plot_width / self.n_bins

        for distr, bin_colors in zip(self.distributions, self._get_colors()):
            bin_count = distr.sum()
            bar_height = bin_count / largest_bin_count * self._plot_height

            bar_layout = QGraphicsLinearLayout(Qt.Vertical)
            bar_layout.setSpacing(0)
            bar_layout.addStretch()
            self.__layout.addItem(bar_layout)

            bar = ProportionalBarItem(  # pylint: disable=blacklisted-name
                distribution=distr, colors=bin_colors, height=bar_height,
                bar_size=bar_size,
            )
            bar_layout.addItem(bar)

        self.layout()

    def _get_colors(self):
        """Compute colors for different kinds of histograms."""
        target = self.target_var
        if target and target.is_discrete:
            colors = [list(target.palette)[:len(target.values)]] * self.n_bins

        elif self.target_var and self.target_var.is_continuous:
            palette = self.target_var.palette

            bins = np.arange(self.n_bins)[:, np.newaxis]
            edges = self.edges if self.attribute.is_discrete else self.edges[1:-1]
            bin_indices = ut.digitize(self.x, bins=edges)
            mask = bin_indices == bins

            colors = []
            for bin_idx in range(self.n_bins):
                biny = self.y[mask[bin_idx]]
                if np.isfinite(biny).any():
                    mean = ut.nanmean(biny) / ut.nanmax(self.y)
                else:
                    mean = 0  # bin is empty, color does not matter
                colors.append([palette.value_to_qcolor(mean)])

        else:
            colors = [[QColor('#ccc')]] * self.n_bins

        return colors

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def sizeHint(self, which, constraint):
        return QSizeF(self.width, self.height)

    def sizePolicy(self):
        return QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


if __name__ == '__main__':
    import sys
    from Orange.data.table import Table
    from AnyQt.QtWidgets import (  # pylint: disable=ungrouped-imports
        QGraphicsView, QGraphicsScene, QApplication, QWidget
    )

    app = QApplication(sys.argv)
    widget = QWidget()
    widget.resize(500, 300)
    scene = QGraphicsScene(widget)
    view = QGraphicsView(scene, widget)
    dataset = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    histogram = Histogram(
        dataset, variable=0, height=300, width=500, n_bins=20, bar_spacing=2,
        border=(0, 0, 5, 0), border_color='#000', color_attribute='iris',
    )
    scene.addItem(histogram)

    widget.show()
    app.exec()
