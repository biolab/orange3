"""

TODO:
  - Sorting by standard deviation: Use coefficient of variation (std/mean)
    or quartile coefficient of dispersion (Q3 - Q1) / (Q3 + Q1)
  - Standard deviation for nominal: try out Variation ratio (1 - n_mode/N)
"""

import datetime
from enum import IntEnum
from itertools import chain
from typing import Any, Optional, Tuple, List

import numpy as np
import scipy.stats as ss
import scipy.sparse as sp
from AnyQt.QtCore import Qt, QSize, QRectF, QModelIndex, pyqtSlot, \
    QItemSelection, QItemSelectionRange, QItemSelectionModel
from AnyQt.QtGui import QPainter, QColor
from AnyQt.QtWidgets import QStyledItemDelegate, QGraphicsScene, QTableView, \
    QHeaderView, QStyle, QStyleOptionViewItem

import Orange.statistics.util as ut
from Orange.data import Table, StringVariable, DiscreteVariable, \
    ContinuousVariable, TimeVariable, Domain, Variable
from Orange.util import utc_from_timestamp
from Orange.widgets import widget, gui
from Orange.widgets.data.utils.histogram import Histogram
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel, AbstractSortTableModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview


def _categorical_entropy(x):
    """Compute the entropy of a dense/sparse matrix, column-wise. Assuming
    categorical values."""
    p = [ut.bincount(row)[0] for row in x.T]
    p = [pk / np.sum(pk) for pk in p]
    return np.fromiter((ss.entropy(pk) for pk in p), dtype=np.float64)


def coefficient_of_variation(x: np.ndarray) -> np.ndarray:
    mu = ut.nanmean(x, axis=0)
    mask = ~np.isclose(mu, 0, atol=1e-12)
    result = np.full_like(mu, fill_value=np.inf)
    result[mask] = np.sqrt(ut.nanvar(x, axis=0)[mask]) / mu[mask]
    return result


def format_time_diff(start, end, round_up_after=2):
    """Return an approximate human readable time difference between two dates.

    Parameters
    ----------
    start : int
        Unix timestamp
    end : int
        Unix timestamp
    round_up_after : int
        The number of time units before we round up to the next, larger time
        unit e.g. setting to 2 will allow up to 2 days worth of hours to be
        shown, after that the difference is shown in days. Or put another way
        we will show from 1-48 hours before switching to days.

    Returns
    -------
    str

    """
    start = utc_from_timestamp(start)
    end = utc_from_timestamp(end)
    diff = abs(end - start)  # type: datetime.timedelta

    # Get the different resolutions
    seconds = diff.total_seconds()
    minutes = seconds // 60
    hours = minutes // 60
    days = diff.days
    weeks = days // 7
    months = (end.year - start.year) * 12 + end.month - start.month
    years = months // 12

    # Check which resolution is most appropriate
    if years >= round_up_after:
        return '~%d years' % years
    elif months >= round_up_after:
        return '~%d months' % months
    elif weeks >= round_up_after:
        return '~%d weeks' % weeks
    elif days >= round_up_after:
        return '~%d days' % days
    elif hours >= round_up_after:
        return '~%d hours' % hours
    elif minutes >= round_up_after:
        return '~%d minutes' % minutes
    else:
        return '%d seconds' % seconds


class FeatureStatisticsTableModel(AbstractSortTableModel):
    CLASS_VAR, META, ATTRIBUTE = range(3)
    COLOR_FOR_ROLE = {
        CLASS_VAR: QColor(160, 160, 160),
        META: QColor(220, 220, 200),
        ATTRIBUTE: QColor(255, 255, 255),
    }

    HIDDEN_VAR_TYPES = (StringVariable,)

    class Columns(IntEnum):
        ICON, NAME, DISTRIBUTION, CENTER, MEDIAN, DISPERSION, MIN, MAX, \
        MISSING = range(9)

        @property
        def name(self):
            return {self.ICON: '',
                    self.NAME: 'Name',
                    self.DISTRIBUTION: 'Distribution',
                    self.CENTER: 'Mean',
                    self.MEDIAN: 'Median',
                    self.DISPERSION: 'Dispersion',
                    self.MIN: 'Min.',
                    self.MAX: 'Max.',
                    self.MISSING: 'Missing',
                    }[self.value]

        @property
        def index(self):
            return self.value

        @classmethod
        def from_index(cls, index):
            return cls(index)

    def __init__(self, data=None, parent=None):
        """

        Parameters
        ----------
        data : Optional[Table]
        parent : Optional[QWidget]

        """
        super().__init__(parent)

        self.table = None  # type: Optional[Table]
        self.domain = None  # type: Optional[Domain]
        self.target_var = None  # type: Optional[Variable]
        self.n_attributes = self.n_instances = 0

        self.__attributes = self.__class_vars = self.__metas = None
        # sets of variables for fast membership tests
        self.__attributes_set = set()
        self.__class_vars_set = set()
        self.__metas_set = set()
        self.__distributions_cache = {}

        no_data = np.array([])
        self._variable_types = self._variable_names = no_data
        self._min = self._max = self._center = self._median = no_data
        self._dispersion = no_data
        self._missing = no_data
        # Clear model initially to set default values
        self.clear()

        self.set_data(data)

    def set_data(self, data):
        if data is None:
            self.clear()
            return

        self.beginResetModel()
        self.table = data
        self.domain = domain = data.domain
        self.target_var = None

        self.__attributes = self.__filter_attributes(
            domain.attributes, self.table.X)
        self.__class_vars = self.__filter_attributes(
            domain.class_vars, self.table.Y.reshape((len(self.table.Y), -1)))
        self.__metas = self.__filter_attributes(
            domain.metas, self.table.metas)
        self.__attributes_set = set(self.__metas[0])
        self.__class_vars_set = set(self.__class_vars[0])
        self.__metas_set = set(self.__metas[0])
        self.n_attributes = len(self.variables)
        self.n_instances = len(data)

        self.__distributions_cache = {}
        self.__compute_statistics()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.table = self.domain = self.target_var = None
        self.n_attributes = self.n_instances = 0
        self.__attributes = (np.array([]), np.array([]))
        self.__class_vars = (np.array([]), np.array([]))
        self.__metas = (np.array([]), np.array([]))
        self.__attributes_set = set()
        self.__class_vars_set = set()
        self.__metas_set = set()
        self.__distributions_cache.clear()
        self.endResetModel()

    @property
    def variables(self):
        matrices = [self.__attributes[0], self.__class_vars[0], self.__metas[0]]
        if not any(m.size for m in matrices):
            return []
        return np.hstack(matrices)

    @staticmethod
    def _attr_indices(attrs):
        # type: (List) -> Tuple[List[int], List[int], List[int], List[int]]
        """Get the indices of different attribute types eg. discrete."""
        disc_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, DiscreteVariable)]
        cont_var_idx = [i for i, attr in enumerate(attrs)
                        if isinstance(attr, ContinuousVariable)
                        and not isinstance(attr, TimeVariable)]
        time_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, TimeVariable)]
        string_var_idx = [i for i, attr in enumerate(attrs) if isinstance(attr, StringVariable)]
        return disc_var_idx, cont_var_idx, time_var_idx, string_var_idx

    def __filter_attributes(self, attributes, matrix):
        """Filter out variables which shouldn't be visualized."""
        attributes, matrix = np.asarray(attributes), matrix
        mask = [idx for idx, attr in enumerate(attributes)
                if not isinstance(attr, self.HIDDEN_VAR_TYPES)]
        return attributes[mask], matrix[:, mask]

    def __compute_statistics(self):
        # Since data matrices can of mixed sparsity, we need to compute
        # attributes separately for each of them.
        matrices = [self.__attributes, self.__class_vars, self.__metas]
        # Filter out any matrices with size 0
        matrices = list(filter(lambda tup: tup[1].size, matrices))

        self._variable_types = np.array([type(var) for var in self.variables])
        self._variable_names = np.array([var.name.lower() for var in self.variables])
        self._min = self.__compute_stat(
            matrices,
            discrete_f=lambda x: ut.nanmin(x, axis=0),
            continuous_f=lambda x: ut.nanmin(x, axis=0),
            time_f=lambda x: ut.nanmin(x, axis=0),
        )
        self._dispersion = self.__compute_stat(
            matrices,
            discrete_f=_categorical_entropy,
            continuous_f=coefficient_of_variation,
        )
        self._missing = self.__compute_stat(
            matrices,
            discrete_f=lambda x: ut.countnans(x, axis=0),
            continuous_f=lambda x: ut.countnans(x, axis=0),
            string_f=lambda x: (x == StringVariable.Unknown).sum(axis=0),
            time_f=lambda x: ut.countnans(x, axis=0),
            default_val=len(matrices[0]) if matrices else 0
        )
        self._max = self.__compute_stat(
            matrices,
            discrete_f=lambda x: ut.nanmax(x, axis=0),
            continuous_f=lambda x: ut.nanmax(x, axis=0),
            time_f=lambda x: ut.nanmax(x, axis=0),
        )

        # Since scipy apparently can't do mode on sparse matrices, cast it to
        # dense. This can be very inefficient for large matrices, and should
        # be changed
        def __mode(x, *args, **kwargs):
            if sp.issparse(x):
                x = x.todense(order="C")
            # return ss.mode(x, *args, **kwargs)[0]
            return ut.nanmode(x, *args, **kwargs)[0]  # Temporary replacement for scipy

        self._center = self.__compute_stat(
            matrices,
            discrete_f=None,
            continuous_f=lambda x: ut.nanmean(x, axis=0),
            time_f=lambda x: ut.nanmean(x, axis=0),
        )

        self._median = self.__compute_stat(
            matrices,
            discrete_f=lambda x: __mode(x, axis=0),
            continuous_f=lambda x: ut.nanmedian(x, axis=0),
            time_f=lambda x: ut.nanmedian(x, axis=0),
        )

    def get_statistics_matrix(self, variables=None, return_labels=False):
        """Get the numeric computed statistics in a single matrix. Optionally,
        we can specify for which variables we want the stats. Also, we can get
        the string column names as labels if desired.

        Parameters
        ----------
        variables : Iterable[Union[Variable, int, str]]
            Return statistics for only the variables specified. Accepts all
            formats supported by `domain.index`
        return_labels : bool
            In addition to the statistics matrix, also return string labels for
            the columns of the matrix e.g. 'Mean' or 'Dispersion', as specified
            in `Columns`.

        Returns
        -------
        Union[Tuple[List[str], np.ndarray], np.ndarray]

        """
        if self.table is None:
            return np.atleast_2d([])

        # If a list of variables is given, select only corresponding stats
        # variables can be a list or array, pylint: disable=len-as-condition
        if variables is not None and len(variables) != 0:
            indices = [self.domain.index(var) for var in variables]
        else:
            indices = ...

        matrix = np.vstack((
            self._center[indices], self._median[indices],
            self._dispersion[indices],
            self._min[indices], self._max[indices], self._missing[indices],
        )).T

        # Return string labels for the returned matrix columns e.g. 'Mean',
        # 'Dispersion' if requested
        if return_labels:
            labels = [self.Columns.CENTER.name, self.Columns.MEDIAN.name,
                      self.Columns.DISPERSION.name,
                      self.Columns.MIN.name, self.Columns.MAX.name,
                      self.Columns.MISSING.name]
            return labels, matrix

        return matrix

    def __compute_stat(self, matrices, discrete_f=None, continuous_f=None,
                       time_f=None, string_f=None, default_val=np.nan):
        """Apply functions to appropriate variable types. The default value is
        returned if there is no function defined for specific variable types.
        """
        if not matrices:
            return np.array([])

        results = []
        for variables, x in matrices:
            result = np.full(len(variables), default_val)

            # While the following caching and checks are messy, the indexing
            # turns out to be a bottleneck for large datasets, so a single
            # indexing operation improves performance
            *idxs, str_idx = self._attr_indices(variables)
            for func, idx in zip((discrete_f, continuous_f, time_f), idxs):
                idx = np.array(idx)
                if func and idx.size:
                    x_ = x[:, idx]
                    if x_.size:
                        if not np.issubdtype(x_.dtype, np.number):
                            x_ = x_.astype(np.float64)
                        try:
                            finites = np.isfinite(x_)
                        except TypeError:
                            result[idx] = func(x_)
                        else:
                            mask = np.any(finites, axis=0)
                            if np.any(mask):
                                result[idx[mask]] = func(x_[:, mask])
            if string_f:
                x_ = x[:, str_idx]
                if x_.size:
                    if x_.dtype is not np.object:
                        x_ = x_.astype(np.object)
                    result[str_idx] = string_f(x_)

            results.append(result)

        return np.hstack(results)

    def sortColumnData(self, column):
        """Prepare the arrays with which we will sort the rows. If we want to
        sort based on a single value e.g. the name, return a 1d array.
        Sometimes we may want to sort by multiple criteria, comparing
        continuous variances with discrete entropies makes no sense, so we want
        to group those variable types together.
        """
        # Prepare indices for variable types so we can group them together
        order = [ContinuousVariable, TimeVariable,
                 DiscreteVariable, StringVariable]
        mapping = {var: idx for idx, var in enumerate(order)}
        vmapping = np.vectorize(mapping.__getitem__)
        var_types_indices = vmapping(self._variable_types)

        # Store the variable name sorted indices so we can pass a default
        # order when sorting by multiple keys
        # Double argsort is "inverse" argsort:
        # data will be *sorted* by these indices
        var_name_indices = np.argsort(np.argsort(self._variable_names))

        # Prepare vartype indices so ready when needed
        disc_idx, _, time_idx, str_idx = self._attr_indices(self.variables)

        # Sort by: (type)
        if column == self.Columns.ICON:
            return var_types_indices
        # Sort by: (name)
        elif column == self.Columns.NAME:
            # We use `_variable_names` here and not the indices because the
            # last (or single) row is actually sorted and we don't want to sort
            # the indices
            return self._variable_names
        # Sort by: (None)
        elif column == self.Columns.DISTRIBUTION:
            return np.ones_like(var_types_indices)
        # Sort by: (type, center)
        elif column == self.Columns.CENTER:
            # Sorting discrete or string values by mean makes no sense
            vals = np.array(self._center)
            vals[disc_idx] = var_name_indices[disc_idx]
            vals[str_idx] = var_name_indices[str_idx]
            return np.vstack((var_types_indices, np.zeros_like(vals), vals)).T
        # Sort by: (type, median)
        elif column == self.Columns.MEDIAN:
            # Sorting discrete or string values by median makes no sense
            vals = np.array(self._median)
            vals[disc_idx] = var_name_indices[disc_idx]
            vals[str_idx] = var_name_indices[str_idx]
            return np.vstack((var_types_indices, np.zeros_like(vals), vals)).T
        # Sort by: (type, dispersion)
        elif column == self.Columns.DISPERSION:
            # Sort time variables by their dispersion, which is not stored in
            # the dispersion array
            vals = np.array(self._dispersion)
            vals[time_idx] = self._max[time_idx] - self._min[time_idx]
            return np.vstack((var_types_indices, np.zeros_like(vals), vals)).T
        # Sort by: (type, min)
        elif column == self.Columns.MIN:
            # Sorting discrete or string values by min makes no sense
            vals = np.array(self._min)
            vals[disc_idx] = var_name_indices[disc_idx]
            vals[str_idx] = var_name_indices[str_idx]
            return np.vstack((var_types_indices, np.zeros_like(vals), vals)).T
        # Sort by: (type, max)
        elif column == self.Columns.MAX:
            # Sorting discrete or string values by min makes no sense
            vals = np.array(self._max)
            vals[disc_idx] = var_name_indices[disc_idx]
            vals[str_idx] = var_name_indices[str_idx]
            return np.vstack((var_types_indices, np.zeros_like(vals), vals)).T
        # Sort by: (missing)
        elif column == self.Columns.MISSING:
            return self._missing

        return None

    def _sortColumnData(self, column):
        """Allow sorting with 2d arrays."""
        data = np.asarray(self.sortColumnData(column))
        data = data[self.mapToSourceRows(Ellipsis)]

        assert data.ndim <= 2, 'Data should be at most 2-dimensional'
        return data

    def _argsortData(self, data, order):
        if data.ndim == 1:
            if np.issubdtype(data.dtype, np.number):
                if order == Qt.DescendingOrder:
                    data = -data
                indices = np.argsort(data, kind='stable')
                # Always sort NaNs last
                if np.issubdtype(data.dtype, np.number):
                    indices = np.roll(indices, -np.isnan(data).sum())
            else:
                # When not sorting by numbers, we can't do data = -data, but
                # use indices = indices[::-1] instead. This is not stable, but
                # doesn't matter because we use this only for variable names
                # which are guaranteed to be unique
                indices = np.argsort(data)
                if order == Qt.DescendingOrder:
                    indices = indices[::-1]
        else:
            assert np.issubdtype(data.dtype, np.number), \
                'We do not deal with non numeric values in sorting by ' \
                'multiple values'
            if order == Qt.DescendingOrder:
                data[:, -1] = -data[:, -1]

            # In order to make sure NaNs always appear at the end, insert a
            # indicator whether NaN or not. Note that the data array must
            # contain an empty column of zeros at index -2 since inserting an
            # extra column after the fact can result in a MemoryError for data
            # with a large amount of variables
            assert np.all(data[:, -2] == 0), \
                'Add an empty column of zeros at index -2 to accomodate NaNs'
            np.isnan(data[:, -1], out=data[:, -2])

            indices = np.lexsort(np.flip(data.T, axis=0))

        return indices

    def headerData(self, section, orientation, role):
        # type: (int, Qt.Orientation, Qt.ItemDataRole) -> Any
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.Columns.from_index(section).name

        return None

    def data(self, index, role):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        def background():
            if attribute in self.__attributes_set:
                return self.COLOR_FOR_ROLE[self.ATTRIBUTE]
            if attribute in self.__metas_set:
                return self.COLOR_FOR_ROLE[self.META]
            if attribute in self.__class_vars_set:
                return self.COLOR_FOR_ROLE[self.CLASS_VAR]
            return None

        def text_alignment():
            if column == self.Columns.NAME:
                return Qt.AlignLeft | Qt.AlignVCenter
            return Qt.AlignRight | Qt.AlignVCenter

        def decoration():
            if column == self.Columns.ICON:
                return gui.attributeIconDict[attribute]
            return None

        def display():
            # pylint: disable=too-many-branches
            def format_zeros(str_val):
                """Zeros should be handled separately as they cannot be negative."""
                if float(str_val) == 0:
                    num_decimals = min(self.variables[row].number_of_decimals, 2)
                    str_val = f"{0:.{num_decimals}f}"
                return str_val

            def render_value(value):
                if np.isnan(value):
                    return ""
                if np.isinf(value):
                    return "∞"

                str_val = attribute.str_val(value)
                if attribute.is_continuous and not attribute.is_time:
                    str_val = format_zeros(str_val)

                return str_val

            if column == self.Columns.NAME:
                return attribute.name
            elif column == self.Columns.DISTRIBUTION:
                if isinstance(attribute,
                              (DiscreteVariable, ContinuousVariable)):
                    if row not in self.__distributions_cache:
                        scene = QGraphicsScene(parent=self)
                        histogram = Histogram(
                            data=self.table,
                            variable=attribute,
                            color_attribute=self.target_var,
                            border=(0, 0, 2, 0),
                            bottom_padding=4,
                            border_color='#ccc',
                        )
                        scene.addItem(histogram)
                        self.__distributions_cache[row] = scene
                    return self.__distributions_cache[row]
            elif column == self.Columns.CENTER:
                return render_value(self._center[row])
            elif column == self.Columns.MEDIAN:
                return render_value(self._median[row])
            elif column == self.Columns.DISPERSION:
                if isinstance(attribute, TimeVariable):
                    return format_time_diff(self._min[row], self._max[row])
                elif isinstance(attribute, DiscreteVariable):
                    return "%.3g" % self._dispersion[row]
                else:
                    return render_value(self._dispersion[row])
            elif column == self.Columns.MIN:
                if not isinstance(attribute, DiscreteVariable):
                    return render_value(self._min[row])
            elif column == self.Columns.MAX:
                if not isinstance(attribute, DiscreteVariable):
                    return render_value(self._max[row])
            elif column == self.Columns.MISSING:
                return '%d (%d%%)' % (
                    self._missing[row],
                    100 * self._missing[row] / self.n_instances
                )
            return None

        roles = {Qt.BackgroundRole: background,
                 Qt.TextAlignmentRole: text_alignment,
                 Qt.DecorationRole: decoration,
                 Qt.DisplayRole: display}

        if not index.isValid() or role not in roles:
            return None

        row, column = self.mapToSourceRows(index.row()), index.column()
        # Make sure we're not out of range
        if not 0 <= row <= self.n_attributes:
            return None

        attribute = self.variables[row]
        return roles[role]()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.n_attributes

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.Columns)

    def set_target_var(self, variable):
        self.target_var = variable
        self.__distributions_cache.clear()
        start_idx = self.index(0, self.Columns.DISTRIBUTION)
        end_idx = self.index(self.rowCount(), self.Columns.DISTRIBUTION)
        self.dataChanged.emit(start_idx, end_idx)


class FeatureStatisticsTableView(QTableView):
    HISTOGRAM_ASPECT_RATIO = (7, 3)
    MINIMUM_HISTOGRAM_HEIGHT = 50
    MAXIMUM_HISTOGRAM_HEIGHT = 80

    def __init__(self, model, parent=None, **kwargs):
        super().__init__(
            parent=parent,
            showGrid=False,
            cornerButtonEnabled=False,
            sortingEnabled=True,
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.ExtendedSelection,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            verticalScrollMode=QTableView.ScrollPerPixel,
            **kwargs
        )
        self.setModel(model)

        hheader = self.horizontalHeader()
        hheader.setStretchLastSection(False)
        # Contents precision specifies how many rows should be taken into
        # account when computing the sizes, 0 being the visible rows. This is
        # crucial, since otherwise the `ResizeToContents` section resize mode
        # would call `sizeHint` on every single row in the data before first
        # render. However this, this cannot be used here, since this only
        # appears to work properly when the widget is actually shown. When the
        # widget is not shown, size `sizeHint` is called on every row.
        hheader.setResizeContentsPrecision(5)
        # Set a nice default size so that headers have some space around titles
        hheader.setDefaultSectionSize(100)
        # Set individual column behaviour in `set_data` since the logical
        # indices must be valid in the model, which requires data.
        hheader.setSectionResizeMode(QHeaderView.Interactive)

        columns = model.Columns
        hheader.setSectionResizeMode(columns.ICON.index, QHeaderView.ResizeToContents)
        hheader.setSectionResizeMode(columns.DISTRIBUTION.index, QHeaderView.Stretch)

        vheader = self.verticalHeader()
        vheader.setVisible(False)
        vheader.setSectionResizeMode(QHeaderView.Fixed)
        hheader.sectionResized.connect(self.bind_histogram_aspect_ratio)
        # TODO: This shifts the scrollarea a bit down when opening widget
        # hheader.sectionResized.connect(self.keep_row_centered)

        self.setItemDelegate(NoFocusRectDelegate(parent=self))
        self.setItemDelegateForColumn(
            FeatureStatisticsTableModel.Columns.DISTRIBUTION,
            DistributionDelegate(parent=self),
        )

    def bind_histogram_aspect_ratio(self, logical_index, _, new_size):
        """Force the horizontal and vertical header to maintain the defined
        aspect ratio specified for the histogram."""
        # Prevent function being exectued more than once per resize
        if logical_index is not self.model().Columns.DISTRIBUTION.index:
            return
        ratio_width, ratio_height = self.HISTOGRAM_ASPECT_RATIO
        unit_width = new_size // ratio_width
        new_height = unit_width * ratio_height
        effective_height = max(new_height, self.MINIMUM_HISTOGRAM_HEIGHT)
        effective_height = min(effective_height, self.MAXIMUM_HISTOGRAM_HEIGHT)
        self.verticalHeader().setDefaultSectionSize(effective_height)

    def keep_row_centered(self, logical_index, _1, _2):
        """When resizing the widget when scrolled further down, the
        positions of rows changes. Obviously, the user resized in order to
        better see the row of interest. This keeps that row centered."""
        # TODO: This does not work properly
        # Prevent function being exectued more than once per resize
        if logical_index is not self.model().Columns.DISTRIBUTION.index:
            return
        top_row = self.indexAt(self.rect().topLeft()).row()
        bottom_row = self.indexAt(self.rect().bottomLeft()).row()
        middle_row = top_row + (bottom_row - top_row) // 2
        self.scrollTo(self.model().index(middle_row, 0), QTableView.PositionAtCenter)


class NoFocusRectDelegate(QStyledItemDelegate):
    """Removes the light blue background and border on a focused item."""

    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        option.state &= ~QStyle.State_HasFocus
        super().paint(painter, option, index)


class DistributionDelegate(NoFocusRectDelegate):
    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        scene = index.data(Qt.DisplayRole)  # type: Optional[QGraphicsScene]
        if scene is None:
            return super().paint(painter, option, index)

        painter.setRenderHint(QPainter.Antialiasing)
        scene.render(painter, target=QRectF(option.rect), mode=Qt.IgnoreAspectRatio)

        # pylint complains about inconsistent return statements
        return None


class OWFeatureStatistics(widget.OWWidget):
    name = 'Feature Statistics'
    description = 'Show basic statistics for data features.'
    icon = 'icons/FeatureStatistics.svg'

    class Inputs:
        data = Input('Data', Table, default=True)

    class Outputs:
        reduced_data = Output('Reduced Data', Table, default=True)
        statistics = Output('Statistics', Table)

    want_main_area = False

    settingsHandler = DomainContextHandler()
    settings_version = 2

    auto_commit = Setting(True)
    color_var = ContextSetting(None)  # type: Optional[Variable]
    # filter_string = ContextSetting('')

    sorting = Setting((0, Qt.AscendingOrder))
    selected_vars = ContextSetting([], schema_only=True)

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Table]

        # Main area
        self.model = FeatureStatisticsTableModel(parent=self)
        self.table_view = FeatureStatisticsTableView(self.model, parent=self)
        self.table_view.selectionModel().selectionChanged.connect(self.on_select)
        self.table_view.horizontalHeader().sectionClicked.connect(self.on_header_click)

        self.controlArea.layout().addWidget(self.table_view)

        self.color_var_model = DomainModel(
            valid_types=(ContinuousVariable, DiscreteVariable),
            placeholder='None',
        )
        self.cb_color_var = gui.comboBox(
            self.buttonsArea, master=self, value='color_var', model=self.color_var_model,
            label='Color:', orientation=Qt.Horizontal, contentsLength=13,
            searchable=True
        )
        self.cb_color_var.activated.connect(self.__color_var_changed)

        gui.rubber(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    @staticmethod
    def sizeHint():
        return QSize(1050, 500)

    @Inputs.data
    def set_data(self, data):
        # Clear outputs and reset widget state
        self.closeContext()
        self.selected_vars = []
        self.model.resetSorting()
        self.Outputs.reduced_data.send(None)
        self.Outputs.statistics.send(None)

        # Setup widget state for new data and restore settings
        self.data = data

        if data is not None:
            self.color_var_model.set_domain(data.domain)
            self.color_var = None
            if self.data.domain.class_vars:
                self.color_var = self.data.domain.class_vars[0]
        else:
            self.color_var_model.set_domain(None)
            self.color_var = None
        self.model.set_data(data)

        self.openContext(self.data)
        self.__restore_selection()
        self.__restore_sorting()
        self.__color_var_changed()

        self.commit()

    def __restore_selection(self):
        """Restore the selection on the table view from saved settings."""
        selection_model = self.table_view.selectionModel()
        selection = QItemSelection()
        if self.selected_vars:
            var_indices = {var: i for i, var in enumerate(self.model.variables)}
            selected_indices = [var_indices[var] for var in self.selected_vars]
            for row in self.model.mapFromSourceRows(selected_indices):
                selection.append(QItemSelectionRange(
                    self.model.index(row, 0),
                    self.model.index(row, self.model.columnCount() - 1)
                ))
        selection_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def __restore_sorting(self):
        """Restore the sort column and order from saved settings."""
        sort_column, sort_order = self.sorting
        if self.model.n_attributes and sort_column < self.model.columnCount():
            self.model.sort(sort_column, sort_order)
            self.table_view.horizontalHeader().setSortIndicator(sort_column, sort_order)

    @pyqtSlot(int)
    def on_header_click(self, *_):
        # Store the header states
        sort_order = self.model.sortOrder()
        sort_column = self.model.sortColumn()
        self.sorting = sort_column, sort_order

    @pyqtSlot(int)
    def __color_var_changed(self, *_):
        if self.model is not None:
            self.model.set_target_var(self.color_var)

    def on_select(self):
        selection_indices = list(self.model.mapToSourceRows([
            i.row() for i in self.table_view.selectionModel().selectedRows()
        ]))
        self.selected_vars = list(self.model.variables[selection_indices])
        self.commit()

    def commit(self):
        if not self.selected_vars:
            self.Outputs.reduced_data.send(None)
            self.Outputs.statistics.send(None)
            return

        # Send a table with only selected columns to output
        variables = self.selected_vars
        self.Outputs.reduced_data.send(self.data[:, variables])

        # Send the statistics of the selected variables to ouput
        labels, data = self.model.get_statistics_matrix(variables, return_labels=True)
        var_names = np.atleast_2d([var.name for var in variables]).T
        domain = Domain(
            attributes=[ContinuousVariable(name) for name in labels],
            metas=[StringVariable('Feature')]
        )
        statistics = Table(domain, data, metas=var_names)
        statistics.name = '%s (Feature Statistics)' % self.data.name
        self.Outputs.statistics.send(statistics)

    def send_report(self):
        view = self.table_view
        self.report_table(view)

    @classmethod
    def migrate_context(cls, context, version):
        if not version or version < 2:
            selected_rows = context.values.pop("selected_rows", None)
            if not selected_rows:
                selected_vars = []
            else:
                # This assumes that dict was saved by Python >= 3.6 so dict is
                # ordered; if not, context hasn't had worked anyway.
                all_vars = [
                    (var, tpe)
                    for (var, tpe) in chain(context.attributes.items(),
                                            context.metas.items())
                    # it would be nicer to use cls.HIDDEN_VAR_TYPES, but there
                    # is no suitable conversion function, and StringVariable (3)
                    # was the only hidden var when settings_version < 2, so:
                    if tpe != 3]
                selected_vars = [all_vars[i] for i in selected_rows]
            context.values["selected_vars"] = selected_vars, -3


if __name__ == '__main__':  # pragma: no cover
    WidgetPreview(OWFeatureStatistics).run(Table("iris"))
