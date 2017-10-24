import numpy as np

from Orange.statistics.basic_stats import DomainBasicStats
from Orange.widgets.settings import Setting
from Orange.widgets.utils import checksum
from Orange.widgets.utils.datacaching import getCached, setCached


class ScaleData:
    jitter_size = Setting(10)
    jitter_continuous = Setting(False)

    def _reset_data(self):
        self.domain = None
        self.data = None
        self.original_data = None  # as numpy array
        self.scaled_data = None  # in [0, 1]
        self.jittered_data = None
        self.attr_values = {}
        self.domain_data_stat = []
        self.valid_data_array = None
        self.attribute_flip_info = {}  # dictionary with attr: 0/1 if flipped
        self.jitter_seed = 0

    def __init__(self):
        self._reset_data()

    def rescale_data(self):
        self._compute_jittered_data()

    def _compute_domain_data_stat(self):
        stt = self.domain_data_stat = \
            getCached(self.data, DomainBasicStats, (self.data,))
        for index in range(len(self.domain)):
            attr = self.domain[index]
            if attr.is_discrete:
                self.attr_values[attr] = [0, len(attr.values)]
            elif attr.is_continuous:
                self.attr_values[attr] = [stt[index].min, stt[index].max]

    def _compute_scaled_data(self):
        data = self.data
        # We cache scaled_data and validArray to share them between widgets
        cached = getCached(data, "visualizationData")
        if cached:
            self.original_data, self.scaled_data, self.valid_data_array = cached
            return

        Y = data.Y if data.Y.ndim == 2 else np.atleast_2d(data.Y).T
        self.original_data = np.hstack((data.X, Y)).T
        self.scaled_data = no_jit = self.original_data.copy()
        self.valid_data_array = np.isfinite(no_jit)
        for index in range(len(data.domain)):
            attr = data.domain[index]
            if attr.is_discrete:
                no_jit[index] *= 2
                no_jit[index] += 1
                no_jit[index] /= 2 * len(attr.values)
            else:
                dstat = self.domain_data_stat[index]
                no_jit[index] -= dstat.min
                if dstat.max != dstat.min:
                    no_jit[index] /= dstat.max - dstat.min
        setCached(data, "visualizationData",
                  (self.original_data, self.scaled_data, self.valid_data_array))

    def _compute_jittered_data(self):
        data = self.data
        self.jittered_data = self.scaled_data.copy()
        random = np.random.RandomState(seed=self.jitter_seed)
        for index, col in enumerate(self.jittered_data):
            # Need to use a different seed for each feature
            attr = data.domain[index]
            if attr.is_discrete:
                off = self.jitter_size / (25 * max(1, len(attr.values)))
            elif attr.is_continuous and self.jitter_continuous:
                off = self.jitter_size / 25
            else:
                continue
            col += random.uniform(-off, off, len(data))
            # fix values outside [0, 1]
            col = np.absolute(col)

            above_1 = col > 1
            col[above_1] = 2 - col[above_1]

    # noinspection PyAttributeOutsideInit
    def set_data(self, data, skip_if_same=False, no_data=False):
        if skip_if_same and checksum(data) == checksum(self.data):
            return
        self._reset_data()
        if data is None:
            return

        self.domain = data.domain
        self.data = data
        self.attribute_flip_info = {}
        if not no_data:
            self._compute_domain_data_stat()
            self._compute_scaled_data()
            self._compute_jittered_data()

    def flip_attribute(self, attr):
        if attr.is_discrete:
            return 0
        index = self.domain.index(attr)
        self.attribute_flip_info[attr] = 1 - self.attribute_flip_info.get(attr, 0)
        if attr.is_continuous:
            self.attr_values[attr] = [-self.attr_values[attr][1],
                                      -self.attr_values[attr][0]]

        self.jittered_data[index] = 1 - self.jittered_data[index]
        self.scaled_data[index] = 1 - self.scaled_data[index]
        return 1

    def get_valid_list(self, indices):
        """
        Get array of 0 and 1 of len = len(self.data). If there is a missing
        value at any attribute in indices return 0 for that instance.
        """
        if self.valid_data_array is None or len(self.valid_data_array) == 0:
            return np.array([], np.bool)
        return np.all(self.valid_data_array[indices], axis=0)

    def get_valid_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_list(indices)
        return np.nonzero(valid_list)[0]


class ScaleScatterPlotData(ScaleData):
    def get_xy_data_positions(self, xattr, yattr, filter_valid=False,
                              copy=True):
        """
        Create x-y projection of attributes in attrlist.

        """
        xattr_index = self.domain.index(xattr)
        yattr_index = self.domain.index(yattr)
        if filter_valid is True:
            filter_valid = self.get_valid_list([xattr_index, yattr_index])
        if isinstance(filter_valid, np.ndarray):
            xdata = self.jittered_data[xattr_index, filter_valid]
            ydata = self.jittered_data[yattr_index, filter_valid]
        elif copy:
            xdata = self.jittered_data[xattr_index].copy()
            ydata = self.jittered_data[yattr_index].copy()
        else:
            xdata = self.jittered_data[xattr_index]
            ydata = self.jittered_data[yattr_index]

        if self.domain[xattr_index].is_discrete:
            xdata *= len(self.domain[xattr_index].values)
            xdata -= 0.5
        else:
            xdata *= self.attr_values[xattr][1] - self.attr_values[xattr][0]
            xdata += float(self.attr_values[xattr][0])
        if self.domain[yattr_index].is_discrete:
            ydata *= len(self.domain[yattr_index].values)
            ydata -= 0.5
        else:
            ydata *= self.attr_values[yattr][1] - self.attr_values[yattr][0]
            ydata += float(self.attr_values[yattr][0])
        return xdata, ydata

    getXYDataPositions = get_xy_data_positions
