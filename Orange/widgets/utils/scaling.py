from itertools import chain

import numpy as np

from Orange.data import Domain
from Orange.statistics.basic_stats import DomainBasicStats
from Orange.widgets.settings import Setting
from Orange.widgets.utils.datacaching import getCached, setCached
from Orange.widgets.utils.scaling import ScaleScatterPlotData


class ScaleData:
    jitter_size = Setting(10)
    jitter_continuous = Setting(False)

    def _reset_data(self):
        self.domain = None
        self.data = None  # as Orange Table
        self.scaled_data = None  # in [0, 1]
        self.jittered_data = None
        self.attr_values = {}
        self.domain_data_stat = {}
        self.valid_data_array = None
        self.attribute_flip_info = {}  # dictionary with attr: 0/1 if flipped
        self.jitter_seed = 0

    def __init__(self):
        self._reset_data()

    def rescale_data(self):
        self._compute_jittered_data()

    def _compute_domain_data_stat(self):
        stt = self.domain_data_stat = \
            getCached(self.data, DomainBasicStats, (self.data, True))
        domain = self.domain
        for attr in chain(domain.variables, domain.metas):
            if attr.is_discrete:
                self.attr_values[attr] = [0, len(attr.values)]
            elif attr.is_continuous:
                self.attr_values[attr] = [stt[attr].min, stt[attr].max]

    def _compute_scaled_data(self):
        data = self.data
        # We cache scaled_data and validArray to share them between widgets
        cached = getCached(data, "visualizationData")
        if cached:
            self.data, self.scaled_data, self.valid_data_array = cached
            return

        Y = data.Y if data.Y.ndim == 2 else np.atleast_2d(data.Y).T
        all_data = np.hstack((data.X, Y, data.metas)).T
        self.scaled_data = self.data.copy()
        self.valid_data_array = np.isfinite(all_data)
        domain = self.domain
        for attr in chain(domain.attributes, domain.class_vars, domain.metas):
            c = self.scaled_data.get_column_view(attr)[0]
            if attr.is_discrete:
                c += 0.5
                c /= len(attr.values)
            else:
                dstat = self.domain_data_stat[attr]
                c -= dstat.min
                if dstat.max != dstat.min:
                    c /= dstat.max - dstat.min
        setCached(data, "visualizationData",
                  (self.data, self.scaled_data, self.valid_data_array))

    def _compute_jittered_data(self):
        data = self.data
        self.jittered_data = self.scaled_data.copy()
        random = np.random.RandomState(seed=self.jitter_seed)
        domain = self.domain
        for attr in chain(domain.variables, domain.metas):
            # Need to use a different seed for each feature
            if attr.is_discrete:
                off = self.jitter_size / (25 * max(1, len(attr.values)))
            elif attr.is_continuous and self.jitter_continuous:
                off = self.jitter_size / 25
            else:
                continue
            col = self.jittered_data.get_column_view(attr)[0]
            col += random.uniform(-off, off, len(data))
            # fix values outside [0, 1]
            col = np.absolute(col)

            above_1 = col > 1
            col[above_1] = 2 - col[above_1]

    # noinspection PyAttributeOutsideInit
    def set_data(self, data, *, no_data=False):
        self._reset_data()
        if data is None:
            return

        domain = data.domain
        new_domain = Domain(attributes=domain.attributes,
                            class_vars=domain.class_vars,
                            metas=tuple(v for v in domain.metas if v.is_primitive()))
        self.data = data.transform(new_domain)
        self.data.metas = self.data.metas.astype(float)
        self.domain = self.data.domain
        self.attribute_flip_info = {}
        if not no_data:
            self._compute_domain_data_stat()
            self._compute_scaled_data()
            self._compute_jittered_data()

    def flip_attribute(self, attr):
        if attr.is_discrete:
            return 0
        self.attribute_flip_info[attr] = 1 - self.attribute_flip_info.get(attr, 0)
        if attr.is_continuous:
            self.attr_values[attr] = [-self.attr_values[attr][1],
                                      -self.attr_values[attr][0]]
        col = self.jittered_data.get_column_view(attr)[0]
        col *= -1
        col += 1
        col = self.scaled_data.get_column_view(attr)[0]
        col *= -1
        col += 1
        return 1

    def get_valid_list(self, attrs):
        """
        Get array of 0 and 1 of len = len(self.data). If there is a missing
        value at any attribute in indices return 0 for that instance.
        """
        if self.valid_data_array is None or len(self.valid_data_array) == 0:
            return np.array([], np.bool)
        domain = self.domain
        indices = []
        for index, attr in enumerate(chain(domain.variables, domain.metas)):
            if attr in attrs:
                indices.append(index)
        return np.all(self.valid_data_array[indices], axis=0)

    def get_valid_indices(self, attrs):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_list(attrs)
        return np.nonzero(valid_list)[0]


class ScaleScatterPlotData(ScaleData):
    def get_xy_data_positions(self, attr_x, attr_y, filter_valid=False,
                              copy=True):
        """
        Create x-y projection of attributes in attrlist.

        """
        jit = self.jittered_data
        if filter_valid is True:
            filter_valid = self.get_valid_list([attr_x, attr_y])
        if isinstance(filter_valid, np.ndarray):
            data_x = jit.get_column_view(attr_x)[0][filter_valid]
            data_y = jit.get_column_view(attr_y)[0][filter_valid]
        elif copy:
            data_x = jit.get_column_view(attr_x)[0].copy()
            data_y = jit.get_column_view(attr_y)[0].copy()
        else:
            data_x = jit.get_column_view(attr_x)[0]
            data_y = jit.get_column_view(attr_y)[0]

        if attr_x.is_discrete:
            data_x *= len(attr_x.values)
            data_x -= 0.5
        else:
            data_x *= self.attr_values[attr_x][1] - self.attr_values[attr_x][0]
            data_x += float(self.attr_values[attr_x][0])
        if attr_y.is_discrete:
            data_y *= len(attr_y.values)
            data_y -= 0.5
        else:
            data_y *= self.attr_values[attr_y][1] - self.attr_values[attr_y][0]
            data_y += float(self.attr_values[attr_y][0])
        return data_x, data_y

    getXYDataPositions = get_xy_data_positions
