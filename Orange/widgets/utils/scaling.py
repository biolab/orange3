from datetime import time
import random
import sys

import numpy as np

import Orange
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
        self.valid_data_array = ~np.isnan(no_jit)
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

    # @deprecated_keywords({"attrIndices": "attr_indices",
    #                       "settingsDict": "settings_dict"})
    def get_projected_point_position(self, attr_indices, values, **settings_dict):
        """
        For attributes in attr_indices and values of these attributes in values
        compute point positions this function has more sense in radviz and
        polyviz methods. settings_dict has to be because radviz and polyviz have
        this parameter.
        """
        return values

    getProjectedPointPosition = get_projected_point_position

    # @deprecated_keywords({"attrIndices": "attr_indices",
    #                       "settingsDict": "settings_dict"})
    def create_projection_as_example_table(self, attr_indices, **settings_dict):
        """
        Create the projection of attribute indices given in attr_indices and
        create an example table with it.

        """
        if self.data_has_class:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous(self.data_domain[attr_indices[0]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[1]].name),
                                         Orange.feature.Discrete(self.data_domain.class_var.name,
                                                                       values = get_variable_values_sorted(self.data_domain.class_var))])
        else:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous(self.data_domain[attr_indices[0]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[1]].name)])

        data = self.create_projection_as_numeric_array(attr_indices,
                                                       **settings_dict)
        if data != None:
            return Orange.data.Table(domain, data)
        else:
            return Orange.data.Table(domain)

    createProjectionAsExampleTable = create_projection_as_example_table

    # @deprecated_keywords({"attrIndices": "attr_indices",
    #                       "settingsDict": "settings_dict"})
    def create_projection_as_example_table_3D(self, attr_indices, **settings_dict):
        """
        Create the projection of attribute indices given in attr_indices and
        create an example table with it.

        """
        if self.data_has_class:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous(self.data_domain[attr_indices[0]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[1]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[2]].name),
                                         Orange.feature.Discrete(self.data_domain.class_var.name,
                                                                       values = get_variable_values_sorted(self.data_domain.class_var))])
        else:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous(self.data_domain[attr_indices[0]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[1]].name),
                                         Orange.feature.Continuous(self.data_domain[attr_indices[2]].name)])

        data = self.create_projection_as_numeric_array_3D(attr_indices,
                                                          **settings_dict)
        if data != None:
            return Orange.data.Table(domain, data)
        else:
            return Orange.data.Table(domain)

    createProjectionAsExampleTable3D = create_projection_as_example_table_3D

    # @deprecated_keywords({"attrIndices": "attr_indices",
    #                       "settingsDict": "settings_dict",
    #                       "validData": "valid_data",
    #                       "classList": "class_list",
    #                       "jutterSize": "jitter_size"})
    def create_projection_as_numeric_array(self, attr_indices, **settings_dict):
        valid_data = settings_dict.get("valid_data")
        class_list = settings_dict.get("class_list")
        jitter_size = settings_dict.get("jitter_size", 0.0)

        if valid_data == None:
            valid_data = self.get_valid_list(attr_indices)
        if sum(valid_data) == 0:
            return None

        if class_list == None and self.data_has_class:
            class_list = self.original_data[self.data_class_index]

        xarray = self.no_jittering_scaled_data[attr_indices[0]]
        yarray = self.no_jittering_scaled_data[attr_indices[1]]
        if jitter_size > 0.0:
            xarray += (np.random.random(len(xarray))-0.5)*jitter_size
            yarray += (np.random.random(len(yarray))-0.5)*jitter_size
        if class_list != None:
            data = np.compress(valid_data, np.array((xarray, yarray, class_list)), axis = 1)
        else:
            data = np.compress(valid_data, np.array((xarray, yarray)), axis = 1)
        data = np.transpose(data)
        return data

    createProjectionAsNumericArray = create_projection_as_numeric_array

    # @deprecated_keywords({"attrIndices": "attr_indices",
    #                       "settingsDict": "settings_dict",
    #                       "validData": "valid_data",
    #                       "classList": "class_list",
    #                       "jutterSize": "jitter_size"})
    def create_projection_as_numeric_array_3D(self, attr_indices, **settings_dict):
        valid_data = settings_dict.get("valid_data")
        class_list = settings_dict.get("class_list")
        jitter_size = settings_dict.get("jitter_size", 0.0)

        if valid_data == None:
            valid_data = self.get_valid_list(attr_indices)
        if sum(valid_data) == 0:
            return None

        if class_list == None and self.data_has_class:
            class_list = self.original_data[self.data_class_index]

        xarray = self.no_jittering_scaled_data[attr_indices[0]]
        yarray = self.no_jittering_scaled_data[attr_indices[1]]
        zarray = self.no_jittering_scaled_data[attr_indices[2]]
        if jitter_size > 0.0:
            xarray += (np.random.random(len(xarray))-0.5)*jitter_size
            yarray += (np.random.random(len(yarray))-0.5)*jitter_size
            zarray += (np.random.random(len(zarray))-0.5)*jitter_size
        if class_list != None:
            data = np.compress(valid_data, np.array((xarray, yarray, zarray, class_list)), axis = 1)
        else:
            data = np.compress(valid_data, np.array((xarray, yarray, zarray)), axis = 1)
        data = np.transpose(data)
        return data

    createProjectionAsNumericArray3D = create_projection_as_numeric_array_3D

    # @deprecated_keywords({"attributeNameOrder": "attribute_name_order",
    #                       "addResultFunct": "add_result_funct"})
    def get_optimal_clusters(self, attribute_name_order, add_result_funct):
        if not self.data_has_class or self.data_has_continuous_class:
            return

        jitter_size = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation
        domain = Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                     Orange.feature.Continuous("yVar"),
                                    self.data_domain.class_var])

        # init again, in case that the attribute ordering took too much time
        start_time = time.time()
        test_index = 0

        count = len(attribute_name_order) * (len(attribute_name_order) - 1) / 2
        with self.scatterWidget.progressBar(count) as progressBar:
            for i in range(len(attribute_name_order)):
                for j in range(i):
                    try:
                        index = self.data_domain.index
                        attr1 = index(attribute_name_order[j])
                        attr2 = index(attribute_name_order[i])
                        test_index += 1
                        if self.clusterOptimization.isOptimizationCanceled():
                            secs = time.time() - start_time
                            self.clusterOptimization.setStatusBarText(
                                "Evaluation stopped "
                                "(evaluated %d projections in %d min, %d sec)"
                                % (test_index, secs / 60, secs % 60))
                            return

                        data = self.create_projection_as_example_table(
                            [attr1, attr2],
                            domain=domain, jitter_size=jitter_size)
                        graph, valuedict, closuredict, polygon_vertices_dict, \
                            enlarged_closure_dict, other_dict = \
                            self.clusterOptimization.evaluateClusters(data)

                        all_value = 0.0
                        classes_dict = {}
                        for key in valuedict.keys():
                            cls = int(graph.objects[polygon_vertices_dict
                                                   [key][0]].getclass())
                            add_result_funct(
                                valuedict[key], closuredict[key],
                                polygon_vertices_dict[key],
                                [attribute_name_order[i],
                                 attribute_name_order[j]],
                                cls,
                                enlarged_closure_dict[key], other_dict[key])
                            classes_dict[key] = cls
                            all_value += valuedict[key]
                        # add all the clusters
                        add_result_funct(
                            all_value, closuredict, polygon_vertices_dict,
                            [attribute_name_order[i], attribute_name_order[j]],
                            classes_dict, enlarged_closure_dict, other_dict)

                        self.clusterOptimization.setStatusBarText(
                            "Evaluated %d projections..." % test_index)
                        progressBar.advance()
                        del data, graph, valuedict, closuredict, \
                            polygon_vertices_dict, enlarged_closure_dict, \
                            other_dict, classes_dict
                    except:
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # print the exception

        secs = time.time() - start_time
        self.clusterOptimization.setStatusBarText(
            "Finished evaluation (evaluated %d projections in %d min, %d sec)"
            % (test_index, secs / 60, secs % 60))

    getOptimalClusters = get_optimal_clusters
