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

    def __init__(self):
        self.raw_data = None           # input data
        self.attribute_names = []    # list of attribute names from self.raw_data
        self.attribute_flip_info = {}   # dictionary with attrName: 0/1 attribute is flipped or not

        self.data_has_class = False
        self.data_has_continuous_class = False
        self.data_has_discrete_class = False
        self.data_class_name = None
        self.data_domain = None
        self.data_class_index = None
        self.have_data = False

        self.jitter_seed = 0

        self.attr_values = {}
        self.domain_data_stat = []
        self.original_data = None    # input (nonscaled) data in a numpy array
        self.scaled_data = None        # scaled data to the interval 0-1
        self.no_jittering_scaled_data = None
        self.valid_data_array = None

    def rescale_data(self):
        """
        Force the existing data to be rescaled due to changes like
        jitter_continuous, jitter_size, ...
        """
        self.set_data(self.raw_data, skipIfSame=0)

    def set_data(self, data, **args):
        if args.get("skipIfSame", 1):
            if checksum(data) == checksum(self.raw_data):
                return

        self.domain_data_stat = []
        self.attr_values = {}
        self.original_data = None
        self.scaled_data = None
        self.no_jittering_scaled_data = None
        self.valid_data_array = None

        self.raw_data = None
        self.have_data = False
        self.data_has_class = False
        self.data_has_continuous_class = False
        self.data_has_discrete_class = False
        self.data_class_name = None
        self.data_domain = None
        self.data_class_index = None

        if data is None:
            return
        full_data = data
        self.raw_data = data

        len_data = data and len(data) or 0

        self.attribute_names = [attr.name for attr in full_data.domain]
        self.attribute_flip_info = {}

        self.data_domain = full_data.domain
        self.data_has_class = bool(full_data.domain.class_var)
        self.data_has_continuous_class = full_data.domain.has_continuous_class
        self.data_has_discrete_class = full_data.domain.has_discrete_class

        self.data_class_name = self.data_has_class and full_data.domain.class_var.name
        if self.data_has_class:
            self.data_class_index = self.data_domain.index(self.data_class_name)
        self.have_data = bool(self.raw_data and len(self.raw_data) > 0)

        self.domain_data_stat = getCached(full_data,
                                          DomainBasicStats,
                                          (full_data,))

        sort_values_for_discrete_attrs = args.get("sort_values_for_discrete_attrs",
                                                  1)

        for index in range(len(full_data.domain)):
            attr = full_data.domain[index]
            if attr.is_discrete:
                self.attr_values[attr.name] = [0, len(attr.values)]
            elif attr.is_continuous:
                self.attr_values[attr.name] = [self.domain_data_stat[index].min,
                                               self.domain_data_stat[index].max]

        if 'no_data' in args:
            return

        # the original_data, no_jittering_scaled_data and validArray are arrays
        # that we can cache so that other visualization widgets don't need to
        # compute it. The scaled_data on the other hand has to be computed for
        # each widget separately because of different
        # jitter_continuous and jitter_size values
        if getCached(data, "visualizationData"):
            self.original_data, self.no_jittering_scaled_data, self.valid_data_array = getCached(data,
                                                                                                 "visualizationData")
        else:
            no_jittering_data = np.c_[full_data.X, full_data.Y].T
            valid_data_array = ~np.isnan(no_jittering_data)
            original_data = no_jittering_data.copy()

            for index in range(len(data.domain)):
                attr = data.domain[index]
                if attr.is_discrete:
                    # see if the values for discrete attributes have to be resorted
                    variable_value_indices = get_variable_value_indices(data.domain[index],
                                                                        sort_values_for_discrete_attrs)
                    if 0 in [i == variable_value_indices[attr.values[i]]
                             for i in range(len(attr.values))]:
                        # make the array a contiguous, otherwise the putmask
                        # function does not work
                        line = no_jittering_data[index].copy()
                        indices = [np.where(line == val, 1, 0)
                                   for val in range(len(attr.values))]
                        for i in range(len(attr.values)):
                            np.putmask(line, indices[i],
                                          variable_value_indices[attr.values[i]])
                        no_jittering_data[index] = line   # save the changed array
                        original_data[index] = line     # reorder also the values in the original data
                    no_jittering_data[index] = ((no_jittering_data[index] * 2.0 + 1.0)
                                                / float(2 * len(attr.values)))

                elif attr.is_continuous:
                    diff = self.domain_data_stat[index].max - self.domain_data_stat[
                        index].min or 1     # if all values are the same then prevent division by zero
                    no_jittering_data[index] = (no_jittering_data[index] -
                                                self.domain_data_stat[index].min) / diff

            self.original_data = original_data
            self.no_jittering_scaled_data = no_jittering_data
            self.valid_data_array = valid_data_array

        if data:
            setCached(data, "visualizationData",
                      (self.original_data, self.no_jittering_scaled_data,
                       self.valid_data_array))

        # compute the scaled_data arrays
        scaled_data = self.no_jittering_scaled_data

        # Random generators for jittering
        random = np.random.RandomState(seed=self.jitter_seed)
        rand_seeds = random.random_integers(0, 2 ** 30 - 1,
                                            size=len(data.domain))
        for index, rseed in zip(list(range(len(data.domain))), rand_seeds):
            # Need to use a different seed for each feature
            random = np.random.RandomState(seed=rseed)
            attr = data.domain[index]
            if attr.is_discrete:
                scaled_data[index] += (self.jitter_size / (50.0 * max(1, len(attr.values)))) * \
                                      (random.rand(len(full_data)) - 0.5)

            elif attr.is_continuous and self.jitter_continuous:
                scaled_data[index] += self.jitter_size / 50.0 * (0.5 - random.rand(len(full_data)))
                scaled_data[index] = np.absolute(scaled_data[index])       # fix values below zero
                ind = np.where(scaled_data[index] > 1.0, 1, 0)     # fix values above 1
                np.putmask(scaled_data[index], ind, 2.0 - np.compress(ind, scaled_data[index]))

        self.scaled_data = scaled_data[:, :len_data]

    def scale_example_value(self, instance, index):
        """
        Scale instance's value at index index to a range between 0 and 1 with
        respect to self.raw_data.
        """
        if instance[index].isSpecial():
            print("Warning: scaling instance with missing value")
            return 0.5
        if instance.domain[index].is_discrete:
            d = get_variable_value_indices(instance.domain[index])
            return (d[instance[index].value] * 2 + 1) / float(2 * len(d))
        elif instance.domain[index].is_continuous:
            diff = self.domain_data_stat[index].max - self.domain_data_stat[index].min
            if diff == 0:
                diff = 1          # if all values are the same then prevent division by zero
            return (instance[index] - self.domain_data_stat[index].min) / diff

    def get_attribute_label(self, attr_name):
        if (self.attribute_flip_info.get(attr_name, 0) and
            self.data_domain[attr_name].is_continuous):
            return "-" + attr_name
        return attr_name

    def flip_attribute(self, attr_name):
        if attr_name not in self.attribute_names:
            return 0
        if self.data_domain[attr_name].is_discrete:
            return 0

        index = self.data_domain.index(attr_name)
        self.attribute_flip_info[attr_name] = 1 - self.attribute_flip_info.get(attr_name, 0)
        if self.data_domain[attr_name].is_continuous:
            self.attr_values[attr_name] = [-self.attr_values[attr_name][1], -self.attr_values[attr_name][0]]

        self.scaled_data[index] = 1 - self.scaled_data[index]
        self.no_jittering_scaled_data[index] = 1 - self.no_jittering_scaled_data[index]
        return 1

    def get_min_max_val(self, attr):
        if type(attr) == int:
            attr = self.attribute_names[attr]
        diff = self.attr_values[attr][1] - self.attr_values[attr][0]
        return diff or 1.0

    def get_valid_list(self, indices, also_class_if_exists=1):
        """
        Get array of 0 and 1 of len = len(self.raw_data). If there is a missing
        value at any attribute in indices return 0 for that instance.
        """
        if self.valid_data_array is None or len(self.valid_data_array) == 0:
            return np.array([], np.bool)

        inds = indices[:]
        if also_class_if_exists and self.data_has_class:
            inds.append(self.data_class_index)
        return np.all(self.valid_data_array[inds], axis=0)

    def get_valid_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_list(indices)
        return np.nonzero(valid_list)[0]

    def rnd_correction(self, max):
        """
        Return a number from -max to max.
        """
        return (random.random() - 0.5) * 2 * max


class ScaleScatterPlotData(ScaleData):
    def get_original_data(self, indices):
        data = self.original_data.take(indices, axis = 0)
        for i, ind in enumerate(indices):
            [minVal, maxVal] = self.attr_values[self.data_domain[ind].name]
            if self.data_domain[ind].is_discrete:
                data[i] += (self.jitter_size/50.0)*(np.random.random(len(self.raw_data)) - 0.5)
            elif self.data_domain[ind].is_continuous and self.jitter_continuous:
                data[i] += (self.jitter_size/(50.0*(maxVal-minVal or 1)))*(np.random.random(len(self.raw_data)) - 0.5)
        return data

    getOriginalData = get_original_data

    # @deprecated_keywords({"xAttr": "xattr", "yAttr": "yattr"})
    def get_xy_data_positions(self, xattr, yattr, filter_valid=False,
                              copy=True):
        """
        Create x-y projection of attributes in attrlist.

        """
        xattr_index = self.data_domain.index(xattr)
        yattr_index = self.data_domain.index(yattr)
        if filter_valid is True:
            filter_valid = self.get_valid_list([xattr_index, yattr_index])
        if isinstance(filter_valid, np.ndarray):
            xdata = self.scaled_data[xattr_index, filter_valid]
            ydata = self.scaled_data[yattr_index, filter_valid]
        elif copy:
            xdata = self.scaled_data[xattr_index].copy()
            ydata = self.scaled_data[yattr_index].copy()
        else:
            xdata = self.scaled_data[xattr_index]
            ydata = self.scaled_data[yattr_index]

        if self.data_domain[xattr_index].is_discrete:
            xdata *= len(self.data_domain[xattr_index].values)
            xdata -= 0.5
        else:
            xdata *= self.attr_values[xattr][1] - self.attr_values[xattr][0]
            xdata += float(self.attr_values[xattr][0])
        if self.data_domain[yattr_index].is_discrete:
            ydata *= len(self.data_domain[yattr_index].values)
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
