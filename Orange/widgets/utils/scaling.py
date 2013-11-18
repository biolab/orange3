import sys
import random
import numpy as np

from Orange.data import Table
from Orange.statistics.basic_stats import DomainBasicStats
from Orange.widgets.settings import Setting
from Orange.widgets.tests.test_settings import VarTypes
from Orange.widgets.utils.datacaching import getCached, setCached


def checksum(x):
    if x is None:
        return None
    return x.checksum()


def get_variable_values_sorted(variable):
    """
    Return a list of sorted values for given attribute.

    EXPLANATION: if variable values have values 1, 2, 3, 4, ... then their
    order in orange depends on when they appear first in the data. With this
    function we get a sorted list of values.
    """
    if variable.var_type == VarTypes.Continuous:
        print("get_variable_values_sorted - attribute %s is a continuous variable" % variable)
        return []

    values = list(variable.values)

    # do all attribute values contain integers?
    try:
        int_values = [(int(val), val) for val in values]
    except ValueError:
        return values

    # if all values were integers, we first sort them in ascending order
    int_values.sort()
    return [val[1] for val in int_values]


def get_variable_value_indices(variable, sort_values_for_discrete_attrs=1):
    """
    Create a dictionary with given variable. Keys are variable values, values
    are indices (transformed from string to int); in case all values are
    integers, we also sort them.
    
    """
    if variable.var_type == VarTypes.Continuous:
        print("get_variable_value_indices - attribute %s is a continuous "
              "variable" % (str(variable)))
        return {}

    if sort_values_for_discrete_attrs:
        values = get_variable_values_sorted(variable)
    else:
        values = list(variable.values)
    return dict([(values[i], i) for i in range(len(values))])


class ScaleData:
    jitter_size = Setting(10)

    def __init__(self):
        self.raw_data = None           # input data
        self.raw_subset_data = None
        self.attribute_names = []    # list of attribute names from self.raw_data
        self.attribute_name_index = {}  # dict with indices to attributes
        self.attribute_flip_info = {}   # dictionary with attrName: 0/1 attribute is flipped or not

        self.data_has_class = False
        self.data_has_continuous_class = False
        self.data_has_discrete_class = False
        self.data_class_name = None
        self.data_domain = None
        self.data_class_index = None
        self.have_data = False
        self.have_subset_data = False

        self.jitter_continuous = 0
        self.jitter_seed = 0

        self.attr_values = {}
        self.domain_data_stat = []
        self.original_data = self.original_subset_data = None    # input (nonscaled) data in a numpy array
        self.scaled_data = self.scaled_subset_data = None        # scaled data to the interval 0-1
        self.no_jittering_scaled_data = self.no_jittering_scaled_subset_data = None
        self.valid_data_array = self.valid_subset_data_array = None

    def merge_data_sets(self, data, subset_data):
        """
        Take examples from data and subset_data and merge them into one
        dataset.
        
        """
        if data is None and subset_data is None:
            return None
        if subset_data is None:
            return data
        elif data is None:
            return subset_data
        else:
            full_data = Table(data)
            full_data.extend(subset_data)
            return full_data

    mergeDataSets = merge_data_sets

    def rescale_data(self):
        """
        Force the existing data to be rescaled due to changes like
        jitter_continuous, jitter_size, ...
        """
        self.set_data(self.raw_data, self.raw_subset_data, skipIfSame=0)

    rescaleData = rescale_data

    def set_data(self, data, subset_data=None, **args):
        if args.get("skipIfSame", 1):
            if checksum(data) == checksum(self.raw_data) and \
               checksum(subset_data) == checksum(self.raw_subset_data):
                return

        self.domain_data_stat = []
        self.attr_values = {}
        self.original_data = self.original_subset_data = None
        self.scaled_data = self.scaled_subset_data = None
        self.no_jittering_scaled_data = self.no_jittering_scaled_subset_data = None
        self.valid_data_array = self.valid_subset_data_array = None

        self.raw_data = None
        self.raw_subset_data = None
        self.have_data = False
        self.have_subset_data = False
        self.data_has_class = False
        self.data_has_continuous_class = False
        self.data_has_discrete_class = False
        self.data_class_name = None
        self.data_domain = None
        self.data_class_index = None

        if data is None:
            return
        full_data = self.merge_data_sets(data, subset_data)

        self.raw_data = data
        self.raw_subset_data = subset_data

        len_data = data and len(data) or 0

        self.attribute_names = [attr.name for attr in full_data.domain]
        self.attribute_name_index = dict([(full_data.domain[i].name, i)
                                          for i in range(len(full_data.domain))])
        self.attribute_flip_info = {}

        self.data_domain = full_data.domain
        self.data_has_class = bool(full_data.domain.class_var)
        self.data_has_continuous_class = bool(self.data_has_class and
                                              full_data.domain.class_var.var_type == VarTypes.Continuous)
        self.data_has_discrete_class = bool(self.data_has_class and
                                            full_data.domain.class_var.var_type == VarTypes.Discrete)
        self.data_class_name = self.data_has_class and full_data.domain.class_var.name
        if self.data_has_class:
            self.data_class_index = self.attribute_name_index[self.data_class_name]
        self.have_data = bool(self.raw_data and len(self.raw_data) > 0)
        self.have_subset_data = bool(self.raw_subset_data and
                                     len(self.raw_subset_data) > 0)

        self.domain_data_stat = getCached(full_data,
                                          DomainBasicStats,
                                          (full_data,))

        sort_values_for_discrete_attrs = args.get("sort_values_for_discrete_attrs",
                                                  1)

        for index in range(len(full_data.domain)):
            attr = full_data.domain[index]
            if attr.var_type == VarTypes.Discrete:
                self.attr_values[attr.name] = [0, len(attr.values)]
            elif attr.var_type == VarTypes.Continuous:
                self.attr_values[attr.name] = [self.domain_data_stat[index].min,
                                               self.domain_data_stat[index].max]

        # the original_data, no_jittering_scaled_data and validArray are arrays
        # that we can cache so that other visualization widgets don't need to
        # compute it. The scaled_data on the other hand has to be computed for
        # each widget separately because of different
        # jitter_continuous and jitter_size values
        if getCached(data, "visualizationData") and subset_data == None:
            self.original_data, self.no_jittering_scaled_data, self.valid_data_array = getCached(data,
                                                                                                 "visualizationData")
            self.original_subset_data = self.no_jittering_scaled_subset_data = self.valid_subset_data_array = np.array(
                []).reshape([len(self.original_data), 0])
        else:
            no_jittering_data = np.hstack((full_data.X, full_data.Y)).T
            valid_data_array = no_jittering_data != np.NaN
            original_data = no_jittering_data.copy()

            for index in range(len(data.domain)):
                attr = data.domain[index]
                if attr.var_type == VarTypes.Discrete:
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

                elif attr.var_type == VarTypes.Continuous:
                    diff = self.domain_data_stat[index].max - self.domain_data_stat[
                        index].min or 1     # if all values are the same then prevent division by zero
                    no_jittering_data[index] = (no_jittering_data[index] -
                                                self.domain_data_stat[index].min) / diff

            self.original_data = original_data[:, :len_data]
            self.original_subset_data = original_data[:, len_data:]
            self.no_jittering_scaled_data = no_jittering_data[:, :len_data]
            self.no_jittering_scaled_subset_data = no_jittering_data[:, len_data:]
            self.valid_data_array = valid_data_array[:, :len_data]
            self.valid_subset_data_array = valid_data_array[:, len_data:]

        if data:
            setCached(data, "visualizationData",
                      (self.original_data, self.no_jittering_scaled_data,
                       self.valid_data_array))
        if subset_data:
            setCached(subset_data, "visualizationData",
                      (self.original_subset_data,
                       self.no_jittering_scaled_subset_data,
                       self.valid_subset_data_array))

        # compute the scaled_data arrays
        scaled_data = np.concatenate([self.no_jittering_scaled_data,
                                         self.no_jittering_scaled_subset_data],
                                        axis=1)

        # Random generators for jittering
        random = np.random.RandomState(seed=self.jitter_seed)
        rand_seeds = random.random_integers(0, sys.maxsize - 1, size=len(data.domain))
        for index, rseed in zip(list(range(len(data.domain))), rand_seeds):
            # Need to use a different seed for each feature
            random = np.random.RandomState(seed=rseed)
            attr = data.domain[index]
            if attr.var_type == VarTypes.Discrete:
                scaled_data[index] += (self.jitter_size / (50.0 * max(1, len(attr.values)))) * \
                                      (random.rand(len(full_data)) - 0.5)

            elif attr.var_type == VarTypes.Continuous and self.jitter_continuous:
                scaled_data[index] += self.jitter_size / 50.0 * (0.5 - random.rand(len(full_data)))
                scaled_data[index] = np.absolute(scaled_data[index])       # fix values below zero
                ind = np.where(scaled_data[index] > 1.0, 1, 0)     # fix values above 1
                np.putmask(scaled_data[index], ind, 2.0 - np.compress(ind, scaled_data[index]))

        if self.have_subset_data:
            # Fix all subset instances which are also in the main data
            # to have the same jittered values
            ids_to_indices = dict((inst.id, i)
                                  for i, inst in enumerate(self.raw_data))

            subset_ids_map = [[i, ids_to_indices[s.id]]
                              for i, s in enumerate(self.raw_subset_data)
                              if s.id in ids_to_indices]
            if len(subset_ids_map):
                subset_ids_map = np.array(subset_ids_map)
                subset_ids_map[:, 0] += len_data
                scaled_data[:, subset_ids_map[:, 0]] = \
                    scaled_data[:, subset_ids_map[:, 1]]

        self.scaled_data = scaled_data[:, :len_data]
        self.scaled_subset_data = scaled_data[:, len_data:]

    setData = set_data

    def scale_example_value(self, instance, index):
        """
        Scale instance's value at index index to a range between 0 and 1 with
        respect to self.raw_data.
        """
        if instance[index].isSpecial():
            print("Warning: scaling instance with missing value")
            return 0.5
        if instance.domain[index].var_type == VarTypes.Discrete:
            d = get_variable_value_indices(instance.domain[index])
            return (d[instance[index].value] * 2 + 1) / float(2 * len(d))
        elif instance.domain[index].var_type == VarTypes.Continuous:
            diff = self.domain_data_stat[index].max - self.domain_data_stat[index].min
            if diff == 0:
                diff = 1          # if all values are the same then prevent division by zero
            return (instance[index] - self.domain_data_stat[index].min) / diff

    scaleExampleValue = scale_example_value

    def get_attribute_label(self, attr_name):
        if self.attribute_flip_info.get(attr_name, 0) and \
                        self.data_domain[attr_name].var_type == VarTypes.Continuous:
            return "-" + attr_name
        return attr_name

    getAttributeLabel = get_attribute_label

    def flip_attribute(self, attr_name):
        if attr_name not in self.attribute_names:
            return 0
        if self.data_domain[attr_name].var_type == VarTypes.Discrete:
            return 0

        index = self.attribute_name_index[attr_name]
        self.attribute_flip_info[attr_name] = 1 - self.attribute_flip_info.get(attr_name, 0)
        if self.data_domain[attr_name].var_type == VarTypes.Continuous:
            self.attr_values[attr_name] = [-self.attr_values[attr_name][1], -self.attr_values[attr_name][0]]

        self.scaled_data[index] = 1 - self.scaled_data[index]
        self.scaled_subset_data[index] = 1 - self.scaled_subset_data[index]
        self.no_jittering_scaled_data[index] = 1 - self.no_jittering_scaled_data[index]
        self.no_jittering_scaled_subset_data[index] = 1 - self.no_jittering_scaled_subset_data[index]
        return 1

    flipAttribute = flip_attribute

    def get_min_max_val(self, attr):
        if type(attr) == int:
            attr = self.attribute_names[attr]
        diff = self.attr_values[attr][1] - self.attr_values[attr][0]
        return diff or 1.0

    getMinMaxVal = get_min_max_val

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
        selected_array = self.valid_data_array.take(inds, axis=0)
        arr = np.add.reduce(selected_array)
        return np.equal(arr, len(inds))

    getValidList = get_valid_list

    def get_valid_subset_list(self, indices, also_class_if_exists=1):
        """
        Get array of 0 and 1 of len = len(self.raw_subset_data). if there is a
        missing value at any attribute in indices return 0 for that instance.
        """
        if self.valid_subset_data_array is None or len(self.valid_subset_data_array) == 0:
            return np.array([], np.bool)
        inds = indices[:]
        if also_class_if_exists and self.data_class_index:
            inds.append(self.data_class_index)
        selected_array = self.valid_subset_data_array.take(inds, axis=0)
        arr = np.add.reduce(selected_array)
        return np.equal(arr, len(inds))

    getValidSubsetList = get_valid_subset_list

    def get_valid_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_list(indices)
        return np.nonzero(valid_list)[0]

    getValidIndices = get_valid_indices

    def get_valid_subset_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_subset_list(indices)
        return np.nonzero(valid_list)[0]

    getValidSubsetIndices = get_valid_subset_indices

    def rnd_correction(self, max):
        """
        Return a number from -max to max.
        """
        return (random.random() - 0.5) * 2 * max
