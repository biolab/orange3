"""
.. index:: data scaling

.. index::
   single: scaling

**************************
Data Scaling (``scaling``)
**************************

This module is a conglomerate of Orange 2.0 modules orngScaleData,
orngScaleLinProjData, orngScaleLinProjData3D, orngScalePolyvizData and orngScaleScatterPlotData. The
documentation is poor and has to be improved in the future.

.. autoclass:: Orange.data.preprocess.scaling.ScaleData
   :members:
   :show-inheritance:

.. autoclass:: Orange.data.preprocess.scaling.ScaleLinProjData
   :members:
   :show-inheritance:

.. autoclass:: Orange.data.preprocess.scaling.ScaleLinProjData3D
   :members:
   :show-inheritance:

.. autoclass:: Orange.data.preprocess.scaling.ScalePolyvizData
   :members:
   :show-inheritance:

.. autoclass:: Orange.data.preprocess.scaling.ScaleScatterPlotData
   :members:
   :show-inheritance:

.. autofunction:: get_variable_values_sorted

.. autofunction:: get_variable_value_indices

.. autofunction:: discretize_domain

"""

import sys
import random
import time

import numpy


try:
    import numpy.ma as MA
except:
    import numpy.core.ma as MA
import math

import Orange
import Orange.core
import Orange.data
from Orange.data import preprocess

from Orange.utils import caching

import warnings

from Orange.utils import deprecated_keywords, deprecated_members


def get_variable_values_sorted(variable):
    """
    Return a list of sorted values for given attribute.
    
    EXPLANATION: if variable values have values 1, 2, 3, 4, ... then their
    order in orange depends on when they appear first in the data. With this
    function we get a sorted list of values.
    
    """
    if variable.var_type == Orange.core.VarTypes.Continuous:
        print("get_variable_values_sorted - attribute %s is a continuous variable" % variable)
        return []

    values = list(variable.values)
    int_values = []

    # do all attribute values containt integers?
    try:
        int_values = [(int(val), val) for val in values]
    except ValueError:
        return values

    # if all values were intergers, we first sort them ascendently
    int_values.sort()
    return [val[1] for val in int_values]


@deprecated_keywords({"sortValuesForDiscreteAttrs":
                          "sort_values_for_discrete_attrs"})
def get_variable_value_indices(variable, sort_values_for_discrete_attrs=1):
    """
    Create a dictionary with given variable. Keys are variable values, values
    are indices (transformed from string to int); in case all values are
    integers, we also sort them.
    
    """
    if variable.var_type == Orange.core.VarTypes.Continuous:
        print("get_variable_value_indices - attribute %s is a continuous " \
              "variable" % (str(variable)))
        return {}

    if sort_values_for_discrete_attrs:
        values = get_variable_values_sorted(variable)
    else:
        values = list(variable.values)
    return dict([(values[i], i) for i in range(len(values))])


@deprecated_keywords({"removeUnusedValues": "remove_unused_values",
                      "numberOfIntervals": "number_of_intervals"})
def discretize_domain(data, remove_unused_values=1, number_of_intervals=2):
    """
    Discretize the domain. If we have a class, remove the instances with missing
    class value, discretize the continuous class into discrete class with two
    values, discretize continuous attributes using entropy discretization (or
    equiN if we don't have a class or class is continuous).
    """
    entro_disc = preprocess.EntropyDiscretization()
    equi_disc = preprocess.EquiNDiscretization(number_of_intervals=number_of_intervals)
    disc_attrs = []

    classname = (data and len(data) > 0 and data.domain.class_var and
                 data.domain.class_var.name or None)

    if not data or len(data) == 0:
        return None

    # if we have a continuous class we have to discretize it before we can
    # discretize the attributes
    if classname and data.domain.class_var.var_type == \
            Orange.core.VarTypes.Continuous:
        try:
            newclass = equi_disc(data.domain.class_var.name, data)
            newclass.name = classname
        except Orange.core.KernelException as ex:
            warnings.warn("Could not discretize class variable '%s'. %s" %
                          (data.domain.class_var.name, ex.message))
            newclass = None
            classname = None
        new_domain = Orange.data.Domain(data.domain.attributes, newclass)
        data = Orange.data.Table(new_domain, data)

    for attr in data.domain.attributes:
        try:
            name = attr.name
            if attr.var_type == Orange.core.VarTypes.Continuous:
                # if continuous attribute then use entropy discretization
                if data.domain.class_var and data.domain.class_var.var_type == \
                        Orange.core.VarTypes.Discrete:
                    new_attr = entro_disc(attr, data)
                else:
                    new_attr = equi_disc(attr, data)
            else:
                new_attr = attr
            if remove_unused_values:
                new_attr = preprocess.RemoveUnusedValues(new_attr, data)
                if new_attr is None:
                    raise Orange.core.KernelException("No values")

            new_attr.name = name
            disc_attrs.append(new_attr)
        except Orange.core.KernelException as ex:
            # if all values are missing, entropy discretization will throw an
            # exception. in such cases ignore the attribute
            warnings.warn("Could not discretize %s attribute. %s" %
                          (attr.name, ex.message))

    if classname: disc_attrs.append(data.domain.class_var)
    d2 = data.translate(disc_attrs, True)
    return d2


class ScaleData:
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

        self.jitter_size = 10
        self.jitter_continuous = 0
        self.jitter_seed = 0

        self.attr_values = {}
        self.domain_data_stat = []
        self.original_data = self.original_subset_data = None    # input (nonscaled) data in a numpy array
        self.scaled_data = self.scaled_subset_data = None        # scaled data to the interval 0-1
        self.no_jittering_scaled_data = self.no_jittering_scaled_subset_data = None
        self.valid_data_array = self.valid_subset_data_array = None

    @deprecated_keywords({"subsetData": "subset_data"})
    def merge_data_sets(self, data, subset_data):
        """
        Take examples from data and subset_data and merge them into one
        dataset.
        
        """
        if data == None and subset_data == None: None
        if subset_data == None:
            full_data = data
        elif data == None:
            full_data = subset_data
        else:
            full_data = Orange.data.Table(data)
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

    @deprecated_keywords({"subsetData": "subset_data",
                          "sortValuesForDiscreteAttrs":
                              "sort_values_for_discrete_attrs"})
    def set_data(self, data, subset_data=None, **args):
        if args.get("skipIfSame", 1):
            if (((data == None and self.raw_data == None) or
                     (self.raw_data != None and data != None and
                              self.raw_data.checksum() == data.checksum())) and
                    ((subset_data == None and self.raw_subset_data == None) or
                         (self.raw_subset_data != None and subset_data != None and
                                  self.raw_subset_data.checksum() == subset_data.checksum()))):
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

        if data == None: return
        full_data = self.merge_data_sets(data, subset_data)

        self.raw_data = data
        self.raw_subset_data = subset_data

        len_data = data and len(data) or 0

        self.attribute_names = [attr.name for attr in full_data.domain]
        self.attribute_name_index = dict([(full_data.domain[i].name, i)
                                          for i in range(len(full_data.domain))])
        self.attribute_flip_info = {}         # dict([(attr.name, 0) for attr in full_data.domain]) # reset the fliping information

        self.data_domain = full_data.domain
        self.data_has_class = bool(full_data.domain.class_var)
        self.data_has_continuous_class = bool(self.data_has_class and
                                              full_data.domain.class_var.var_type == Orange.core.VarTypes.Continuous)
        self.data_has_discrete_class = bool(self.data_has_class and
                                            full_data.domain.class_var.var_type == Orange.core.VarTypes.Discrete)
        self.data_class_name = self.data_has_class and full_data.domain.class_var.name
        if self.data_has_class:
            self.data_class_index = self.attribute_name_index[self.data_class_name]
        self.have_data = bool(self.raw_data and len(self.raw_data) > 0)
        self.have_subset_data = bool(self.raw_subset_data and
                                     len(self.raw_subset_data) > 0)

        self.domain_data_stat = caching.getCached(full_data,
                                                  Orange.core.DomainBasicAttrStat,
                                                  (full_data,))

        sort_values_for_discrete_attrs = args.get("sort_values_for_discrete_attrs",
                                                  1)

        for index in range(len(full_data.domain)):
            attr = full_data.domain[index]
            if attr.var_type == Orange.core.VarTypes.Discrete:
                self.attr_values[attr.name] = [0, len(attr.values)]
            elif attr.var_type == Orange.core.VarTypes.Continuous:
                self.attr_values[attr.name] = [self.domain_data_stat[index].min,
                                               self.domain_data_stat[index].max]

        # the original_data, no_jittering_scaled_data and validArray are arrays
        # that we can cache so that other visualization widgets don't need to
        # compute it. The scaled_data on the other hand has to be computed for
        # each widget separately because of different
        # jitter_continuous and jitter_size values
        if caching.getCached(data, "visualizationData") and subset_data == None:
            self.original_data, self.no_jittering_scaled_data, self.valid_data_array = caching.getCached(data,
                                                                                                         "visualizationData")
            self.original_subset_data = self.no_jittering_scaled_subset_data = self.valid_subset_data_array = numpy.array(
                []).reshape([len(self.original_data), 0])
        else:
            no_jittering_data = full_data.toNumpyMA("ac")[0].T
            valid_data_array = numpy.array(1 - no_jittering_data.mask,
                                           numpy.short)  # have to convert to int array, otherwise when we do some operations on this array we get overflow
            no_jittering_data = numpy.array(MA.filled(no_jittering_data,
                                                      Orange.core.Illegal_Float))
            original_data = no_jittering_data.copy()

            for index in range(len(data.domain)):
                attr = data.domain[index]
                if attr.var_type == Orange.core.VarTypes.Discrete:
                    # see if the values for discrete attributes have to be resorted
                    variable_value_indices = get_variable_value_indices(data.domain[index],
                                                                        sort_values_for_discrete_attrs)
                    if 0 in [i == variable_value_indices[attr.values[i]]
                             for i in range(len(attr.values))]:
                        # make the array a contiguous, otherwise the putmask 
                        # function does not work
                        line = no_jittering_data[index].copy()
                        indices = [numpy.where(line == val, 1, 0)
                                   for val in range(len(attr.values))]
                        for i in range(len(attr.values)):
                            numpy.putmask(line, indices[i],
                                          variable_value_indices[attr.values[i]])
                        no_jittering_data[index] = line   # save the changed array
                        original_data[index] = line     # reorder also the values in the original data
                    no_jittering_data[index] = ((no_jittering_data[index] * 2.0 + 1.0)
                                                / float(2 * len(attr.values)))

                elif attr.var_type == Orange.core.VarTypes.Continuous:
                    diff = self.domain_data_stat[index].max - self.domain_data_stat[
                        index].min or 1     # if all values are the same then prevent division by zero
                    no_jittering_data[index] = (no_jittering_data[index] -
                                                self.domain_data_stat[index].min) / diff

            self.original_data = original_data[:, :len_data];
            self.original_subset_data = original_data[:, len_data:]
            self.no_jittering_scaled_data = no_jittering_data[:, :len_data];
            self.no_jittering_scaled_subset_data = no_jittering_data[:, len_data:]
            self.valid_data_array = valid_data_array[:, :len_data];
            self.valid_subset_data_array = valid_data_array[:, len_data:]

        if data: caching.setCached(data, "visualizationData",
                                   (self.original_data, self.no_jittering_scaled_data,
                                    self.valid_data_array))
        if subset_data: caching.setCached(subset_data, "visualizationData",
                                          (self.original_subset_data,
                                           self.no_jittering_scaled_subset_data,
                                           self.valid_subset_data_array))

        # compute the scaled_data arrays
        scaled_data = numpy.concatenate([self.no_jittering_scaled_data,
                                         self.no_jittering_scaled_subset_data],
                                        axis=1)

        # Random generators for jittering
        random = numpy.random.RandomState(seed=self.jitter_seed)
        rand_seeds = random.random_integers(0, sys.maxsize - 1, size=len(data.domain))
        for index, rseed in zip(list(range(len(data.domain))), rand_seeds):
            # Need to use a different seed for each feature
            random = numpy.random.RandomState(seed=rseed)
            attr = data.domain[index]
            if attr.var_type == Orange.core.VarTypes.Discrete:
                scaled_data[index] += (self.jitter_size / (50.0 * max(1, len(attr.values)))) * \
                                      (random.rand(len(full_data)) - 0.5)

            elif attr.var_type == Orange.core.VarTypes.Continuous and self.jitter_continuous:
                scaled_data[index] += self.jitter_size / 50.0 * (0.5 - random.rand(len(full_data)))
                scaled_data[index] = numpy.absolute(scaled_data[index])       # fix values below zero
                ind = numpy.where(scaled_data[index] > 1.0, 1, 0)     # fix values above 1
                numpy.putmask(scaled_data[index], ind, 2.0 - numpy.compress(ind, scaled_data[index]))

        if self.have_subset_data:
            # Fix all subset instances which are also in the main data
            # to have the same jittered values
            ids_to_indices = dict((inst.id, i) \
                                  for i, inst in enumerate(self.raw_data))

            subset_ids_map = [[i, ids_to_indices[s.id]] \
                              for i, s in enumerate(self.raw_subset_data) \
                              if s.id in ids_to_indices]
            if len(subset_ids_map):
                subset_ids_map = numpy.array(subset_ids_map)
                subset_ids_map[:, 0] += len_data
                scaled_data[:, subset_ids_map[:, 0]] = \
                    scaled_data[:, subset_ids_map[:, 1]]

        self.scaled_data = scaled_data[:, :len_data];
        self.scaled_subset_data = scaled_data[:, len_data:]

    setData = set_data

    @deprecated_keywords({"example": "instance"})
    def scale_example_value(self, instance, index):
        """
        Scale instance's value at index index to a range between 0 and 1 with
        respect to self.raw_data.
        """
        if instance[index].isSpecial():
            print("Warning: scaling instance with missing value")
            return 0.5     #1e20
        if instance.domain[index].var_type == Orange.core.VarTypes.Discrete:
            d = get_variable_value_indices(instance.domain[index])
            return (d[instance[index].value] * 2 + 1) / float(2 * len(d))
        elif instance.domain[index].var_type == Orange.core.VarTypes.Continuous:
            diff = self.domain_data_stat[index].max - self.domain_data_stat[index].min
            if diff == 0: diff = 1          # if all values are the same then prevent division by zero
            return (instance[index] - self.domain_data_stat[index].min) / diff

    scaleExampleValue = scale_example_value

    @deprecated_keywords({"attrName": "attr_name"})
    def get_attribute_label(self, attr_name):
        if self.attribute_flip_info.get(attr_name, 0) and self.data_domain[
            attr_name].var_type == Orange.core.VarTypes.Continuous:
            return "-" + attr_name
        return attr_name

    getAttributeLabel = get_attribute_label

    @deprecated_keywords({"attrName": "attr_name"})
    def flip_attribute(self, attr_name):
        if attr_name not in self.attribute_names: return 0
        if self.data_domain[attr_name].var_type == Orange.core.VarTypes.Discrete: return 0

        index = self.attribute_name_index[attr_name]
        self.attribute_flip_info[attr_name] = 1 - self.attribute_flip_info.get(attr_name, 0)
        if self.data_domain[attr_name].var_type == Orange.core.VarTypes.Continuous:
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

    @deprecated_keywords({"alsoClassIfExists": "also_class_if_exists"})
    def get_valid_list(self, indices, also_class_if_exists=1):
        """
        Get array of 0 and 1 of len = len(self.raw_data). If there is a missing
        value at any attribute in indices return 0 for that instance.
        """
        if self.valid_data_array == None or len(self.valid_data_array) == 0:
            return numpy.array([], numpy.bool)

        inds = indices[:]
        if also_class_if_exists and self.data_has_class:
            inds.append(self.data_class_index)
        selected_array = self.valid_data_array.take(inds, axis=0)
        arr = numpy.add.reduce(selected_array)
        return numpy.equal(arr, len(inds))

    getValidList = get_valid_list

    @deprecated_keywords({"alsoClassIfExists": "also_class_if_exists"})
    def get_valid_subset_list(self, indices, also_class_if_exists=1):
        """
        Get array of 0 and 1 of len = len(self.raw_subset_data). if there is a
        missing value at any attribute in indices return 0 for that instance.
        """
        if self.valid_subset_data_array == None or len(self.valid_subset_data_array) == 0:
            return numpy.array([], numpy.bool)
        inds = indices[:]
        if also_class_if_exists and self.data_class_index:
            inds.append(self.data_class_index)
        selected_array = self.valid_subset_data_array.take(inds, axis=0)
        arr = numpy.add.reduce(selected_array)
        return numpy.equal(arr, len(inds))

    getValidSubsetList = get_valid_subset_list

    def get_valid_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_list(indices)
        return numpy.nonzero(valid_list)[0]

    getValidIndices = get_valid_indices

    def get_valid_subset_indices(self, indices):
        """
        Get array with numbers that represent the instance indices that have a
        valid data value.
        """
        valid_list = self.get_valid_subset_list(indices)
        return numpy.nonzero(valid_list)[0]

    getValidSubsetIndices = get_valid_subset_indices

    def rnd_correction(self, max):
        """
        Return a number from -max to max.
        """
        return (random.random() - 0.5) * 2 * max

    rndCorrection = rnd_correction


ScaleData = deprecated_members({"rawData": "raw_data",
                                "rawSubsetData": "raw_subset_data",
                                "attributeNames": "attribute_names",
                                "attributeNameIndex": "attribute_name_index",
                                "attributeFlipInfo": "attribute_flip_info",
                                "dataHasClass": "data_has_class",
                                "dataHasContinuousClass": "data_has_continuous_class",
                                "dataHasDiscreteClass": "data_has_discrete_class",
                                "dataClassName": "data_class_name",
                                "dataDomain": "data_domain",
                                "dataClassIndex": "data_class_index",
                                "haveData": "have_data",
                                "haveSubsetData": "have_subset_data",
                                "jitterSize": "jitter_size",
                                "jitterContinuous": "jitter_continuous",
                                "attrValues": "attr_values",
                                "domainDataStat": "domain_data_stat",
                                "originalData": "original_data",
                                "originalSubsetData": "original_subset_data",
                                "scaledData": "scaled_data",
                                "scaledSubsetData": "scaled_subset_data",
                                "noJitteringScaledData": "no_jittering_scaled_data",
                                "noJitteringScaledSubsetData": "no_jittering_scaled_subset_data",
                                "validDataArray": "valid_data_array",
                                "validSubsetDataArray": "valid_subset_data_array",
                                "mergeDataSets": "merge_data_sets",
                                "rescaleData": "rescale_data",
                                "setData": "set_data",
                                "scaleExampleValue": "scale_example_value",
                                "getAttributeLabel": "get_attribute_label",
                                "flipAttribute": "flip_attribute",
                                "getMinMaxVal": "get_min_max_val",
                                "getValidList": "get_valid_list",
                                "getValidSubsetList": "get_valid_subset_list",
                                "getValidIndices": "get_valid_indices",
                                "getValidSubsetIndices": "get_valid_subset_indices",
                                "rndCorrection": "rnd_correction",
})(ScaleData)


def jitter_array(array, ratio=0.01, axis=0, rand_seed=None):
    """
    """
    array = numpy.array(array)
    shape = array.shape

    if array.ndim == 1:
        if axis != 0:
            raise ValueError("Invalid axis")
        array = array.reshape((-1, 1))

    if array.ndim > 2:
        raise ValueError("'array' must be at most 2 dimensional.")

    axis_min = array.min(axis=axis)
    axis_max = array.max(axis=axis)
    axis_span = axis_max - axis_min

    # roll axis to front
    array = numpy.rollaxis(array, axis, 0)

    random = numpy.random.RandomState(rand_seed)
    for i, span in enumerate(axis_span):
        array[:, i] += random.uniform(-ratio * span / 2, ratio * span / 2,
                                      array.shape[0])

    # roll axis back to its original position
    array = numpy.rollaxis(array, 0, axis + 1)
    array = array.reshape(shape)
    return array


class ScaleLinProjData(ScaleData):
    def __init__(self):
        ScaleData.__init__(self)
        self.normalize_examples = 1
        self.anchor_data = []        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.last_attr_indices = None
        self.anchor_dict = {}

    @deprecated_keywords({"xAnchors": "xanchors", "yAnchors": "yanchors"})
    def set_anchors(self, xanchors, yanchors, attributes):
        if attributes:
            if xanchors != None and yanchors != None:
                self.anchor_data = [(xanchors[i], yanchors[i], attributes[i])
                                    for i in range(len(attributes))]
            else:
                self.anchor_data = self.create_anchors(len(attributes), attributes)

    setAnchors = set_anchors

    @deprecated_keywords({"numOfAttr": "num_of_attr"})
    def create_anchors(self, num_of_attr, labels=None):
        """
        Create anchors around the circle.
        
        """
        xanchors = self.create_xanchors(num_of_attr)
        yanchors = self.create_yanchors(num_of_attr)
        if labels:
            return [(xanchors[i], yanchors[i], labels[i]) for i in range(num_of_attr)]
        else:
            return [(xanchors[i], yanchors[i]) for i in range(num_of_attr)]

    createAnchors = create_anchors

    @deprecated_keywords({"numOfAttrs": "num_of_attrs"})
    def create_xanchors(self, num_of_attrs):
        if num_of_attrs not in self.anchor_dict:
            self.anchor_dict[num_of_attrs] = (numpy.cos(numpy.arange(num_of_attrs)
                                                        * 2 * math.pi
                                                        / float(num_of_attrs)),
                                              numpy.sin(numpy.arange(num_of_attrs)
                                                        * 2 * math.pi
                                                        / float(num_of_attrs)))
        return self.anchor_dict[num_of_attrs][0]

    createXAnchors = create_xanchors

    @deprecated_keywords({"numOfAttrs": "num_of_attrs"})
    def create_yanchors(self, num_of_attrs):
        if num_of_attrs not in self.anchor_dict:
            self.anchor_dict[num_of_attrs] = (numpy.cos(numpy.arange(num_of_attrs)
                                                        * 2 * math.pi
                                                        / float(num_of_attrs)),
                                              numpy.sin(numpy.arange(num_of_attrs)
                                                        * 2 * math.pi
                                                        / float(num_of_attrs)))
        return self.anchor_dict[num_of_attrs][1]

    createYAnchors = create_yanchors

    @deprecated_keywords({"fileName": "filename", "attrList": "attrlist",
                          "useAnchorData": "use_anchor_data"})
    def save_projection_as_tab_data(self, filename, attrlist, use_anchor_data=0):
        """
        Save projection (xattr, yattr, classval) into a filename filename.
        
        """
        Orange.core.saveTabDelimited(filename,
                                     self.create_projection_as_example_table([self.attribute_name_index[i]
                                                                              for i in attrlist],
                                                                             use_anchor_data=use_anchor_data))

    saveProjectionAsTabData = save_projection_as_tab_data

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "useAnchorData": "use_anchor_data",
                          "xAnchors": "xanchors",
                          "yAnchors": "yanchors",
                          "anchorRadius": "anchor_radius",
                          "normalizeExample": "normalize_example"
    })
    def get_projected_point_position(self, attr_indices, values, **settings_dict):
        """
        For attributes in attr_indices and values of these attributes in values
        compute point positions. This function has more sense in radviz and
        polyviz methods.
    
        """
        # load the elements from the settings dict
        use_anchor_data = settings_dict.get("use_anchor_data")
        xanchors = settings_dict.get("x_anchors")
        yanchors = settings_dict.get("y_anchors")
        anchor_radius = settings_dict.get("anchor_radius")
        normalize_example = settings_dict.get("normalize_example")

        if attr_indices != self.last_attr_indices:
            print("get_projected_point_position. Warning: Possible bug. The " + \
                  "set of attributes is not the same as when computing the " + \
                  "whole projection")

        if xanchors != None and yanchors != None:
            xanchors = numpy.array(xanchors)
            yanchors = numpy.array(yanchors)
            if anchor_radius == None: anchor_radius = numpy.sqrt(xanchors * xanchors +
                                                                 yanchors * yanchors)
        elif use_anchor_data and self.anchor_data:
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            if anchor_radius == None: anchor_radius = numpy.sqrt(xanchors * xanchors +
                                                                 yanchors * yanchors)
        else:
            xanchors = self.create_xanchors(len(attr_indices))
            yanchors = self.create_yanchors(len(attr_indices))
            anchor_radius = numpy.ones(len(attr_indices), numpy.float)

        if normalize_example == 1 or (normalize_example == None
                                      and self.normalize_examples):
            m = min(values);
            M = max(values)
            if m < 0.0 or M > 1.0:
            # we have to do rescaling of values so that all the values will
                # be in the 0-1 interval
                #print "example values are not in the 0-1 interval"
                values = [max(0.0, min(val, 1.0)) for val in values]
                #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
                #values = [(val-m) / float(diff) for val in values]

            s = sum(numpy.array(values) * anchor_radius)
            if s == 0: return [0.0, 0.0]
            x = self.trueScaleFactor * numpy.dot(xanchors * anchor_radius,
                                                 values) / float(s)
            y = self.trueScaleFactor * numpy.dot(yanchors * anchor_radius,
                                                 values) / float(s)
        else:
            x = self.trueScaleFactor * numpy.dot(xanchors, values)
            y = self.trueScaleFactor * numpy.dot(yanchors, values)

        return [x, y]

    getProjectedPointPosition = get_projected_point_position

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
    def create_projection_as_example_table(self, attr_indices, **settings_dict):
        """
        Create the projection of attribute indices given in attr_indices and
        create an example table with it.
        """
        if self.data_domain.class_var:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar"),
                                         Orange.feature.Discrete(self.data_domain.class_var.name,
                                                                 values=get_variable_values_sorted(
                                                                     self.data_domain.class_var))])
        else:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar")])
        data = self.create_projection_as_numeric_array(attr_indices,
                                                       **settings_dict)
        if data != None:
            return Orange.data.Table(domain, data)
        else:
            return Orange.data.Table(domain)

    createProjectionAsExampleTable = create_projection_as_example_table

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "validData": "valid_data",
                          "classList": "class_list",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors",
                          "scaleFactor": "scale_factor",
                          "jitterSize": "jitter_size",
                          "useAnchorData": "use_anchor_data",
                          "removeMissingData": "remove_missing_data",
                          "useSubsetData": "use_subset_data",
    })
    def create_projection_as_numeric_array(self, attr_indices, **settings_dict):
        # load the elements from the settings dict
        valid_data = settings_dict.get("valid_data")
        class_list = settings_dict.get("class_list")
        sum_i = settings_dict.get("sum_i")
        xanchors = settings_dict.get("xanchors")
        yanchors = settings_dict.get("yanchors")
        scale_factor = settings_dict.get("scale_factor", 1.0)
        normalize = settings_dict.get("normalize")
        jitter_size = settings_dict.get("jitter_size", 0.0)
        use_anchor_data = settings_dict.get("use_anchor_data", 0)
        remove_missing_data = settings_dict.get("remove_missing_data", 1)
        use_subset_data = settings_dict.get("use_subset_data", 0)        # use the data or subsetData?
        #minmaxVals = settings_dict.get("minmaxVals", None)

        # if we want to use anchor data we can get attr_indices from the anchor_data
        if use_anchor_data and self.anchor_data:
            attr_indices = [self.attribute_name_index[val[2]] for val in self.anchor_data]

        if valid_data == None:
            if use_subset_data:
                valid_data = self.get_valid_subset_list(attr_indices)
            else:
                valid_data = self.get_valid_list(attr_indices)
        if sum(valid_data) == 0:
            return None

        if class_list == None and self.data_domain.class_var:
            if use_subset_data:
                class_list = self.original_subset_data[self.data_class_index]
            else:
                class_list = self.original_data[self.data_class_index]

        # if jitterSize is set below zero we use scaled_data that has already jittered data
        if use_subset_data:
            if jitter_size < 0.0:
                data = self.scaled_subset_data
            else:
                data = self.no_jittering_scaled_subset_data
        else:
            if jitter_size < 0.0:
                data = self.scaled_data
            else:
                data = self.no_jittering_scaled_data

        selectedData = numpy.take(data, attr_indices, axis=0)
        if remove_missing_data:
            selectedData = numpy.compress(valid_data, selectedData, axis=1)
            if class_list != None and len(class_list) != numpy.shape(selectedData)[1]:
                class_list = numpy.compress(valid_data, class_list)

        if use_anchor_data and self.anchor_data:
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            r = numpy.sqrt(
                xanchors * xanchors + yanchors * yanchors)     # compute the distance of each anchor from the center of the circle
            if normalize == 1 or (normalize == None and self.normalize_examples):
                xanchors *= r
                yanchors *= r
        elif (xanchors != None and yanchors != None):
            xanchors = numpy.array(xanchors);
            yanchors = numpy.array(yanchors)
            r = numpy.sqrt(
                xanchors * xanchors + yanchors * yanchors)     # compute the distance of each anchor from the center of the circle
        else:
            xanchors = self.create_xanchors(len(attr_indices))
            yanchors = self.create_yanchors(len(attr_indices))
            r = numpy.ones(len(xanchors), numpy.float)

        x_positions = numpy.dot(xanchors, selectedData)
        y_positions = numpy.dot(yanchors, selectedData)

        if normalize == 1 or (normalize == None and self.normalize_examples):
            if sum_i == None:
                sum_i = self._getSum_i(selectedData, use_anchor_data, r)
            x_positions /= sum_i
            y_positions /= sum_i
            self.trueScaleFactor = scale_factor
        else:
            if not remove_missing_data:
                try:
                    x_valid_data = numpy.compress(valid_data, x_positions)
                    y_valid_data = numpy.compress(valid_data, y_positions)
                except:
                    print(valid_data)
                    print(x_positions)
                    print(numpy.shape(valid_data))
                    print(numpy.shape(x_positions))
            else:
                x_valid_data = x_positions
                y_valid_data = y_positions

            dist = math.sqrt(max(x_valid_data * x_valid_data + y_valid_data * y_valid_data)) or 1
            self.trueScaleFactor = scale_factor / dist

        self.unscaled_x_positions = numpy.array(x_positions)
        self.unscaled_y_positions = numpy.array(y_positions)

        if self.trueScaleFactor != 1.0:
            x_positions *= self.trueScaleFactor
            y_positions *= self.trueScaleFactor

        if jitter_size > 0.0:
            positions = numpy.vstack((x_positions, y_positions))
            positions = jitter_array(positions, jitter_size / 100., axis=1,
                                     rand_seed=self.jitter_seed)
            x_positions, y_positions = numpy.vsplit(positions, 2)
            x_positions = x_positions.ravel()
            y_positions = y_positions.ravel()

        self.last_attr_indices = attr_indices
        if class_list != None:
            return numpy.transpose(numpy.array((x_positions, y_positions, class_list)))
        else:
            return numpy.transpose(numpy.array((x_positions, y_positions)))

    createProjectionAsNumericArray = create_projection_as_numeric_array

    @deprecated_keywords({"useAnchorData": "use_anchor_data",
                          "anchorRadius": "anchor_radius"})
    def _getsum_i(self, data, use_anchor_data=0, anchor_radius=None):
        """
        Function to compute the sum of all values for each element in the data.
        Used to normalize.
        
        """
        if use_anchor_data:
            if anchor_radius == None:
                anchor_radius = numpy.sqrt([a[0] ** 2 + a[1] ** 2 for a in self.anchor_data])
            sum_i = numpy.add.reduce(numpy.transpose(numpy.transpose(data) * anchor_radius))
        else:
            sum_i = numpy.add.reduce(data)
        if len(sum_i[numpy.nonzero(sum_i)]) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += numpy.where(sum_i == 0, 1.0, 0.0)
        return sum_i

    _getSum_i = _getsum_i


graph_deprecator = deprecated_members({"setData": "set_data",
                                       "updateData": "update_data",
                                       "scaleFactor": "scale_factor"})

ScaleLinProjData = deprecated_members({"setAnchors": "set_anchors",
                                       "createAnchors": "create_anchors",
                                       "createXAnchors": "create_xanchors",
                                       "createYAnchors": "create_yanchors",
                                       "saveProjectionAsTabData": "save_projection_as_tab_data",
                                       "getProjectedPointPosition":
                                           "get_projected_point_position",
                                       "createProjectionAsExampleTable":
                                           "create_projection_as_example_table",
                                       "createProjectionAsNumericArray":
                                           "create_projection_as_numeric_array",
                                       "_getSum_i": "_getsum_i",
                                       "normalizeExamples": "normalize_examples",
                                       "anchorData": "anchor_data",
                                       "lastAttrIndices": "last_attr_indices",
                                       "anchorDict": "anchor_dict"})(ScaleLinProjData)


class ScaleLinProjData3D(ScaleData):
    def __init__(self):
        ScaleData.__init__(self)
        self.normalize_examples = 1
        self.anchor_data = []        # form: [(anchor1x, anchor1y, anchor1z, label1),(anchor2x, anchor2y, anchor2z, label2), ...]
        self.last_attr_indices = None
        self.anchor_dict = {}

    @deprecated_keywords({"xAnchors": "xanchors", "yAnchors": "yanchors"})
    def set_anchors(self, xanchors, yanchors, zanchors, attributes):
        if attributes:
            if xanchors != None and yanchors != None and zanchors != None:
                self.anchor_data = [(xanchors[i], yanchors[i], zanchors[i], attributes[i])
                                    for i in range(len(attributes))]
            else:
                self.anchor_data = self.create_anchors(len(attributes), attributes)

    setAnchors = set_anchors

    @deprecated_keywords({"numOfAttr": "num_of_attr"})
    def create_anchors(self, num_of_attrs, labels=None):
        """
        Create anchors on the sphere.
        
        """
        # Golden Section Spiral algorithm approximates even distribution of points on a sphere
        # (read more here http://www.softimageblog.com/archives/115)
        n = num_of_attrs
        xanchors = []
        yanchors = []
        zanchors = []

        inc = math.pi * (3 - math.sqrt(5))
        off = 2. / n
        for k in range(n):
            y = k * off - 1 + (off / 2)
            r = math.sqrt(1 - y * y)
            phi = k * inc
            xanchors.append(math.cos(phi) * r)
            yanchors.append(y)
            zanchors.append(math.sin(phi) * r)

        self.anchor_dict[num_of_attrs] = [xanchors, yanchors, zanchors]

        if labels:
            return [(xanchors[i], yanchors[i], zanchors[i], labels[i]) for i in range(num_of_attrs)]
        else:
            return [(xanchors[i], yanchors[i], zanchors[i]) for i in range(num_of_attrs)]

    createAnchors = create_anchors

    @deprecated_keywords({"numOfAttrs": "num_of_attrs"})
    def create_xanchors(self, num_of_attrs):
        if num_of_attrs not in self.anchor_dict:
            self.create_anchors(num_of_attrs)
        return self.anchor_dict[num_of_attrs][0]

    createXAnchors = create_xanchors

    @deprecated_keywords({"numOfAttrs": "num_of_attrs"})
    def create_yanchors(self, num_of_attrs):
        if num_of_attrs not in self.anchor_dict:
            self.create_anchors(num_of_attrs)
        return self.anchor_dict[num_of_attrs][1]

    createYAnchors = create_yanchors

    @deprecated_keywords({"numOfAttrs": "num_of_attrs"})
    def create_zanchors(self, num_of_attrs):
        if num_of_attrs not in self.anchor_dict:
            self.create_anchors(num_of_attrs)
        return self.anchor_dict[num_of_attrs][2]

    createZAnchors = create_zanchors

    @deprecated_keywords({"fileName": "filename", "attrList": "attrlist",
                          "useAnchorData": "use_anchor_data"})
    def save_projection_as_tab_data(self, filename, attrlist, use_anchor_data=0):
        """
        Save projection (xattr, yattr, zattr, classval) into a filename filename.
        
        """
        Orange.core.saveTabDelimited(filename,
                                     self.create_projection_as_example_table([self.attribute_name_index[i]
                                                                              for i in attrlist],
                                                                             use_anchor_data=use_anchor_data))

    saveProjectionAsTabData = save_projection_as_tab_data

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "useAnchorData": "use_anchor_data",
                          "xAnchors": "xanchors",
                          "yAnchors": "yanchors",
                          "zAnchors": "zanchors",
                          "anchorRadius": "anchor_radius",
                          "normalizeExample": "normalize_example",
    })
    def get_projected_point_position(self, attr_indices, values, **settings_dict):
        """
        For attributes in attr_indices and values of these attributes in values
        compute point positions. This function has more sense in radviz and
        polyviz methods.
    
        """
        # load the elements from the settings dict
        use_anchor_data = settings_dict.get("use_anchor_data")
        xanchors = settings_dict.get('xanchors')
        yanchors = settings_dict.get('yanchors')
        zanchors = settings_dict.get('zanchors')
        anchor_radius = settings_dict.get("anchor_radius")
        normalize_example = settings_dict.get("normalize_example")

        if attr_indices != self.last_attr_indices:
            print("get_projected_point_position. Warning: Possible bug. The " + \
                  "set of attributes is not the same as when computing the " + \
                  "whole projection")

        if xanchors != None and yanchors != None and zanchors != None:
            xanchors = numpy.array(xanchors)
            yanchors = numpy.array(yanchors)
            zanchors = numpy.array(zanchors)
            if anchor_radius == None: anchor_radius = numpy.sqrt(xanchors * xanchors +
                                                                 yanchors * yanchors +
                                                                 zanchors * zanchors)
        elif use_anchor_data and self.anchor_data:
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            zanchors = numpy.array([val[2] for val in self.anchor_data])
            if anchor_radius == None: anchor_radius = numpy.sqrt(xanchors * xanchors +
                                                                 yanchors * yanchors +
                                                                 zanchors * zanchors)
        else:
            self.create_anchors(len(attr_indices))
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            zanchors = numpy.array([val[2] for val in self.anchor_data])
            anchor_radius = numpy.ones(len(attr_indices), numpy.float)

        if normalize_example == 1 or (normalize_example == None
                                      and self.normalize_examples):
            m = min(values);
            M = max(values)
            if m < 0.0 or M > 1.0:
            # we have to do rescaling of values so that all the values will
                # be in the 0-1 interval
                #print "example values are not in the 0-1 interval"
                values = [max(0.0, min(val, 1.0)) for val in values]
                #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
                #values = [(val-m) / float(diff) for val in values]

            s = sum(numpy.array(values) * anchor_radius)
            if s == 0: return [0.0, 0.0]
            x = self.true_scale_factor * numpy.dot(xanchors * anchor_radius,
                                                   values) / float(s)
            y = self.true_scale_factor * numpy.dot(yanchors * anchor_radius,
                                                   values) / float(s)
            z = self.true_scale_factor * numpy.dot(zanchors * anchor_radius,
                                                   values) / float(s)
        else:
            x = self.true_scale_factor * numpy.dot(xanchors, values)
            y = self.true_scale_factor * numpy.dot(yanchors, values)
            z = self.true_scale_factor * numpy.dot(zanchors, values)

        return [x, y, z]

    getProjectedPointPosition = get_projected_point_position

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
    def create_projection_as_example_table(self, attr_indices, **settings_dict):
        """
        Create the projection of attribute indices given in attr_indices and
        create an example table with it.
        """
        if self.data_domain.class_var:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar"),
                                         Orange.feature.Continuous("zVar"),
                                         Orange.feature.Discrete(self.data_domain.class_var.name,
                                                                 values=get_variable_values_sorted(
                                                                     self.data_domain.class_var))])
        else:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar"),
                                         Orange.feature.Continuous("zVar")])
        data = self.create_projection_as_numeric_array(attr_indices,
                                                       **settings_dict)
        if data != None:
            return Orange.data.Table(domain, data)
        else:
            return Orange.data.Table(domain)

    createProjectionAsExampleTable = create_projection_as_example_table

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "validData": "valid_data",
                          "classList": "class_list",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors",
                          "ZAnchors": "zanchors",
                          "scaleFactor": "scale_factor",
                          "jitterSize": "jitter_size",
                          "useAnchorData": "use_anchor_data",
                          "removeMissingData": "remove_missing_data",
                          "useSubsetData": "use_subset_data",
    })
    def create_projection_as_numeric_array(self, attr_indices, **settings_dict):
        # load the elements from the settings dict
        valid_data = settings_dict.get("valid_data")
        class_list = settings_dict.get("class_list")
        sum_i = settings_dict.get("sum_i")
        xanchors = settings_dict.get("xanchors")
        yanchors = settings_dict.get("yanchors")
        zanchors = settings_dict.get("zanchors")
        scale_factor = settings_dict.get("scale_factor", 1.0)
        normalize = settings_dict.get("normalize")
        jitter_size = settings_dict.get("jitter_size", 0.0)
        use_anchor_data = settings_dict.get("use_anchor_data", 0)
        remove_missing_data = settings_dict.get("remove_missing_data", 1)
        use_subset_data = settings_dict.get("use_subset_data", 0)        # use the data or subsetData?
        #minmaxVals = settings_dict.get("minmaxVals", None)

        # if we want to use anchor data we can get attr_indices from the anchor_data
        if use_anchor_data and self.anchor_data:
            attr_indices = [self.attribute_name_index[val[3]] for val in self.anchor_data]

        if valid_data == None:
            if use_subset_data:
                valid_data = self.get_valid_subset_list(attr_indices)
            else:
                valid_data = self.get_valid_list(attr_indices)
        if sum(valid_data) == 0:
            return None

        if class_list == None and self.data_domain.class_var:
            if use_subset_data:
                class_list = self.original_subset_data[self.data_class_index]
            else:
                class_list = self.original_data[self.data_class_index]

        # if jitterSize is set below zero we use scaled_data that has already jittered data
        if use_subset_data:
            if jitter_size < 0.0:
                data = self.scaled_subset_data
            else:
                data = self.no_jittering_scaled_subset_data
        else:
            if jitter_size < 0.0:
                data = self.scaled_data
            else:
                data = self.no_jittering_scaled_data

        selected_data = numpy.take(data, attr_indices, axis=0)
        if remove_missing_data:
            selected_data = numpy.compress(valid_data, selected_data, axis=1)
            if class_list != None and len(class_list) != numpy.shape(selected_data)[1]:
                class_list = numpy.compress(valid_data, class_list)

        if use_anchor_data and self.anchor_data:
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            zanchors = numpy.array([val[2] for val in self.anchor_data])
            r = numpy.sqrt(
                xanchors * xanchors + yanchors * yanchors + zanchors * zanchors)     # compute the distance of each anchor from the center of the circle
            if normalize == 1 or (normalize == None and self.normalize_examples):
                xanchors *= r
                yanchors *= r
                zanchors *= r
        elif (xanchors != None and yanchors != None and zanchors != None):
            xanchors = numpy.array(xanchors)
            yanchors = numpy.array(yanchors)
            zanchors = numpy.array(zanchors)
            r = numpy.sqrt(
                xanchors * xanchors + yanchors * yanchors + zanchors * zanchors)     # compute the distance of each anchor from the center of the circle
        else:
            self.create_anchors(len(attr_indices))
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
            zanchors = numpy.array([val[2] for val in self.anchor_data])
            r = numpy.ones(len(xanchors), numpy.float)

        x_positions = numpy.dot(xanchors, selected_data)
        y_positions = numpy.dot(yanchors, selected_data)
        z_positions = numpy.dot(zanchors, selected_data)

        if normalize == 1 or (normalize == None and self.normalize_examples):
            if sum_i == None:
                sum_i = self._getSum_i(selected_data, use_anchor_data, r)
            x_positions /= sum_i
            y_positions /= sum_i
            z_positions /= sum_i
            self.true_scale_factor = scale_factor
        else:
            if not remove_missing_data:
                try:
                    x_valid_data = numpy.compress(valid_data, x_positions)
                    y_valid_data = numpy.compress(valid_data, y_positions)
                    z_valid_data = numpy.compress(valid_data, z_positions)
                except:
                    print(valid_data)
                    print(x_positions)
                    print(numpy.shape(valid_data))
                    print(numpy.shape(x_positions))
            else:
                x_valid_data = x_positions
                y_valid_data = y_positions
                z_valid_data = z_positions

            dist = math.sqrt(
                max(x_valid_data * x_valid_data + y_valid_data * y_valid_data + z_valid_data * z_valid_data)) or 1
            self.true_scale_factor = scale_factor / dist

        self.unscaled_x_positions = numpy.array(x_positions)
        self.unscaled_y_positions = numpy.array(y_positions)
        self.unscaled_z_positions = numpy.array(z_positions)

        if self.true_scale_factor != 1.0:
            x_positions *= self.true_scale_factor
            y_positions *= self.true_scale_factor
            z_positions *= self.true_scale_factor

        if jitter_size > 0.0:
            x_positions += numpy.random.uniform(-jitter_size, jitter_size, len(x_positions))
            y_positions += numpy.random.uniform(-jitter_size, jitter_size, len(y_positions))
            z_positions += numpy.random.uniform(-jitter_size, jitter_size, len(z_positions))

        self.last_attr_indices = attr_indices
        if class_list != None:
            return numpy.transpose(numpy.array((x_positions, y_positions, z_positions, class_list)))
        else:
            return numpy.transpose(numpy.array((x_positions, y_positions, z_positions)))

    createProjectionAsNumericArray = create_projection_as_numeric_array

    @deprecated_keywords({"useAnchorData": "use_anchor_data",
                          "anchorRadius": "anchor_radius"})
    def _getsum_i(self, data, use_anchor_data=0, anchor_radius=None):
        """
        Function to compute the sum of all values for each element in the data.
        Used to normalize.
        
        """
        if use_anchor_data:
            if anchor_radius == None:
                anchor_radius = numpy.sqrt([a[0] ** 2 + a[1] ** 2 + a[2] ** 2 for a in self.anchor_data])
            sum_i = numpy.add.reduce(numpy.transpose(numpy.transpose(data) * anchor_radius))
        else:
            sum_i = numpy.add.reduce(data)
        if len(numpy.nonzero(sum_i)) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += numpy.where(sum_i == 0, 1.0, 0.0)
        return sum_i

    _getSum_i = _getsum_i


ScaleLinProjData3D = deprecated_members({"setAnchors": "set_anchors",
                                         "createAnchors": "create_anchors",
                                         "saveProjectionAsTabData": "save_projection_as_tab_data",
                                         "getProjectedPointPosition":
                                             "get_projected_point_position",
                                         "createProjectionAsExampleTable":
                                             "create_projection_as_example_table",
                                         "createProjectionAsNumericArray":
                                             "create_projection_as_numeric_array",
                                         "_getSum_i": "_getsum_i",
                                         "normalizeExamples": "normalize_examples",
                                         "anchorData": "anchor_data",
                                         "lastAttrIndices": "last_attr_indices",
                                         "anchorDict": "anchor_dict",
                                         "trueScaleFactor": "true_scale_factor"
})(ScaleLinProjData3D)


class ScalePolyvizData(ScaleLinProjData):
    def __init__(self):
        ScaleLinProjData.__init__(self)
        self.normalize_examples = 1
        self.anchor_data = []        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]


    # attributeReverse, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, jitterSize = 0.0
    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
    def create_projection_as_example_table(self, attr_list, **settings_dict):
        if self.data_domain.class_var:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar"),
                                         Orange.feature.Discrete(self.data_domain.class_var.name,
                                                                 values=get_variable_values_sorted(
                                                                     self.data_domain.class_var))])
        else:
            domain = settings_dict.get("domain") or \
                     Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                         Orange.feature.Continuous("yVar")])
        data = self.create_projection_as_numeric_array(attr_list, **settings_dict)
        if data != None:
            return Orange.data.Table(domain, data)
        else:
            return Orange.data.Table(domain)

    createProjectionAsExampleTable = create_projection_as_example_table

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "validData": "valid_data",
                          "classList": "class_list",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors",
                          "scaleFactor": "scale_factor",
                          "jitterSize": "jitter_size",
                          "removeMissingData": "remove_missing_data",
    })
    def create_projection_as_numeric_array(self, attr_indices, **settings_dict):
        # load the elements from the settings dict
        attribute_reverse = settings_dict.get("reverse", [0] * len(attr_indices))
        valid_data = settings_dict.get("valid_data")
        class_list = settings_dict.get("class_list")
        sum_i = settings_dict.get("sum_i")
        xanchors = settings_dict.get("xanchors")
        yanchors = settings_dict.get("yanchors")
        scale_factor = settings_dict.get("scale_factor", 1.0)
        jitter_size = settings_dict.get("jitter_size", 0.0)
        remove_missing_data = settings_dict.get("remove_missing_data", 1)

        if valid_data == None:
            valid_data = self.get_valid_list(attr_indices)
        if sum(valid_data) == 0:
            return None

        if class_list == None and self.data_has_class:
            class_list = self.original_data[self.data_class_index]

        if remove_missing_data:
            selected_data = numpy.compress(valid_data,
                                           numpy.take(self.no_jittering_scaled_data,
                                                      attr_indices, axis=0),
                                           axis=1)
            if class_list != None and len(class_list) != numpy.shape(selected_data)[1]:
                class_list = numpy.compress(valid_data, class_list)
        else:
            selected_data = numpy.take(self.no_jittering_scaled_data,
                                       attr_indices, axis=0)

        if sum_i == None:
            sum_i = self._getSum_i(selected_data)

        if xanchors == None or yanchors == None:
            xanchors = self.create_xanchors(len(attr_indices))
            yanchors = self.create_yanchors(len(attr_indices))

        xanchors = numpy.zeros(numpy.shape(selected_data), numpy.float)
        yanchors = numpy.zeros(numpy.shape(selected_data), numpy.float)
        length = len(attr_indices)

        for i in range(length):
            if attribute_reverse[i]:
                xanchors[i] = selected_data[i] * xanchors[i] + (1 - selected_data[i]) * xanchors[(i + 1) % length]
                yanchors[i] = selected_data[i] * yanchors[i] + (1 - selected_data[i]) * yanchors[(i + 1) % length]
            else:
                xanchors[i] = (1 - selected_data[i]) * xanchors[i] + selected_data[i] * xanchors[(i + 1) % length]
                yanchors[i] = (1 - selected_data[i]) * yanchors[i] + selected_data[i] * yanchors[(i + 1) % length]

        x_positions = numpy.sum(numpy.multiply(xanchors, selected_data), axis=0) / sum_i
        y_positions = numpy.sum(numpy.multiply(yanchors, selected_data), axis=0) / sum_i
        #x_positions = numpy.sum(numpy.transpose(xanchors* numpy.transpose(selectedData)), axis=0) / sum_i
        #y_positions = numpy.sum(numpy.transpose(yanchors* numpy.transpose(selectedData)), axis=0) / sum_i

        if scale_factor != 1.0:
            x_positions = x_positions * scale_factor
            y_positions = y_positions * scale_factor
        if jitter_size > 0.0:
            x_positions += (numpy.random.random(len(x_positions)) - 0.5) * jitter_size
            y_positions += (numpy.random.random(len(y_positions)) - 0.5) * jitter_size

        if class_list != None:
            return numpy.transpose(numpy.array((x_positions, y_positions, class_list)))
        else:
            return numpy.transpose(numpy.array((x_positions, y_positions)))

    createProjectionAsNumericArray = create_projection_as_numeric_array

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "useAnchorData": "use_anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def get_projected_point_position(self, attr_indices, values, **settings_dict):
        # load the elements from the settings dict
        attribute_reverse = settings_dict.get("reverse", [0] * len(attr_indices))
        use_anchor_data = settings_dict.get("use_anchor_data")
        xanchors = settings_dict.get("xanchors")
        yanchors = settings_dict.get("yanchors")

        if xanchors != None and yanchors != None:
            xanchors = numpy.array(xanchors)
            yanchors = numpy.array(yanchors)
        elif use_anchor_data:
            xanchors = numpy.array([val[0] for val in self.anchor_data])
            yanchors = numpy.array([val[1] for val in self.anchor_data])
        else:
            xanchors = self.create_xanchors(len(attr_indices))
            yanchors = self.create_yanchors(len(attr_indices))

        m = min(values);
        M = max(values)
        if m < 0.0 or M > 1.0:  # we have to do rescaling of values so that all
            # the values will be in the 0-1 interval
            values = [max(0.0, min(val, 1.0)) for val in values]
            #m = min(m, 0.0); M = max(M, 1.0); diff = max(M-m, 1e-10)
            #values = [(val-m) / float(diff) for val in values]

        s = sum(numpy.array(values))
        if s == 0: return [0.0, 0.0]

        length = len(values)
        xanchors = numpy.zeros(length, numpy.float)
        yanchors = numpy.zeros(length, numpy.float)
        for i in range(length):
            if attribute_reverse[i]:
                xanchors[i] = values[i] * xanchors[i] + (1 - values[i]) * xanchors[(i + 1) % length]
                yanchors[i] = values[i] * yanchors[i] + (1 - values[i]) * yanchors[(i + 1) % length]
            else:
                xanchors[i] = (1 - values[i]) * xanchors[i] + values[i] * xanchors[(i + 1) % length]
                yanchors[i] = (1 - values[i]) * yanchors[i] + values[i] * yanchors[(i + 1) % length]

        x_positions = numpy.sum(numpy.dot(xanchors, values), axis=0) / float(s)
        y_positions = numpy.sum(numpy.dot(yanchors, values), axis=0) / float(s)
        return [x, y]

    getProjectedPointPosition = get_projected_point_position


ScalePolyvizData = deprecated_members({"createProjectionAsExampleTable":
                                           "create_projection_as_example_table",
                                       "createProjectionAsNumericArray":
                                           "create_projection_as_numeric_array",
                                       "getProjectedPointPosition":
                                           "get_projected_point_position"
})(ScalePolyvizData)


class ScaleScatterPlotData(ScaleData):
    def get_original_data(self, indices):
        data = self.original_data.take(indices, axis=0)
        for i, ind in enumerate(indices):
            [minVal, maxVal] = self.attr_values[self.data_domain[ind].name]
            if self.data_domain[ind].varType == Orange.core.VarTypes.Discrete:
                data[i] += (self.jitter_size / 50.0) * (numpy.random.random(len(self.raw_data)) - 0.5)
            elif self.data_domain[ind].varType == Orange.core.VarTypes.Continuous and self.jitter_continuous:
                data[i] += (self.jitter_size / (50.0 * (maxVal - minVal or 1))) * (
                    numpy.random.random(len(self.raw_data)) - 0.5)
        return data

    getOriginalData = get_original_data

    def get_original_subset_data(self, indices):
        data = self.original_subset_data.take(indices, axis=0)
        for i, ind in enumerate(indices):
            [minVal, maxVal] = self.attr_values[self.raw_subset_data.domain[ind].name]
            if self.data_domain[ind].varType == Orange.core.VarTypes.Discrete:
                data[i] += (self.jitter_size / (50.0 * max(1, maxVal))) * (
                    numpy.random.random(len(self.raw_subset_data)) - 0.5)
            elif self.data_domain[ind].varType == Orange.core.VarTypes.Continuous and self.jitter_continuous:
                data[i] += (self.jitter_size / (50.0 * (maxVal - minVal or 1))) * (
                    numpy.random.random(len(self.raw_subset_data)) - 0.5)
        return data

    getOriginalSubsetData = get_original_subset_data

    @deprecated_keywords({"xAttr": "xattr", "yAttr": "yattr"})
    def get_xy_data_positions(self, xattr, yattr):
        """
        Create x-y projection of attributes in attrlist.
        
        """
        xattr_index, yattr_index = self.attribute_name_index[xattr], self.attribute_name_index[yattr]

        xdata = self.scaled_data[xattr_index].copy()
        ydata = self.scaled_data[yattr_index].copy()

        if self.data_domain[xattr_index].varType == Orange.core.VarTypes.Discrete:
            xdata = ((xdata * 2 * len(self.data_domain[xattr_index].values)) - 1.0) / 2.0
        else:
            xdata = xdata * (self.attr_values[xattr][1] - self.attr_values[xattr][0]) + float(
                self.attr_values[xattr][0])

        if self.data_domain[yattr_index].varType == Orange.core.VarTypes.Discrete:
            ydata = ((ydata * 2 * len(self.data_domain[yattr_index].values)) - 1.0) / 2.0
        else:
            ydata = ydata * (self.attr_values[yattr][1] - self.attr_values[yattr][0]) + float(
                self.attr_values[yattr][0])
        return (xdata, ydata)

    getXYDataPositions = get_xy_data_positions

    @deprecated_keywords({"xAttr": "xattr", "yAttr": "yattr"})
    def get_xy_subset_data_positions(self, xattr, yattr):
        """
        Create x-y projection of attributes in attr_list.
        
        """
        xattr_index, yattr_index = self.attribute_name_index[xattr], self.attribute_name_index[yattr]

        xdata = self.scaled_subset_data[xattr_index].copy()
        ydata = self.scaled_subset_data[yattr_index].copy()

        if self.data_domain[xattr_index].varType == Orange.core.VarTypes.Discrete:
            xdata = ((xdata * 2 * len(self.data_domain[xattr_index].values)) - 1.0) / 2.0
        else:
            xdata = xdata * (self.attr_values[xattr][1] - self.attr_values[xattr][0]) + float(
                self.attr_values[xattr][0])

        if self.data_domain[yattr_index].varType == Orange.core.VarTypes.Discrete:
            ydata = ((ydata * 2 * len(self.data_domain[yattr_index].values)) - 1.0) / 2.0
        else:
            ydata = ydata * (self.attr_values[yattr][1] - self.attr_values[yattr][0]) + float(
                self.attr_values[yattr][0])
        return (xdata, ydata)

    getXYSubsetDataPositions = get_xy_subset_data_positions

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
    def get_projected_point_position(self, attr_indices, values, **settings_dict):
        """
        For attributes in attr_indices and values of these attributes in values
        compute point positions this function has more sense in radviz and
        polyviz methods. settings_dict has to be because radviz and polyviz have
        this parameter.
        """
        return values

    getProjectedPointPosition = get_projected_point_position

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
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
                                                                 values=get_variable_values_sorted(
                                                                     self.data_domain.class_var))])
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

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict"})
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
                                                                 values=get_variable_values_sorted(
                                                                     self.data_domain.class_var))])
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

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "validData": "valid_data",
                          "classList": "class_list",
                          "jutterSize": "jitter_size"})
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
            xarray += (numpy.random.random(len(xarray)) - 0.5) * jitter_size
            yarray += (numpy.random.random(len(yarray)) - 0.5) * jitter_size
        if class_list != None:
            data = numpy.compress(valid_data, numpy.array((xarray, yarray, class_list)), axis=1)
        else:
            data = numpy.compress(valid_data, numpy.array((xarray, yarray)), axis=1)
        data = numpy.transpose(data)
        return data

    createProjectionAsNumericArray = create_projection_as_numeric_array

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "settingsDict": "settings_dict",
                          "validData": "valid_data",
                          "classList": "class_list",
                          "jutterSize": "jitter_size"})
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
            xarray += (numpy.random.random(len(xarray)) - 0.5) * jitter_size
            yarray += (numpy.random.random(len(yarray)) - 0.5) * jitter_size
            zarray += (numpy.random.random(len(zarray)) - 0.5) * jitter_size
        if class_list != None:
            data = numpy.compress(valid_data, numpy.array((xarray, yarray, zarray, class_list)), axis=1)
        else:
            data = numpy.compress(valid_data, numpy.array((xarray, yarray, zarray)), axis=1)
        data = numpy.transpose(data)
        return data

    createProjectionAsNumericArray3D = create_projection_as_numeric_array_3D

    @deprecated_keywords({"attributeNameOrder": "attribute_name_order",
                          "addResultFunct": "add_result_funct"})
    def get_optimal_clusters(self, attribute_name_order, add_result_funct):
        if not self.data_has_class or self.data_has_continuous_class:
            return

        jitter_size = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation
        domain = Orange.data.Domain([Orange.feature.Continuous("xVar"),
                                     Orange.feature.Continuous("yVar"),
                                     self.data_domain.class_var])

        # init again, in case that the attribute ordering took too much time
        self.scatterWidget.progressBarInit()
        start_time = time.time()
        count = len(attribute_name_order) * (len(attribute_name_order) - 1) / 2
        test_index = 0

        for i in range(len(attribute_name_order)):
            for j in range(i):
                try:
                    attr1 = self.attribute_name_index[attribute_name_order[j]]
                    attr2 = self.attribute_name_index[attribute_name_order[i]]
                    test_index += 1
                    if self.clusterOptimization.isOptimizationCanceled():
                        secs = time.time() - start_time
                        self.clusterOptimization.setStatusBarText(
                            "Evaluation stopped (evaluated %d projections in %d min, %d sec)"
                            % (test_index, secs / 60, secs % 60))
                        self.scatterWidget.progressBarFinished()
                        return

                    data = self.create_projection_as_example_table([attr1, attr2],
                                                                   domain=domain,
                                                                   jitter_size=jitter_size)
                    graph, valuedict, closuredict, polygon_vertices_dict, enlarged_closure_dict, other_dict = self.clusterOptimization.evaluateClusters(
                        data)

                    all_value = 0.0
                    classes_dict = {}
                    for key in list(valuedict.keys()):
                        add_result_funct(valuedict[key], closuredict[key],
                                         polygon_vertices_dict[key],
                                         [attribute_name_order[i],
                                          attribute_name_order[j]],
                                         int(graph.objects[polygon_vertices_dict[key][0]].getclass()),
                                         enlarged_closure_dict[key], other_dict[key])
                        classes_dict[key] = int(graph.objects[polygon_vertices_dict[key][0]].getclass())
                        all_value += valuedict[key]
                    add_result_funct(all_value, closuredict, polygon_vertices_dict,
                                     [attribute_name_order[i], attribute_name_order[j]],
                                     classes_dict, enlarged_closure_dict, other_dict)     # add all the clusters

                    self.clusterOptimization.setStatusBarText("Evaluated %d projections..."
                                                              % (test_index))
                    self.scatterWidget.progressBarSet(100.0 * test_index / float(count))
                    del data, graph, valuedict, closuredict, polygon_vertices_dict, enlarged_closure_dict, other_dict, classes_dict
                except:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # print the exception

        secs = time.time() - start_time
        self.clusterOptimization.setStatusBarText(
            "Finished evaluation (evaluated %d projections in %d min, %d sec)" % (test_index, secs / 60, secs % 60))
        self.scatterWidget.progressBarFinished()

    getOptimalClusters = get_optimal_clusters


ScaleScatterPlotData = deprecated_members({"getOriginalData": "get_original_data",
                                           "getOriginalSubsetData": "get_original_subset_data",
                                           "getXYDataPositions": "get_xy_data_positions",
                                           "getXYSubsetDataPositions": "get_xy_subset_data_positions",
                                           "getProjectedPointPosition": "get_projected_point_position",
                                           "createProjectionAsExampleTable": "create_projection_as_example_table",
                                           "createProjectionAsExampleTable3D": "create_projection_as_example_table_3D",
                                           "createProjectionAsNumericArray": "create_projection_as_numeric_array",
                                           "createProjectionAsNumericArray3D": "create_projection_as_numeric_array_3D",
                                           "getOptimalClusters": "get_optimal_clusters"
})(ScaleScatterPlotData)
