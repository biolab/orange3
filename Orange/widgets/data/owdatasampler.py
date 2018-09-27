import sys
import math

from AnyQt.QtWidgets import QFormLayout, QApplication
from AnyQt.QtCore import Qt

import numpy as np
import sklearn.model_selection as skl

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.data import Table
from Orange.data.sql.table import SqlTable
from Orange.widgets.widget import Msg, OWWidget, Input, Output
from Orange.util import Reprable


class OWDataSampler(OWWidget):
    name = "Data Sampler"
    description = "Randomly draw a subset of data points " \
                  "from the input dataset."
    icon = "icons/DataSampler.svg"
    priority = 100
    category = "Data"
    keywords = ["random"]

    _MAX_SAMPLE_SIZE = 2 ** 31 - 1

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data_sample = Output("Data Sample", Table, default=True)
        remaining_data = Output("Remaining Data", Table)

    want_main_area = False
    resizing_enabled = False

    RandomSeed = 42
    FixedProportion, FixedSize, CrossValidation, Bootstrap, Oversample = range(5)
    SqlTime, SqlProportion = range(2)
    MethodSMOTE = 0

    use_seed = Setting(False)
    replacement = Setting(False)
    stratify = Setting(False)
    sql_dl = Setting(False)
    sampling_type = Setting(FixedProportion)
    sampleSizeNumber = Setting(1)
    sampleSizePercentage = Setting(70)
    sampleSizeSqlTime = Setting(1)
    sampleSizeSqlPercentage = Setting(0.1)
    number_of_folds = Setting(10)
    selectedFold = Setting(1)
    oversampling_factor = Setting(1)
    oversampling_method = Setting(MethodSMOTE)
    oversampling_k = Setting(5)
    min_class = Setting(0)

    class Warning(OWWidget.Warning):
        could_not_stratify = Msg("Stratification failed\n{}")
        bigger_sample = Msg('Sample is bigger than input')

    class Error(OWWidget.Error):
        too_many_folds = Msg("Number of folds exceeds data size")
        sample_larger_than_data = Msg("Sample must be smaller than data")
        not_enough_to_stratify = Msg("Data is too small to stratify")
        no_data = Msg("Dataset is empty")
        too_many_neighbors = Msg("Number of neighbors considered must be lower than " 
                                 "number of all examples")
        invalid_target = Msg("Target variable is either undefined or non-categorical")

    def __init__(self):
        super().__init__()
        self.data = None
        self.indices = None
        self.sampled_instances = self.remaining_instances = None

        box = gui.vBox(self.controlArea, "Information")
        self.dataInfoLabel = gui.widgetLabel(box, 'No data on input.')
        self.outputInfoLabel = gui.widgetLabel(box, ' ')

        self.sampling_box = gui.vBox(self.controlArea, "Sampling Type")
        sampling = gui.radioButtons(self.sampling_box, self, "sampling_type",
                                    callback=self.sampling_type_changed)

        def set_sampling_type(i):
            def set_sampling_type_i():
                self.sampling_type = i
                self.sampling_type_changed()
            return set_sampling_type_i

        gui.appendRadioButton(sampling, "Fixed proportion of data:")
        self.sampleSizePercentageSlider = gui.hSlider(
            gui.indentedBox(sampling), self,
            "sampleSizePercentage",
            minValue=0, maxValue=99, ticks=10, labelFormat="%d %%",
            callback=set_sampling_type(self.FixedProportion),
            addSpace=12)

        gui.appendRadioButton(sampling, "Fixed sample size")
        ibox = gui.indentedBox(sampling)
        self.sampleSizeSpin = gui.spin(
            ibox, self, "sampleSizeNumber", label="Instances: ",
            minv=1, maxv=self._MAX_SAMPLE_SIZE,
            callback=set_sampling_type(self.FixedSize),
            controlWidth=90)
        gui.checkBox(
            ibox, self, "replacement", "Sample with replacement",
            callback=set_sampling_type(self.FixedSize),
            addSpace=12)

        gui.appendRadioButton(sampling, "Cross validation")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft | Qt.AlignTop,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        ibox = gui.indentedBox(sampling, addSpace=True, orientation=form)
        form.addRow("Number of folds:",
                    gui.spin(
                        ibox, self, "number_of_folds", 2, 100,
                        addToLayout=False,
                        callback=self.number_of_folds_changed))
        self.selected_fold_spin = gui.spin(
            ibox, self, "selectedFold", 1, self.number_of_folds,
            addToLayout=False, callback=self.fold_changed)
        form.addRow("Selected fold:", self.selected_fold_spin)

        gui.appendRadioButton(sampling, "Bootstrap")

        gui.appendRadioButton(sampling, "Oversample")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft | Qt.AlignTop,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        ibox = gui.indentedBox(sampling, orientation=form)
        form.addRow("Method:",
                    gui.comboBox(ibox, self, "oversampling_method",
                                 items=["SMOTE"],
                                 addToLayout=False,
                                 callback=set_sampling_type(self.Oversample)))
        self.min_class_combo = gui.comboBox(ibox, self, "min_class",
                                            addToLayout=False,
                                            callback=set_sampling_type(self.Oversample))
        form.addRow("Minority class:", self.min_class_combo)
        form.addRow("Oversampling factor:",
                    gui.spin(
                        ibox, self, "oversampling_factor",
                        minv=1, maxv=100, step=1,
                        addToLayout=False,
                        callback=set_sampling_type(self.Oversample)))
        form.addRow("Nearest neighbors:",
                    gui.spin(
                        ibox, self, "oversampling_k",
                        minv=1, maxv=10, step=1,
                        addToLayout=False,
                        callback=set_sampling_type(self.Oversample)))

        self.sql_box = gui.vBox(self.controlArea, "Sampling Type")
        sampling = gui.radioButtons(self.sql_box, self, "sampling_type",
                                    callback=self.sampling_type_changed)
        gui.appendRadioButton(sampling, "Time:")
        ibox = gui.indentedBox(sampling)
        spin = gui.spin(ibox, self, "sampleSizeSqlTime", minv=1, maxv=3600,
                        callback=set_sampling_type(self.SqlTime))
        spin.setSuffix(" sec")
        gui.appendRadioButton(sampling, "Percentage")
        ibox = gui.indentedBox(sampling)
        spin = gui.spin(ibox, self, "sampleSizeSqlPercentage", spinType=float,
                        minv=0.0001, maxv=100, step=0.1, decimals=4,
                        callback=set_sampling_type(self.SqlProportion))
        spin.setSuffix(" %")
        self.sql_box.setVisible(False)

        self.options_box = gui.vBox(self.controlArea, "Options")
        self.cb_seed = gui.checkBox(
            self.options_box, self, "use_seed",
            "Replicable (deterministic) sampling",
            callback=self.settings_changed)
        self.cb_stratify = gui.checkBox(
            self.options_box, self, "stratify",
            "Stratify sample (when possible)", callback=self.settings_changed)
        self.cb_sql_dl = gui.checkBox(
            self.options_box, self, "sql_dl", "Download data to local memory",
            callback=self.settings_changed)
        self.cb_sql_dl.setVisible(False)

        gui.button(self.buttonsArea, self, "Sample Data",
                   callback=self.commit)

    def sampling_type_changed(self):
        self.settings_changed()

    def number_of_folds_changed(self):
        self.selected_fold_spin.setMaximum(self.number_of_folds)
        self.sampling_type = self.CrossValidation
        self.settings_changed()

    def fold_changed(self):
        # a separate callback - if we decide to cache indices
        self.sampling_type = self.CrossValidation

    def settings_changed(self):
        self._update_sample_max_size()
        self.indices = None

    @Inputs.data
    def set_data(self, dataset):
        self.data = dataset
        if dataset is not None:
            sql = isinstance(dataset, SqlTable)
            self.sampling_box.setVisible(not sql)
            self.sql_box.setVisible(sql)
            self.cb_seed.setVisible(not sql)
            self.cb_stratify.setVisible(not sql)
            self.cb_sql_dl.setVisible(sql)
            self.dataInfoLabel.setText(
                '{}{} instances in input dataset.'.format(*(
                    ('~', dataset.approx_len()) if sql else
                    ('', len(dataset)))))
            if not sql:
                self.min_class_combo.clear()
                if self.data.domain.has_discrete_class:
                    self.min_class_combo.addItems(self.data.domain.class_var.values)
                self.min_class = 0
                self._update_sample_max_size()
                self.updateindices()
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.outputInfoLabel.setText('')
            self.min_class_combo.clear()
            self.indices = None
            self.clear_messages()
        self.commit()

    def _update_sample_max_size(self):
        """Limit number of instances to input size unless using replacement."""
        if not self.data or self.replacement:
            self.sampleSizeSpin.setMaximum(self._MAX_SAMPLE_SIZE)
        else:
            self.sampleSizeSpin.setMaximum(len(self.data))

    def commit(self):
        if self.data is None:
            sample = other = None
            self.sampled_instances = self.remaining_instances = None
            self.outputInfoLabel.setText("")
        elif isinstance(self.data, SqlTable):
            other = None
            if self.sampling_type == self.SqlProportion:
                sample = self.data.sample_percentage(
                    self.sampleSizeSqlPercentage, no_cache=True)
            else:
                sample = self.data.sample_time(
                    self.sampleSizeSqlTime, no_cache=True)
            if self.sql_dl:
                sample.download_data()
                sample = Table(sample)

        else:
            if self.indices is None or not self.use_seed:
                self.updateindices()
                if self.indices is None:
                    return
            if self.sampling_type in (
                    self.FixedProportion, self.FixedSize, self.Bootstrap):
                remaining, sample = self.indices
                self.outputInfoLabel.setText(
                    'Outputting %d instance%s.' %
                    (len(sample), "s" * (len(sample) != 1)))
            elif self.sampling_type == self.CrossValidation:
                remaining, sample = self.indices[self.selectedFold - 1]
                self.outputInfoLabel.setText(
                    'Outputting fold %d, %d instance%s.' %
                    (self.selectedFold, len(sample), "s" * (len(sample) != 1))
                )
            elif self.sampling_type == self.Oversample:
                self.outputInfoLabel.setText(
                    'Outputting %d instance%s (%d new)' %
                    (len(self.indices), "s" * (len(self.indices) != 1),
                     len(self.indices) - len(self.data))
                )

            # Oversampling produces new examples, while other types of sampling
            # return indices to use with existing examples
            if self.sampling_type == self.Oversample:
                sample = self.indices
                other = None

                self.sampled_instances = len(sample)
                self.remaining_instances = 0
            else:
                sample = self.data[sample]
                other = self.data[remaining]
                self.sampled_instances = len(sample)
                self.remaining_instances = len(other)

        self.Outputs.data_sample.send(sample)
        self.Outputs.remaining_data.send(other)

    def updateindices(self):
        self.Error.clear()
        self.Warning.clear()
        repl = True
        data_length = len(self.data)
        num_classes = len(self.data.domain.class_var.values) \
            if self.data.domain.has_discrete_class else 0

        size = None
        if not data_length:
            self.Error.no_data()
        elif self.sampling_type == self.FixedSize:
            size = self.sampleSizeNumber
            repl = self.replacement
        elif self.sampling_type == self.FixedProportion:
            size = np.ceil(self.sampleSizePercentage / 100 * data_length)
            repl = False
        elif self.sampling_type == self.CrossValidation:
            if data_length < self.number_of_folds:
                self.Error.too_many_folds()
        elif self.sampling_type == self.Oversample:
            if not self.data.domain.has_discrete_class:
                self.Error.invalid_target()
            elif data_length <= self.oversampling_k:
                self.Error.too_many_neighbors()
        else:
            assert self.sampling_type == self.Bootstrap

        if not repl and size is not None and (data_length <= size):
            self.Error.sample_larger_than_data()
        if not repl and data_length <= num_classes and self.stratify:
            self.Error.not_enough_to_stratify()

        if self.Error.active:
            self.indices = None
            return

        # By the above, we can safely assume there is data
        if self.sampling_type == self.FixedSize and repl and size and \
                size > len(self.data):
            # This should only be possible when using replacement
            self.Warning.bigger_sample()

        stratified = (self.stratify and
                      isinstance(self.data, Table) and
                      self.data.domain.has_discrete_class)
        try:
            self.indices = self.sample(data_length, size, stratified)
        except ValueError as ex:
            self.Warning.could_not_stratify(str(ex))
            self.indices = self.sample(data_length, size, stratified=False)

    def sample(self, data_length, size, stratified):
        rnd = self.RandomSeed if self.use_seed else None
        if self.sampling_type == self.FixedSize:
            self.indice_gen = SampleRandomN(
                size, stratified=stratified, replace=self.replacement,
                random_state=rnd)
        elif self.sampling_type == self.FixedProportion:
            self.indice_gen = SampleRandomP(
                self.sampleSizePercentage / 100, stratified=stratified,
                random_state=rnd)
        elif self.sampling_type == self.Bootstrap:
            self.indice_gen = SampleBootstrap(data_length, random_state=rnd)
        elif self.sampling_type == self.Oversample:
            if self.oversampling_method == self.MethodSMOTE:
                self.indice_gen = OversampleSMOTE(self.min_class, self.oversampling_factor,
                                                  self.oversampling_k, random_state=rnd)
            else:
                return None  # should not come here at all
        else:
            self.indice_gen = SampleFoldIndices(
                self.number_of_folds, stratified=stratified, random_state=rnd)
        return self.indice_gen(self.data)

    def send_report(self):
        if self.sampling_type == self.FixedProportion:
            tpe = "Random sample with {} % of data".format(
                self.sampleSizePercentage)
        elif self.sampling_type == self.FixedSize:
            if self.sampleSizeNumber == 1:
                tpe = "Random data instance"
            else:
                tpe = "Random sample with {} data instances".format(
                    self.sampleSizeNumber)
                if self.replacement:
                    tpe += ", with replacement"
        elif self.sampling_type == self.CrossValidation:
            tpe = "Fold {} of {}-fold cross-validation".format(
                self.selectedFold, self.number_of_folds)
        elif self.sampling_type == self.Oversample:
            if self.data.domain.has_discrete_class:
                tpe = "Data set with class '{}' oversampled at {}%".format(
                    self.data.domain.class_var.values[self.min_class],
                    int(self.oversampling_factor * 100))
            else:
                tpe = "N/A (unsuccessful)"
        else:
            tpe = "Undefined"  # should not come here at all
        if self.stratify:
            tpe += ", stratified (if possible)"
        if self.use_seed:
            tpe += ", deterministic"
        items = [("Sampling type", tpe)]
        if self.sampled_instances is not None:
            items += [
                ("Input", "{} instances".format(len(self.data))),
                ("Sample", "{} instances".format(self.sampled_instances)),
                ("Remaining", "{} instances".format(self.remaining_instances)),
            ]
        self.report_items(items)


class SampleFoldIndices(Reprable):
    def __init__(self, folds=10, stratified=False, random_state=None):
        """Samples data based on a number of folds.

        Args:
            folds (int): Number of folds
            stratified (bool): Return stratified indices (if applicable).
            random_state (Random): An initial state for replicable random
            behavior

        Returns:
            tuple-of-arrays: A tuple of array indices one for each fold.

        """
        self.folds = folds
        self.stratified = stratified
        self.random_state = random_state

    def __call__(self, table):
        if self.stratified and table.domain.has_discrete_class:
            splitter = skl.StratifiedKFold(
                self.folds, random_state=self.random_state)
            splitter.get_n_splits(table.X, table.Y)
            ind = splitter.split(table.X, table.Y)
        else:
            splitter = skl.KFold(
                self.folds, random_state=self.random_state)
            splitter.get_n_splits(table)
            ind = splitter.split(table)
        return tuple(ind)


class SampleRandomN(Reprable):
    def __init__(self, n=0, stratified=False, replace=False,
                 random_state=None):
        self.n = n
        self.stratified = stratified
        self.replace = replace
        self.random_state = random_state

    def __call__(self, table):
        if self.replace:
            rgen = np.random.RandomState(self.random_state)
            sample = rgen.random_integers(0, len(table) - 1, self.n)
            o = np.ones(len(table))
            o[sample] = 0
            others = np.nonzero(o)[0]
            return others, sample
        if self.stratified and table.domain.has_discrete_class:
            test_size = max(len(table.domain.class_var.values), self.n)
            splitter = skl.StratifiedShuffleSplit(
                n_splits=1, test_size=test_size,
                train_size=len(table) - test_size,
                random_state=self.random_state)
            splitter.get_n_splits(table.X, table.Y)
            ind = splitter.split(table.X, table.Y)
        else:
            splitter = skl.ShuffleSplit(
                n_splits=1, test_size=self.n, random_state=self.random_state)
            splitter.get_n_splits(table)
            ind = splitter.split(table)
        return next(iter(ind))


class SampleRandomP(Reprable):
    def __init__(self, p=0, stratified=False, random_state=None):
        self.p = p
        self.stratified = stratified
        self.random_state = random_state

    def __call__(self, table):
        n = int(math.ceil(len(table) * self.p))
        return SampleRandomN(n, self.stratified,
                             random_state=self.random_state)(table)


class SampleBootstrap(Reprable):
    def __init__(self, size=0, random_state=None):
        self.size = size
        self.random_state = random_state

    def __call__(self, table=None):
        """Bootstrap indices

        Args:
            table: Not used (but part of the signature)
        Returns:
            tuple (out_of_sample, sample) indices
        """
        rgen = np.random.RandomState(self.random_state)
        sample = rgen.randint(0, self.size, self.size)
        sample.sort()  # not needed for the code below, just for the user
        insample = np.ones((self.size,), dtype=np.bool)
        insample[sample] = False
        remaining = np.flatnonzero(insample)
        return remaining, sample


class OversampleSMOTE(Reprable):
    def __init__(self, min_class, over_factor, k, random_state=None):
        """ Oversamples the minority class by a factor, determined by `over_factor`, using
        SMOTE (synthetic minority oversampling technique).

        Args:
            min_class (float): Minority class
            over_factor (float): Oversampling factor
            k (int): number of neighbors taken into account when generating new examples
            random_state (Random): An initial state for replicable random behaviour
        """
        self.min_class = min_class
        self.over_factor = int(over_factor)
        self.k = k
        self.random_state = random_state

    def gen_disc(self, column):
        vals, counts = np.unique(column, return_counts=True)
        most_freq = vals[np.argmax(counts)]
        return np.repeat(most_freq, repeats=self.over_factor)

    def __call__(self, table):
        cont_attrs = np.array([True if attr.is_continuous else False
                               for attr in table.domain.attributes])
        othr_attrs = np.logical_not(cont_attrs)
        min_examples = table[table.Y == self.min_class]
        n_metas = len(table.domain.metas)
        n_examples = len(table)

        if n_examples <= self.k:
            raise ValueError("Number of neighbors considered must be lower than "
                             "number of all examples")

        # note: these do not contain target variable(s)
        min_examples_cont = min_examples.X[:, cont_attrs]
        min_examples_othr = min_examples.X[:, othr_attrs]

        reps = np.histogram(np.arange(self.over_factor), bins=self.k)[0]
        sq_med_sd = np.square(np.median(np.std(min_examples_cont, axis=0)))
        rgen = np.random.RandomState(self.random_state)
        new_table = table.copy()

        for idx_example in range(len(min_examples)):
            dists = np.sqrt(np.sum(np.square(
                min_examples_cont[idx_example, :] - table.X[:, cont_attrs]), axis=1))

            n_diffs = np.sum(min_examples_othr[idx_example, :] != table.X[:, othr_attrs],
                             axis=1)
            dists += n_diffs * sq_med_sd

            # closest example to current example will always be the example itself
            # (or an identical example)
            nneigh = np.argsort(dists)[1: self.k + 1]
            rgen.shuffle(nneigh)

            # if self.over_factor > self.k, some neighbors will be taken multiple times
            nneigh_repeated = np.repeat(nneigh, repeats=reps, axis=0)

            new_data = []
            new_metas = [self.gen_disc(table.metas[nneigh, idx_meta])
                         for idx_meta in range(n_metas)]
            new_labels = np.repeat(self.min_class, repeats=self.over_factor)

            for idx_attr in range(len(table.domain.attributes)):
                if cont_attrs[idx_attr]:
                    # new attr = orig. attr + rand(0, 1) * diff
                    diff = table.X[nneigh_repeated, idx_attr] - \
                           min_examples.X[idx_example, idx_attr]

                    new_attr = (min_examples.X[idx_example, idx_attr] +
                                rgen.rand(self.over_factor) * diff)
                else:
                    # new attr = most frequent value among neighbors
                    new_attr = self.gen_disc(table.X[nneigh, idx_attr])

                new_data.append(new_attr)

            new_data = np.column_stack(new_data)
            new_table.extend(np.column_stack((new_data, new_labels)))

            if n_metas > 0:
                new_metas = np.column_stack(new_metas)
                new_table.metas[-self.over_factor:] = new_metas

        return new_table


def test_main():
    app = QApplication([])
    data = Table("iris")
    w = OWDataSampler()
    w.set_data(data)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(test_main())
