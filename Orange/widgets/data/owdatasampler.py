import math

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

import numpy as np
import sklearn.model_selection as skl

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.data import Table
from Orange.data.sql.table import SqlTable
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, OWWidget, Input, Output
from Orange.util import Reprable


class OWDataSampler(OWWidget):
    name = "Data Sampler"
    description = "Randomly draw a subset of data points " \
                  "from the input dataset."
    icon = "icons/DataSampler.svg"
    priority = 100
    category = "Transform"
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
    FixedProportion, FixedSize, CrossValidation, Bootstrap = range(4)
    SqlTime, SqlProportion = range(2)

    selectedFold: int

    use_seed = Setting(True)
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

    # Older versions of the widget had swapped outputs for cross validation
    # Migrations set this to True for compability with older workflows
    compatibility_mode = Setting(False, schema_only=True)

    settings_version = 2

    class Information(OWWidget.Information):
        compatibility_mode = Msg(
            "Compatibility mode\n"
            "New versions of widget have swapped outputs for cross validation"
        )

    class Warning(OWWidget.Warning):
        could_not_stratify = Msg("Stratification failed\n{}")
        bigger_sample = Msg('Sample is bigger than input')

    class Error(OWWidget.Error):
        too_many_folds = Msg("Number of subsets exceeds data size")
        sample_larger_than_data = Msg("Sample can't be larger than data")
        not_enough_to_stratify = Msg("Data is too small to stratify")
        no_data = Msg("Dataset is empty")

    def __init__(self):
        super().__init__()
        if self.compatibility_mode:
            self.Information.compatibility_mode()

        self.data = None
        self.indices = None
        self.sampled_instances = self.remaining_instances = None

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
            minValue=0, maxValue=100, ticks=10, labelFormat="%d %%",
            callback=set_sampling_type(self.FixedProportion))

        gui.appendRadioButton(sampling, "Fixed sample size")
        ibox = gui.indentedBox(sampling)
        self.sampleSizeSpin = gui.spin(
            ibox, self, "sampleSizeNumber", label="Instances: ",
            minv=1, maxv=self._MAX_SAMPLE_SIZE,
            callback=set_sampling_type(self.FixedSize),
            controlWidth=90)
        gui.checkBox(
            ibox, self, "replacement", "Sample with replacement",
            callback=set_sampling_type(self.FixedSize))

        gui.appendRadioButton(sampling, "Cross validation")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft | Qt.AlignTop,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        ibox = gui.indentedBox(sampling, orientation=form)
        form.addRow("Number of subsets:",
                    gui.spin(
                        ibox, self, "number_of_folds", 2, 100,
                        addToLayout=False,
                        callback=self.number_of_folds_changed))
        self.selected_fold_spin = gui.spin(
            ibox, self, "selectedFold", 1, self.number_of_folds,
            addToLayout=False, callback=self.fold_changed)
        form.addRow("Unused subset:" if not self.compatibility_mode
                    else "Selected subset:", self.selected_fold_spin)

        gui.appendRadioButton(sampling, "Bootstrap")

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

        self.options_box = gui.vBox(self.controlArea, "Options", addSpaceBefore=False)
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

            if not sql:
                self._update_sample_max_size()
                self.updateindices()
        else:
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
            elif self.sampling_type == self.CrossValidation:
                if self.compatibility_mode:
                    remaining, sample = self.indices[self.selectedFold - 1]
                else:
                    sample, remaining = self.indices[self.selectedFold - 1]

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
        else:
            assert self.sampling_type == self.Bootstrap

        if not repl and size is not None and (size > data_length):
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
            sampler = SampleRandomN(
                size, stratified=stratified, replace=self.replacement,
                random_state=rnd)
        elif self.sampling_type == self.FixedProportion:
            sampler = SampleRandomP(
                self.sampleSizePercentage / 100, stratified=stratified,
                random_state=rnd)
        elif self.sampling_type == self.Bootstrap:
            sampler = SampleBootstrap(data_length, random_state=rnd)
        else:
            sampler = SampleFoldIndices(
                self.number_of_folds, stratified=stratified, random_state=rnd)
        return sampler(self.data)

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
            tpe = f"{self.number_of_folds}-fold cross-validation " \
                  f"without subset #{self.selectedFold}"
        elif self.sampling_type == self.Bootstrap:
            tpe = "Bootstrap"
        else:  # pragma: no cover
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

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version or version < 2 \
                and settings["sampling_type"] == cls.CrossValidation:
            settings["compatibility_mode"] = True


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
                self.folds, shuffle=True, random_state=self.random_state)
            splitter.get_n_splits(table.X, table.Y)
            ind = splitter.split(table.X, table.Y)
        else:
            splitter = skl.KFold(
                self.folds, shuffle=True, random_state=self.random_state)
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
            # pylint: disable=no-member
            rgen = np.random.RandomState(self.random_state)
            sample = rgen.randint(0, len(table), self.n)
            o = np.ones(len(table))
            o[sample] = 0
            others = np.nonzero(o)[0]
            return others, sample
        if self.n == len(table):
            rgen = np.random.RandomState(self.random_state)
            sample = np.arange(self.n)
            rgen.shuffle(sample)
            return np.array([], dtype=int), sample
        elif self.stratified and table.domain.has_discrete_class:
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
        # pylint: disable=no-member
        rgen = np.random.RandomState(self.random_state)
        sample = rgen.randint(0, self.size, self.size)
        sample.sort()  # not needed for the code below, just for the user
        insample = np.ones((self.size,), dtype=bool)
        insample[sample] = False
        remaining = np.flatnonzero(insample)
        return remaining, sample


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataSampler).run(Table("iris"))
