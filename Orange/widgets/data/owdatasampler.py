import sys
import math

from AnyQt.QtWidgets import QFormLayout, QApplication
from AnyQt.QtCore import Qt

import numpy as np
import sklearn.cross_validation as skl_cross_validation

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table
from Orange.data.sql.table import SqlTable
from Orange.widgets.widget import Msg, OWWidget


class OWDataSampler(OWWidget):
    name = "Data Sampler"
    description = "Randomly draw a subset of data points " \
                  "from the input data set."
    icon = "icons/DataSampler.svg"
    priority = 100
    category = "Data"
    keywords = ["data", "sample"]
    inputs = [("Data", Table, "set_data")]
    outputs = [("Data Sample", Table, widget.Default),
               ("Remaining Data", Table)]

    want_main_area = False
    resizing_enabled = False

    RandomSeed = 42
    FixedProportion, FixedSize, CrossValidation, Bootstrap = range(4)
    SqlTime, SqlProportion = range(2)

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

    class Error(OWWidget.Error):
        too_many_folds = Msg("Number of folds exceeds data size")
        sample_larger_than_data = Msg("Sample must be smaller than data")
        not_enough_to_stratify = Msg("Data is too small to stratify")

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
            def f():
                self.sampling_type = i
                self.sampling_type_changed()
            return f

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
            minv=1, maxv=2 ** 31 - 1,
            callback=set_sampling_type(self.FixedSize))
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

        gui.appendRadioButton(sampling, "Boostrap")

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
        self.indices = None

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
                '{}{} instances in input data set.'.format(*(
                    ('~', dataset.approx_len()) if sql else
                    ('', len(dataset)))))
            if not sql:
                self.sampleSizeSpin.setMaximum(len(dataset))
                self.updateindices()
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.outputInfoLabel.setText('')
            self.indices = None
            self.clear_messages()
        self.commit()

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
            sample = self.data[sample]
            other = self.data[remaining]
            self.sampled_instances = len(sample)
            self.remaining_instances = len(other)
        self.send("Data Sample", sample)
        self.send("Remaining Data", other)

    def updateindices(self):
        self.Error.clear()
        repl = True
        data_length = len(self.data)
        num_classes = len(self.data.domain.class_var.values) \
            if self.data.domain.has_discrete_class else 0

        size = None
        if self.sampling_type == self.FixedSize:
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

        if not repl and size is not None and (data_length <= size):
            self.Error.sample_larger_than_data()
        if not repl and data_length <= num_classes and self.stratify:
            self.Error.not_enough_to_stratify()

        if self.Error.active:
            self.indices = None
            return

        rnd = self.RandomSeed if self.use_seed else None
        stratified = (self.stratify and
                      type(self.data) == Table and
                      self.data.domain.has_discrete_class)
        if self.sampling_type == self.FixedSize:
            self.indices = sample_random_n(
                self.data, size,
                stratified=stratified, replace=self.replacement,
                random_state=rnd)
        elif self.sampling_type == self.FixedProportion:
            self.indices = sample_random_p(
                self.data, self.sampleSizePercentage / 100,
                stratified=stratified, random_state=rnd)
        elif self.sampling_type == self.Bootstrap:
            self.indices = sample_bootstrap(data_length, random_state=rnd)
        else:
            self.indices = sample_fold_indices(
                self.data, self.number_of_folds, stratified=stratified,
                random_state=rnd)

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


def sample_fold_indices(table, folds=10, stratified=False, random_state=None):
    """
    :param Orange.data.Table table:
    :param int folds: Number of folds
    :param bool stratified: Return stratified indices (if applicable).
    :param Random random_state:
    :rval tuple-of-arrays: A tuple of array indices one for each fold.
    """
    if stratified and table.domain.has_discrete_class:
        # XXX: StratifiedKFold does not support random_state
        ind = skl_cross_validation.StratifiedKFold(
            table.Y.ravel(), folds, random_state=random_state)
    else:
        ind = skl_cross_validation.KFold(
            len(table), folds, shuffle=True, random_state=random_state)
    return tuple(ind)


def sample_random_n(table, n, stratified=False, replace=False,
                    random_state=None):
    if replace:
        if random_state is None:
            rgen = np.random
        else:
            rgen = np.random.mtrand.RandomState(random_state)
        sample = rgen.randint(len(table), size=n)
        o = np.ones(len(table))
        o[sample] = 0
        others = np.nonzero(o)[0]
        return others, sample
    if stratified and table.domain.has_discrete_class:
        test_size = max(len(table.domain.class_var.values), n)
        ind = skl_cross_validation.StratifiedShuffleSplit(
            table.Y.ravel(), n_iter=1,
            test_size=test_size, train_size=len(table) - test_size,
            random_state=random_state)
    else:
        ind = skl_cross_validation.ShuffleSplit(
            len(table), n_iter=1,
            test_size=n, random_state=random_state)
    return next(iter(ind))


def sample_random_p(table, p, stratified=False, random_state=None):
    n = int(math.ceil(len(table) * p))
    return sample_random_n(table, n, stratified, False, random_state)


def sample_bootstrap(size, random_state=None):
    rgen = np.random.RandomState(random_state)
    sample = rgen.randint(0, size, size)
    sample.sort()  # not needed for the code below, just for the user
    insample = np.ones((size,), dtype=np.bool)
    insample[sample] = False
    remaining = np.flatnonzero(insample)
    return remaining, sample


def test_main():
    app = QApplication([])
    data = Table("iris")
    w = OWDataSampler()
    w.set_data(data)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(test_main())
