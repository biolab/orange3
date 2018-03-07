# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.data.owdatasampler import OWDataSampler
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSampler(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.zoo = Table("zoo")

    def setUp(self):
        self.widget = self.create_widget(OWDataSampler)  # type: OWDataSampler

    def test_error_message(self):
        """ Check if error message appears and then disappears when
        data is removed from input"""
        self.widget.controls.sampling_type.buttons[2].click()
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", self.iris[:5])
        self.assertTrue(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", Table(self.iris.domain))
        self.assertTrue(self.widget.Error.no_data.is_shown())

    def test_stratified_on_unbalanced_data(self):
        unbalanced_data = self.iris[:51]

        self.widget.controls.stratify.setChecked(True)
        self.send_signal("Data", unbalanced_data)
        self.assertTrue(self.widget.Warning.could_not_stratify.is_shown())

    def test_bootstrap(self):
        self.select_sampling_type(self.widget.Bootstrap)

        self.send_signal("Data", self.iris)

        in_input = set(self.iris.ids)
        sample = self.get_output("Data Sample")
        in_sample = set(sample.ids)
        in_remaining = set(self.get_output("Remaining Data").ids)

        # Bootstrap should sample len(input) instances
        self.assertEqual(len(sample), len(self.iris))
        # Sample and remaining should cover all instances, while none
        # should be present in both
        self.assertEqual(len(in_sample | in_remaining), len(in_input))
        self.assertEqual(len(in_sample & in_remaining), 0)
        # Sampling with replacement will always produce at least one distinct
        # instance in sample, and at least one instance in remaining with
        # high probability (1-(1/150*2/150*...*150/150) ~= 1-2e-64)
        self.assertGreater(len(in_sample), 0)
        self.assertGreater(len(in_remaining), 0)

    def select_sampling_type(self, sampling_type):
        buttons = self.widget.controls.sampling_type.group.buttons()
        buttons[sampling_type].click()

    def test_no_intersection_in_outputs(self):
        """ Check whether outputs intersect and whether length of outputs sums
        to length of original data """
        self.send_signal("Data", self.zoo)
        w = self.widget
        sampling_types = [w.FixedProportion, w.FixedSize, w.CrossValidation]

        for replicable in [True, False]:
            for stratified in [True, False]:
                for sampling_type in sampling_types:
                    self.widget.cb_seed.setChecked(replicable)
                    self.widget.cb_stratify.setChecked(stratified)
                    self.select_sampling_type(sampling_type)
                    self.widget.commit()

                    sample = self.get_output("Data Sample")
                    other = self.get_output("Remaining Data")
                    self.assertEqual(len(self.zoo), len(sample) + len(other))
                    self.assertNoIntersection(sample, other)

    def test_bigger_size_with_replacement(self):
        """Allow bigger output without replacement."""
        self.send_signal('Data', self.iris[:2])
        sample_size = self.set_fixed_sample_size(3, with_replacement=True)
        self.assertEqual(3, sample_size, 'Should be able to set a bigger size '
                         'with replacement')

    def test_bigger_size_without_replacement(self):
        """Lower output samples to match input's without replacement."""
        self.send_signal('Data', self.iris[:2])
        sample_size = self.set_fixed_sample_size(3)
        self.assertEqual(2, sample_size)

    def test_bigger_output_warning(self):
        """Should warn when sample size is bigger than input."""
        self.send_signal('Data', self.iris[:2])
        self.set_fixed_sample_size(3, with_replacement=True)
        self.assertTrue(self.widget.Warning.bigger_sample.is_shown())

    def set_fixed_sample_size(self, sample_size, with_replacement=False):
        """Set fixed sample size and return the number of gui spin.

        Return the actual number in gui so we can check whether it is different
        from sample_size. The number can be changed depending on the spin
        maximum value.
        """
        self.select_sampling_type(self.widget.FixedSize)
        self.widget.controls.replacement.setChecked(with_replacement)
        self.widget.sampleSizeSpin.setValue(sample_size)
        self.widget.commit()
        return self.widget.sampleSizeSpin.value()

    def assertNoIntersection(self, sample, other):
        for inst in sample:
            self.assertNotIn(inst, other)
