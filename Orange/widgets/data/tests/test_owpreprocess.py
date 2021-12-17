# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import numpy as np

from Orange.data import Table
from Orange.preprocess import (
    Randomize, Scale, Discretize, Continuize, Impute, ProjectPCA, \
         ProjectCUR, RemoveSparse
)
from Orange.preprocess import discretize, impute, fss, score
from Orange.widgets.data import owpreprocess
from Orange.widgets.data.owpreprocess import OWPreprocess, \
    UnivariateFeatureSelect, Scale as ScaleEditor
from Orange.widgets.tests.base import WidgetTest, datasets


class TestOWPreprocess(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.zoo = Table("zoo")

    def test_randomize(self):
        saved = {"preprocessors": [("orange.preprocess.randomize",
                                    {"rand_type": Randomize.RandomizeClasses,
                                     "rand_seed": 1})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, self.zoo)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        r = Randomize(Randomize.RandomizeClasses, rand_seed=1)
        expected = r(self.zoo)

        np.testing.assert_array_equal(expected.X, output.X)
        np.testing.assert_array_equal(expected.Y, output.Y)
        np.testing.assert_array_equal(expected.metas, output.metas)

        np.testing.assert_array_equal(self.zoo.X, output.X)
        np.testing.assert_array_equal(self.zoo.metas, output.metas)
        self.assertFalse(np.array_equal(self.zoo.Y, output.Y))

    def test_remove_sparse(self):
        data = Table("iris")
        idx = int(data.X.shape[0]/10)
        with data.unlocked():
            data.X[:idx+1, 0] = np.zeros((idx+1,))
        saved = {"preprocessors": [("orange.preprocess.remove_sparse",
                                    {'filter0': True, 'useFixedThreshold': False,
                                     'percThresh':10, 'fixedThresh': 50})]}
        model = self.widget.load(saved)

        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        np.testing.assert_array_equal(output.X, data.X[:, 1:])
        np.testing.assert_array_equal(output.Y, data.Y)
        np.testing.assert_array_equal(output.metas, data.metas)

    def test_normalize(self):
        data = Table("iris")
        saved = {"preprocessors": [("orange.preprocess.scale",
                                    {"method": ScaleEditor.NormalizeBySD})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        np.testing.assert_allclose(output.X.mean(0), 0, atol=1e-7)
        np.testing.assert_allclose(output.X.std(0), 1, atol=1e-7)

        saved = {"preprocessors": [("orange.preprocess.scale",
                                    {"method": ScaleEditor.CenterByMean})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        np.testing.assert_allclose(output.X.mean(0), 0, atol=1e-7)
        self.assertTrue((output.X.std(0) > 0).all())

        saved = {"preprocessors": [("orange.preprocess.scale",
                                    {"method": ScaleEditor.ScaleBySD})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertTrue((output.X.mean(0) != 0).all())
        np.testing.assert_allclose(output.X.std(0), 1, atol=1e-7)

        saved = {"preprocessors":
                 [("orange.preprocess.scale",
                   {"method": ScaleEditor.NormalizeBySpan_ZeroBased})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertAlmostEqual(output.X.min(), 0)
        self.assertAlmostEqual(output.X.max(), 1)

        saved = {"preprocessors":
                 [("orange.preprocess.scale",
                   {"method": ScaleEditor.NormalizeSpan_NonZeroBased})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertAlmostEqual(output.X.min(), -1)
        self.assertAlmostEqual(output.X.max(), 1)

    def test_select_features(self):
        data = Table("iris")
        saved = {"preprocessors": [("orange.preprocess.fss",
                                    {"strategy": UnivariateFeatureSelect.Fixed,
                                     "k": 2})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertEqual(len(output.domain.attributes), 2)

        saved = {"preprocessors": [
            ("orange.preprocess.fss",
             {"strategy": UnivariateFeatureSelect.Proportion,
              "p": 75})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.preprocessed_data)
        self.assertEqual(len(output.domain.attributes), 3)

    def test_data_column_nans(self):
        """
        ZeroDivisonError - Weights sum to zero, can't be normalized
        In case when all rows in a column are NaN then it throws that error.
        GH-2064
        """
        table = datasets.data_one_column_nans()
        saved = {"preprocessors": [("orange.preprocess.scale",
                                    {"center": Scale.CenteringType.Mean,
                                     "scale": Scale.ScalingType.Std})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal(self.widget.Inputs.data, table)


# Test for editors
class TestDiscretizeEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.DiscretizeEditor()
        self.assertEqual(widget.parameters(),
                         {"method": owpreprocess.DiscretizeEditor.EqualFreq,
                          "n": 4})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Discretize)
        self.assertIsInstance(p.method, discretize.EqualFreq)
        widget.setParameters(
            {"method": owpreprocess.DiscretizeEditor.EntropyMDL}
        )
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Discretize)
        self.assertIsInstance(p.method, discretize.EntropyMDL)

        widget.setParameters(
            {"method": owpreprocess.DiscretizeEditor.EqualWidth,
             "n": 10}
        )
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Discretize)
        self.assertIsInstance(p.method, discretize.EqualWidth)
        self.assertEqual(p.method.n, 10)


class TestContinuizeEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.ContinuizeEditor()
        self.assertEqual(widget.parameters(),
                         {"multinomial_treatment": Continuize.Indicators})

        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Continuize)
        self.assertEqual(p.multinomial_treatment, Continuize.Indicators)

        widget.setParameters(
            {"multinomial_treatment": Continuize.FrequentAsBase})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Continuize)
        self.assertEqual(p.multinomial_treatment, Continuize.FrequentAsBase)


class TestImputeEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.ImputeEditor()
        self.assertEqual(widget.parameters(),
                         {"method": owpreprocess.ImputeEditor.Average})
        widget.setParameters(
            {"method": owpreprocess.ImputeEditor.Average}
        )
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Impute)
        self.assertIsInstance(p.method, impute.Average)


class TestFeatureSelectEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.FeatureSelectEditor()
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, fss.SelectBestFeatures)
        self.assertIsInstance(p.method, score.InfoGain)
        self.assertEqual(p.k, 10)

        widget.setParameters({"k": 9999})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, fss.SelectBestFeatures)
        self.assertEqual(p.k, 9999)


class TestRandomFeatureSelectEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.RandomFeatureSelectEditor()
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, fss.SelectRandomFeatures)
        self.assertEqual(p.k, 10)

        widget.setParameters({"strategy": owpreprocess.RandomFeatureSelectEditor.Fixed,
                              "k": 9999})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, fss.SelectRandomFeatures)
        self.assertEqual(p.k, 9999)

        widget.setParameters(
            {"strategy": owpreprocess.RandomFeatureSelectEditor.Percentage,
             "p": 25})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, fss.SelectRandomFeatures)
        self.assertEqual(p.k, 0.25)


class TestRandomizeEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.Randomize()
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Randomize)
        self.assertEqual(p.rand_type, Randomize.RandomizeClasses)

        widget.setParameters({"rand_type": Randomize.RandomizeAttributes})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, Randomize)
        self.assertEqual(p.rand_type, Randomize.RandomizeAttributes)


class TestPCAEditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.PCA()
        self.assertEqual(widget.parameters(),
                         {"n_components": 10})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, ProjectPCA)
        self.assertEqual(p.n_components, 10)

        widget.setParameters({"n_components": 5})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, ProjectPCA)
        self.assertEqual(p.n_components, 5)


class TestCUREditor(WidgetTest):
    def test_editor(self):
        widget = owpreprocess.CUR()
        self.assertEqual(widget.parameters(),
                         {"rank": 10, "max_error": 1})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, ProjectCUR)
        self.assertEqual(p.rank, 10)
        self.assertEqual(p.max_error, 1)

        widget.setParameters({"rank": 9999})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, ProjectCUR)
        self.assertEqual(p.rank, 9999)

        widget.setParameters({"rank": 5, "max_error": 0.5})
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, ProjectCUR)
        self.assertEqual(p.rank, 5)
        self.assertEqual(p.max_error, 0.5)

class TestRemoveSparseEditor(WidgetTest):

    def test_editor(self):
        widget = owpreprocess.RemoveSparseEditor()
        self.assertEqual(
            widget.parameters(),
            dict(fixedThresh=50, percThresh=5, filter0=True,
                 useFixedThreshold=False))

        p = widget.createinstance(widget.parameters())
        widget.filterSettingsClicked()
        self.assertTrue(widget.percSpin.isEnabled())
        self.assertFalse(widget.fixedSpin.isEnabled())
        self.assertIsInstance(p, RemoveSparse)
        self.assertEqual(p.filter0, True)
        self.assertEqual(p.threshold, 0.05)

        widget.setParameters(
            dict(useFixedThreshold=True, fixedThresh=30, filter0=False))
        p = widget.createinstance(widget.parameters())
        self.assertIsInstance(p, RemoveSparse)
        self.assertEqual(p.threshold, 30)
        self.assertFalse(p.filter0)
