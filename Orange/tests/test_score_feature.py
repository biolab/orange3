# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import warnings

import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange import preprocess
from Orange.modelling import RandomForestLearner
from Orange.preprocess.score import InfoGain, GainRatio, Gini, Chi2, ANOVA,\
    UnivariateLinearRegression, ReliefF, FCBF, RReliefF
from Orange.projection import PCA
from Orange.tests import test_filename


class FeatureScoringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.zoo = Table("zoo")  # disc. features, disc. class
        cls.housing = Table("housing")  # cont. features, cont. class
        cls.breast = Table(test_filename(
            "datasets/breast-cancer-wisconsin.tab"))
        cls.lenses = Table(test_filename("datasets/lenses.tab"))

    def test_info_gain(self):
        scorer = InfoGain()
        correct = [0.79067, 0.71795, 0.83014, 0.97432, 0.46970]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)],
                                       correct, decimal=5)

    def test_gain_ratio(self):
        scorer = GainRatio()
        correct = [0.80351, 1.00000, 0.84754, 1.00000, 0.59376]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)],
                                       correct, decimal=5)

    def test_gini(self):
        scorer = Gini()
        correct = [0.23786, 0.20855, 0.26235, 0.29300, 0.11946]
        np.testing.assert_almost_equal([scorer(self.zoo, a) for a in range(5)],
                                       correct, decimal=5)

    def test_classless(self):
        classless = Table.from_table(Domain(self.zoo.domain.attributes),
                                     self.zoo[:, 0:-1])
        scorers = [Gini(), InfoGain(), GainRatio()]
        for scorer in scorers:
            with self.assertRaises(ValueError):
                scorer(classless, 0)

    def test_wrong_class_type(self):
        scorers = [Gini(), InfoGain(), GainRatio()]
        for scorer in scorers:
            with self.assertRaises(ValueError):
                scorer(self.housing, 0)

        with self.assertRaises(ValueError):
            Chi2()(self.housing, 0)
        with self.assertRaises(ValueError):
            ANOVA()(self.housing, 2)
        UnivariateLinearRegression()(self.housing, 2)

    def test_chi2(self):
        nrows, ncols = 500, 5
        X = np.random.randint(4, size=(nrows, ncols))
        y = 10 + (-3*X[:, 1] + X[:, 3]) // 2
        domain = Domain.from_numpy(X, y)
        domain = Domain(domain.attributes,
                        DiscreteVariable('c',
                                         values=[str(v) for v in np.unique(y)]))
        table = Table(domain, X, y)
        data = preprocess.Discretize()(table)
        scorer = Chi2()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_anova(self):
        nrows, ncols = 500, 5
        X = np.random.rand(nrows, ncols)
        y = 4 + (-3*X[:, 1] + X[:, 3]) // 2
        domain = Domain.from_numpy(X, y)
        domain = Domain(domain.attributes,
                        DiscreteVariable('c',
                                         values=[str(v) for v in np.unique(y)]))
        data = Table(domain, X, y)
        scorer = ANOVA()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_regression(self):
        nrows, ncols = 500, 5
        X = np.random.rand(nrows, ncols)
        y = (-3*X[:, 1] + X[:, 3]) / 2
        data = Table.from_numpy(None, X, y)
        scorer = UnivariateLinearRegression()
        sc = [scorer(data, a) for a in range(ncols)]
        self.assertTrue(np.argmax(sc) == 1)

    def test_relieff(self):
        old_breast = self.breast.copy()
        weights = ReliefF(random_state=42)(self.breast, None)
        found = [self.breast.domain[attr].name for attr in reversed(weights.argsort()[-3:])]
        reference = ['Bare_Nuclei', 'Clump thickness', 'Marginal_Adhesion']
        self.assertEqual(sorted(found), reference)
        # Original data is unchanged
        np.testing.assert_equal(old_breast.X, self.breast.X)
        np.testing.assert_equal(old_breast.Y, self.breast.Y)
        # Ensure it doesn't crash on adult dataset
        weights = ReliefF(random_state=42)(self.lenses, None)
        found = [self.lenses.domain[attr].name for attr in weights.argsort()[-2:]]
        # some leeway for randomness in relieff random instance selection
        self.assertIn('tear_rate', found)
        # Ensure it doesn't crash on missing target class values
        with old_breast.unlocked():
            old_breast.Y[0] = np.nan
        weights = ReliefF()(old_breast, None)

        np.testing.assert_array_equal(
            ReliefF(random_state=1)(self.breast, None),
            ReliefF(random_state=1)(self.breast, None)
        )

    def test_rrelieff(self):
        X = np.random.random((100, 5))
        y = ((X[:, 0] > .5) ^ (X[:, 1] < .5) - 1).astype(float)
        xor = Table.from_numpy(Domain.from_numpy(X, y), X, y)

        scorer = RReliefF(random_state=42)
        weights = scorer(xor, None)
        best = {xor.domain[attr].name for attr in weights.argsort()[-2:]}
        self.assertSetEqual(set(a.name for a in xor.domain.attributes[:2]), best)
        weights = scorer(self.housing, None)
        best = {self.housing.domain[attr].name for attr in weights.argsort()[-6:]}
        for feature in ('LSTAT', 'RM'):
            self.assertIn(feature, best)

        np.testing.assert_array_equal(
            RReliefF(random_state=1)(self.housing, None),
            RReliefF(random_state=1)(self.housing, None)
        )

    def test_fcbf(self):
        scorer = FCBF()
        weights = scorer(self.zoo, None)
        found = [self.zoo.domain[attr].name for attr in reversed(weights.argsort()[-5:])]
        reference = ['legs', 'milk', 'toothed', 'feathers', 'backbone']
        self.assertEqual(found, reference)

        # GH-1916
        data = Table(Domain([ContinuousVariable('1'), ContinuousVariable('2')],
                            DiscreteVariable('target')),
                     np.full((2, 2), np.nan),
                     np.r_[0., 1])
        with warnings.catch_warnings():
            # these warnings are expected
            warnings.filterwarnings("ignore", "invalid value.*double_scalars")
            warnings.filterwarnings("ignore", "invalid value.*true_divide")

            weights = scorer(data, None)
            np.testing.assert_equal(weights, np.nan)

    def test_learner_with_transformation(self):
        learner = RandomForestLearner(random_state=0)
        iris = Table("iris")
        data = PCA(n_components=2)(iris)(iris)
        scores = learner.score_data(data)
        np.testing.assert_almost_equal(scores, [[0.7760495, 0.2239505]])

    def test_learner_transform_without_variable(self):
        data = self.housing

        def preprocessor_random_column(data):
            # a compute_value without .variable
            def random_column(d):
                return np.random.RandomState(42).rand(len(d))
            nat = ContinuousVariable("nat", compute_value=random_column)
            ndom = Domain(data.domain.attributes + (nat,), data.domain.class_vars)
            return data.transform(ndom)

        learner = RandomForestLearner(random_state=42,
                                      preprocessors=[])
        scores1 = learner.score_data(preprocessor_random_column(data))

        learner = RandomForestLearner(random_state=42,
                                      preprocessors=[preprocessor_random_column])
        # the following line caused an infinite loop due to a bug fix in this commit
        scores2 = learner.score_data(data)

        np.testing.assert_equal(scores1[0][:-1], scores2[0])
