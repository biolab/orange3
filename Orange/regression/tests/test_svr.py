import unittest

import numpy as np

from Orange.data import Table
from Orange.preprocess import RemoveNaNRows
from Orange.regression import SVRLearner, NuSVRLearner


class TestSVM(unittest.TestCase):
    def test_support_vectors(self):
        data = Table("housing")[:60]

        for learner_class in [SVRLearner, NuSVRLearner]:
            with self.subTest(learner=learner_class.__name__):
                learner = learner_class(
                    preprocessors=[RemoveNaNRows(), *SVRLearner.preprocessors])
                model = learner(data)
                sv_orange = data[model.support_].transform(model.domain).X
                assert len(sv_orange) < len(data)
                sv_sklearn = model.skl_model.support_vectors_
                np.testing.assert_equal(sv_orange, sv_sklearn)

                learner = learner_class()
                model = learner(data)
                sv_orange = data[model.support_].transform(model.domain).X
                sv_sklearn = model.skl_model.support_vectors_
                np.testing.assert_equal(sv_orange, sv_sklearn)


if __name__ == "__main__":
    unittest.main()
