import numpy as np
from Orange.classification.utils.fasterrisk.fasterrisk import (
    RiskScoreOptimizer,
    RiskScoreClassifier,
)

from Orange.classification import Learner, Model
from Orange.data import Table, Storage
from Orange.data.filter import HasClass
from Orange.preprocess import Discretize, Impute, Continuize, SelectBestFeatures
from Orange.preprocess.discretize import Binning
from Orange.preprocess.score import ReliefF


def _change_class_var_values(y):
    """
    Changes the class variable values from 0 and 1 to -1 and 1 or vice versa.
    """
    return np.where(y == 0, -1, np.where(y == -1, 0, y))


class ScoringSheetModel(Model):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def predict_storage(self, table):
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")

        y_pred = _change_class_var_values(self.model.predict(table.X))
        y_prob = self.model.predict_prob(table.X)

        scores = np.hstack(((1 - y_prob).reshape(-1, 1), y_prob.reshape(-1, 1)))
        return y_pred, scores


class ScoringSheetLearner(Learner):
    __returns__ = ScoringSheetModel
    preprocessors = [HasClass(), Discretize(method=Binning()), Impute(), Continuize()]

    def __init__(
        self,
        num_attr_after_selection=20,
        num_decision_params=5,
        max_points_per_param=5,
        num_input_features=None,
        preprocessors=None,
    ):
        # Set the num_decision_params, max_points_per_param, and num_input_features normally
        self.num_decision_params = num_decision_params
        self.max_points_per_param = max_points_per_param
        self.num_input_features = num_input_features
        self.feature_to_group = None

        if preprocessors is None:
            self.preprocessors = [
                *self.preprocessors,
                SelectBestFeatures(method=ReliefF(), k=num_attr_after_selection),
            ]

        super().__init__(preprocessors=preprocessors)

    def incompatibility_reason(self, domain):
        reason = None
        if len(domain.class_vars) > 1 and not self.supports_multiclass:
            reason = "Too many target variables."
        elif not domain.has_discrete_class:
            reason = "Categorical class variable expected."
        elif len(domain.class_vars[0].values) > 2:
            reason = "Too many target variable values."
        return reason

    def fit_storage(self, table):
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")
        elif table.get_nan_count_class() > 0:
            raise ValueError("Class variable contains missing values.")

        if self.num_input_features is not None:
            self._generate_feature_group_index(table)

        X, y, _ = table.X, table.Y, table.W if table.has_weights() else None
        learner = RiskScoreOptimizer(
            X=X,
            y=_change_class_var_values(y),
            k=self.num_decision_params,
            select_top_m=1,
            lb=-self.max_points_per_param,
            ub=self.max_points_per_param,
            group_sparsity=self.num_input_features,
            featureIndex_to_groupIndex=self.feature_to_group,
        )

        self._optimize_decision_params_adjustment(learner)

        multipliers, intercepts, coefficients = learner.get_models()

        model = RiskScoreClassifier(
            multiplier=multipliers[0],
            intercept=intercepts[0],
            coefficients=coefficients[0],
            featureNames=[attribute.name for attribute in table.domain.attributes],
            X_train=X if self.num_decision_params > 10 else None,
        )

        return ScoringSheetModel(model)

    def _optimize_decision_params_adjustment(self, learner):
        """
        This function attempts to optimize (fit) the learner, reducing the number of decision
        parameters ('k')if optimization fails due to being too high.

        Sometimes, the number of decision parameters is too high for the
        number of input features. Which results in a ValueError.
        Continues until successful or 'k' cannot be reduced further.
        """
        while True:
            try:
                learner.optimize()
                return True
            except ValueError as e:
                learner.k -= 1
                if learner.k < 1:
                    # Raise a custom error when k falls below 1
                    raise ValueError(
                        "The number of input features is too low for the current settings."
                    ) from e

    def _generate_feature_group_index(self, table):
        """
        Returns a feature index to group index mapping. The group index is used to group
        binarized features that belong to the same original feature.
        """
        original_feature_names = [
            attribute.compute_value.variable.name
            for attribute in table.domain.attributes
        ]
        feature_to_group_index = {
            feature: idx for idx, feature in enumerate(set(original_feature_names))
        }
        feature_to_group = [
            feature_to_group_index[feature] for feature in original_feature_names
        ]
        self.feature_to_group = np.asarray(feature_to_group)


if __name__ == "__main__":
    mock_learner = ScoringSheetLearner(20, 5, 10, None)
    mock_table = Table("https://datasets.biolab.si/core/heart_disease.tab")
    mock_model = mock_learner(mock_table)
    mock_model(mock_table)
