import unittest
from unittest.mock import patch, Mock
import sys

from Orange.classification import GBClassifier

try:
    from Orange.classification import XGBClassifier, XGBRFClassifier
except ImportError:
    XGBClassifier = XGBRFClassifier = None
try:
    from Orange.classification import CatGBClassifier
except ImportError:
    CatGBClassifier = None
from Orange.data import Table
from Orange.modelling import GBLearner
from Orange.preprocess.score import Scorer
from Orange.regression import GBRegressor

try:
    from Orange.regression import XGBRegressor, XGBRFRegressor
except ImportError:
    XGBRegressor = XGBRFRegressor = None
try:
    from Orange.regression import CatGBRegressor
except ImportError:
    CatGBRegressor = None
from Orange.widgets.model.owgradientboosting import OWGradientBoosting, \
    LearnerItemModel, GBLearnerEditor, XGBLearnerEditor, XGBRFLearnerEditor, \
    CatGBLearnerEditor
from Orange.widgets.settings import SettingProvider
from Orange.widgets.tests.base import WidgetTest, ParameterMapping, \
    WidgetLearnerTestMixin, datasets, simulate, GuiTest
from Orange.widgets.widget import OWWidget


def create_parent(editor_class):
    class DummyWidget(OWWidget):
        name = "Mock"
        settings_changed = Mock()
        editor = SettingProvider(editor_class)

    return DummyWidget()


class TestLearnerItemModel(GuiTest):
    def test_model(self):
        classifiers = [GBClassifier, XGBClassifier,
                       XGBRFClassifier, CatGBClassifier]
        widget = create_parent(CatGBLearnerEditor)
        model = LearnerItemModel(widget)
        n_items = 4
        self.assertEqual(model.rowCount(), n_items)
        for i in range(n_items):
            self.assertEqual(model.item(i).isEnabled(),
                             classifiers[i] is not None)

    @patch("Orange.widgets.model.owgradientboosting.LearnerItemModel.LEARNERS",
           [(GBLearner, "", ""),
            (None, "Gradient Boosting (catboost)", "catboost")])
    def test_missing_lib(self):
        widget = create_parent(CatGBLearnerEditor)
        model = LearnerItemModel(widget)
        self.assertEqual(model.rowCount(), 2)
        self.assertTrue(model.item(0).isEnabled())
        self.assertFalse(model.item(1).isEnabled())


class TestGBLearnerEditor(GuiTest):
    def setUp(self):
        editor_class = GBLearnerEditor
        self.widget = create_parent(editor_class)
        self.editor = editor_class(self.widget)

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                "random_state": 0, "subsample": 1, "min_samples_split": 2}
        self.assertDictEqual(self.editor.get_arguments(), args)

    def test_learner_parameters(self):
        params = (("Method", "Gradient Boosting (scikit-learn)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.1),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 3),
                  ("Fraction of training instances", 1),
                  ("Stop splitting nodes with maximum instances", 2))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = GBClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(params["learning_rate"], self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["min_samples_split"],
                         self.editor.min_samples_split)
        self.assertTrue(self.editor.random_state)  # different than default
        self.assertIsNone(params["random_state"])

    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = GBRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(params["learning_rate"], self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["min_samples_split"],
                         self.editor.min_samples_split)
        self.assertTrue(self.editor.random_state)  # different than default
        self.assertIsNone(params["random_state"])


class TestXGBLearnerEditor(GuiTest):
    def setUp(self):
        editor_class = XGBLearnerEditor
        self.widget = create_parent(editor_class)
        self.editor = editor_class(self.widget)

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 1, "colsample_bytree": 1, "colsample_bylevel": 1,
                "colsample_bynode": 1, "subsample": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    @unittest.skipIf(XGBClassifier is None, "Missing 'xgboost' package")
    def test_learner_parameters(self):
        params = (("Method", "Extreme Gradient Boosting (xgboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 1),
                  ("Fraction of training instances", 1),
                  ("Fraction of features for each tree", 1),
                  ("Fraction of features for each level", 1),
                  ("Fraction of features for each split", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    @unittest.skipIf(XGBClassifier is None, "Missing 'xgboost' package")
    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = XGBClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(round(params["learning_rate"], 1),
                         self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["reg_lambda"], self.editor.lambda_)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["colsample_bytree"],
                         self.editor.colsample_bytree)
        self.assertEqual(params["colsample_bylevel"],
                         self.editor.colsample_bylevel)
        self.assertEqual(params["colsample_bynode"],
                         self.editor.colsample_bynode)

    @unittest.skipIf(XGBRegressor is None, "Missing 'xgboost' package")
    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = XGBRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(round(params["learning_rate"], 1),
                         self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["reg_lambda"], self.editor.lambda_)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["colsample_bytree"],
                         self.editor.colsample_bytree)
        self.assertEqual(params["colsample_bylevel"],
                         self.editor.colsample_bylevel)
        self.assertEqual(params["colsample_bynode"],
                         self.editor.colsample_bynode)


class TestXGBRFLearnerEditor(GuiTest):
    def setUp(self):
        editor_class = XGBRFLearnerEditor
        self.widget = create_parent(editor_class)
        self.editor = editor_class(self.widget)

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 1, "colsample_bytree": 1, "colsample_bylevel": 1,
                "colsample_bynode": 1, "subsample": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    @unittest.skipIf(XGBRFClassifier is None, "Missing 'xgboost' package")
    def test_learner_parameters(self):
        params = (("Method",
                   "Extreme Gradient Boosting Random Forest (xgboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 1),
                  ("Fraction of training instances", 1),
                  ("Fraction of features for each tree", 1),
                  ("Fraction of features for each level", 1),
                  ("Fraction of features for each split", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    @unittest.skipIf(XGBRFClassifier is None, "Missing 'xgboost' package")
    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = XGBRFClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(round(params["learning_rate"], 1),
                         self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["reg_lambda"], self.editor.lambda_)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["colsample_bytree"],
                         self.editor.colsample_bytree)
        self.assertEqual(params["colsample_bylevel"],
                         self.editor.colsample_bylevel)
        self.assertEqual(params["colsample_bynode"],
                         self.editor.colsample_bynode)

    @unittest.skipIf(XGBRFRegressor is None, "Missing 'xgboost' package")
    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = XGBRFRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(round(params["learning_rate"], 1),
                         self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["reg_lambda"], self.editor.lambda_)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["colsample_bytree"],
                         self.editor.colsample_bytree)
        self.assertEqual(params["colsample_bylevel"],
                         self.editor.colsample_bylevel)
        self.assertEqual(params["colsample_bynode"],
                         self.editor.colsample_bynode)


class TestCatGBLearnerEditor(GuiTest):
    def setUp(self):
        editor_class = CatGBLearnerEditor
        self.widget = create_parent(editor_class)
        self.editor = editor_class(self.widget)

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 3, "colsample_bylevel": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    @unittest.skipIf(CatGBClassifier is None, "Missing 'catboost' package")
    def test_learner_parameters(self):
        params = (("Method", "Gradient Boosting (catboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 3),
                  ("Fraction of features for each tree", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    @unittest.skipIf(CatGBClassifier is None, "Missing 'catboost' package")
    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = CatGBClassifier()
        model = booster(data)
        params = model.cat_model.get_all_params()
        self.assertEqual(self.editor.n_estimators, 100)
        self.assertEqual(params["iterations"], 1000)
        self.assertEqual(params["depth"], self.editor.max_depth)
        self.assertEqual(params["l2_leaf_reg"], self.editor.lambda_)
        self.assertEqual(params["rsm"], self.editor.colsample_bylevel)
        self.assertEqual(self.editor.learning_rate, 0.3)
        # params["learning_rate"] is automatically defined so don't test it

    @unittest.skipIf(CatGBRegressor is None, "Missing 'catboost' package")
    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = CatGBRegressor()
        model = booster(data)
        params = model.cat_model.get_all_params()
        self.assertEqual(self.editor.n_estimators, 100)
        self.assertEqual(params["iterations"], 1000)
        self.assertEqual(params["depth"], self.editor.max_depth)
        self.assertEqual(params["l2_leaf_reg"], self.editor.lambda_)
        self.assertEqual(params["rsm"], self.editor.colsample_bylevel)
        self.assertEqual(self.editor.learning_rate, 0.3)
        # params["learning_rate"] is automatically defined so don't test it

class TestOWGradientBoosting(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWGradientBoosting,
                                         stored_settings={"auto_apply": False})
        self.init()

        controls = self.widget.editor.controls
        self.parameters = [
            ParameterMapping("n_estimators", controls.n_estimators, [500, 10]),
            ParameterMapping("learning_rate", controls.learning_rate),
            ParameterMapping("max_depth", controls.max_depth),
            ParameterMapping("min_samples_split", controls.min_samples_split),
        ]

    def test_scorer(self):
        self.assertIsInstance(
            self.get_output(self.widget.Outputs.learner), Scorer
        )

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)

    @unittest.skipIf(XGBClassifier is None, "Missing 'xgboost' package")
    def test_xgb_params(self):
        simulate.combobox_activate_index(self.widget.controls.method_index, 1)
        editor = self.widget.editor
        controls = editor.controls
        reg_slider = controls.lambda_index

        self.parameters = [
            ParameterMapping("n_estimators", controls.n_estimators, [500, 10]),
            ParameterMapping("learning_rate", controls.learning_rate),
            ParameterMapping("max_depth", controls.max_depth),
            ParameterMapping("reg_lambda", reg_slider,
                             values=[editor.LAMBDAS[0], editor.LAMBDAS[-1]],
                             getter=lambda: editor.LAMBDAS[reg_slider.value()],
                             setter=lambda val: reg_slider.setValue(
                                 editor.LAMBDAS.index(val))),
        ]
        self.test_parameters()

    def test_methods(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        method_cb = self.widget.controls.method_index
        for i, (cls, _, _) in enumerate(LearnerItemModel.LEARNERS):
            if cls is None:
                continue
            simulate.combobox_activate_index(method_cb, i)
            self.widget.apply_button.button.click()
            self.assertIsInstance(self.widget.learner, cls)

    def test_missing_lib(self):
        modules = {k: v for k, v in sys.modules.items()
                   if "orange" not in k.lower()}  # retain built-ins
        modules["xgboost"] = None
        modules["catboost"] = None
        # pylint: disable=reimported,redefined-outer-name
        # pylint: disable=import-outside-toplevel
        with patch.dict(sys.modules, modules, clear=True):
            from Orange.widgets.model.owgradientboosting import \
                OWGradientBoosting
            widget = self.create_widget(OWGradientBoosting,
                                        stored_settings={"method_index": 3})
            self.assertEqual(widget.method_index, 0)


if __name__ == "__main__":
    unittest.main()
