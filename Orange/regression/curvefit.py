import ast
from typing import Callable, List, Optional, Union, Dict, Tuple, Any

import numpy as np
from scipy.optimize import curve_fit

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.filter import HasClass
from Orange.data.util import get_unique_names
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange.regression import Learner, Model

__all__ = ["CurveFitLearner"]


class CurveFitModel(Model):
    def __init__(
            self,
            domain: Domain,
            original_domain: Domain,
            parameters_names: List[str],
            parameters: np.ndarray,
            function: Optional[Callable],
            create_lambda_args: Optional[Tuple]
    ):
        super().__init__(domain, original_domain)
        self.__parameters_names = parameters_names
        self.__parameters = parameters

        if function is None and create_lambda_args is not None:
            function, names, _ = _create_lambda(**create_lambda_args)
            assert parameters_names == names

        assert function

        self.__function = function
        self.__create_lambda_args = create_lambda_args

    @property
    def coefficients(self) -> Table:
        return Table(Domain([ContinuousVariable("coef")],
                            metas=[StringVariable("name")]),
                     self.__parameters[:, None],
                     metas=np.array(self.__parameters_names)[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted = self.__function(X, *self.__parameters)
        if not isinstance(predicted, np.ndarray):
            # handle constant function; i.e. len(self.domain.attributes) == 0
            return np.full(len(X), predicted, dtype=float)
        return predicted.flatten()

    def __getstate__(self) -> Dict:
        if not self.__create_lambda_args:
            raise AttributeError(
                "Can't pickle/copy callable. Use str expression instead."
            )
        return {
            "domain": self.domain,
            "original_domain": self.original_domain,
            "parameters_names": self.__parameters_names,
            "parameters": self.__parameters,
            "function": None,
            "args": self.__create_lambda_args,
        }

    def __setstate__(self, state: Dict):
        self.__init__(*state.values())


class CurveFitLearner(Learner):
    """
    Fit a function to data.
    It uses the scipy.curve_fit to find the optimal values of parameters.

    Parameters
    ----------
    expression : callable or str
        A modeling function.
        If callable, it must take the independent variable as the first
        argument and the parameters to fit as separate remaining arguments.
        If string, a lambda function is created,
        using `expression`, `available_feature_names`, `function` and `env`
        attributes.
        Should be string for pickling the model.
    parameters_names : list of str
        List of parameters names. Only needed when the expression
        is callable.
    features_names : list of str
        List of features names. Only needed when the expression
        is callable.
    available_feature_names : list of str
        List of all available features names. Only needed when the expression
        is string. Needed to distinguish between parameters and features when
        translating the expression into the lambda.
    functions : list of str
        List of all available functions. Only needed when the expression
        is string. Needed to distinguish between parameters and functions when
        translating the expression into the lambda.
    sanitizer : callable
        Function for sanitizing names.
    env : dict
        An environment to capture in the lambda's closure.
    p0 : list of floats, optional
        Initial guess for the parameters.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters.
    preprocessors : tuple of Orange preprocessors, optional
        The processors that will be used when data is passed to the learner.

    Examples
    --------
    >>> import numpy as np
    >>> from Orange.data import Table
    >>> from Orange.regression import CurveFitLearner
    >>> data = Table("housing")
    >>> # example with callable expression
    >>> cfun = lambda x, a, b, c: a * np.exp(-b * x[:, 0] * x[:, 1]) + c
    >>> learner = CurveFitLearner(cfun, ["a", "b", "c"], ["CRIM", "LSTAT"])
    >>> model = learner(data)
    >>> pred = model(data)
    >>> coef = model.coefficients
    >>> # example with str expression
    >>> sfun = "a * exp(-b * CRIM * LSTAT) + c"
    >>> names = [a.name for a in data.domain.attributes]
    >>> learner = CurveFitLearner(sfun, available_feature_names=names,
    ...                           functions=["exp"])
    >>> model = learner(data)
    >>> pred = model(data)
    >>> coef = model.coefficients

    """
    preprocessors = [HasClass(), RemoveNaNColumns(), Impute()]
    __returns__ = CurveFitModel
    name = "Curve Fit"

    def __init__(
            self,
            expression: Union[Callable, ast.Expression, str],
            parameters_names: Optional[List[str]] = None,
            features_names: Optional[List[str]] = None,
            available_feature_names: Optional[List[str]] = None,
            functions: Optional[List[str]] = None,
            sanitizer: Optional[Callable] = None,
            env: Optional[Dict[str, Any]] = None,
            p0: Union[List, Dict, None] = None,
            bounds: Union[Tuple, Dict] = (-np.inf, np.inf),
            preprocessors=None
    ):
        super().__init__(preprocessors)

        if callable(expression):
            if parameters_names is None:
                raise TypeError("Provide 'parameters_names' parameter.")
            if features_names is None:
                raise TypeError("Provide 'features_names' parameter.")

            args = None
            function = expression
        else:
            if available_feature_names is None:
                raise TypeError("Provide 'available_feature_names' parameter.")
            if functions is None:
                raise TypeError("Provide 'functions' parameter.")

            args = dict(expression=expression,
                        available_feature_names=available_feature_names,
                        functions=functions, sanitizer=sanitizer, env=env)
            function, parameters_names, features_names = _create_lambda(**args)

        if isinstance(p0, dict):
            p0 = [p0.get(p, 1) for p in parameters_names]
        if isinstance(bounds, dict):
            d = [-np.inf, np.inf]
            lower_bounds = [bounds.get(p, d)[0] for p in parameters_names]
            upper_bounds = [bounds.get(p, d)[1] for p in parameters_names]
            bounds = lower_bounds, upper_bounds

        self.__function = function
        self.__parameters_names = parameters_names
        self.__features_names = features_names
        self.__p0 = p0
        self.__bounds = bounds

        # needed for pickling - if the expression is a lambda function, the
        # learner is not picklable
        self.__create_lambda_args = args

    @property
    def parameters_names(self) -> List[str]:
        return self.__parameters_names

    def fit_storage(self, data: Table) -> CurveFitModel:
        domain: Domain = data.domain
        attributes = []
        for attr in domain.attributes:
            if attr.name in self.__features_names:
                if not attr.is_continuous:
                    raise ValueError("Numeric feature expected.")
                attributes.append(attr)

        new_domain = Domain(attributes, domain.class_vars, domain.metas)
        transformed = data.transform(new_domain)
        params = curve_fit(self.__function, transformed.X, transformed.Y,
                           p0=self.__p0, bounds=self.__bounds)[0]
        return CurveFitModel(new_domain, domain,
                             self.__parameters_names, params, self.__function,
                             self.__create_lambda_args)

    def __getstate__(self) -> Dict:
        if not self.__create_lambda_args:
            raise AttributeError(
                "Can't pickle/copy callable. Use str expression instead."
            )
        state = self.__create_lambda_args.copy()
        state["parameters_names"] = None
        state["features_names"] = None
        state["p0"] = self.__p0
        state["bounds"] = self.__bounds
        state["preprocessors"] = self.preprocessors
        return state

    def __setstate__(self, state: Dict):
        expression = state.pop("expression")
        self.__init__(expression, **state)


def _create_lambda(
        expression: Union[str, ast.Expression] = "",
        available_feature_names: List[str] = None,
        functions: List[str] = None,
        sanitizer: Callable = None,
        env: Optional[Dict[str, Any]] = None
) -> Tuple[Callable, List[str], List[str]]:
    """
    Create a lambda function from a string expression.

    Parameters
    ----------
    expression : str or ast.Expression
        Right side of a modeling function.
    available_feature_names : list of str
        List of all available features names.
        Needed to distinguish between parameters, features and functions.
    functions : list of str
        List of all available functions.
        Needed to distinguish between parameters, features and functions.
    sanitizer : callable, optional
        Function for sanitizing variable names.
    env : dict, optional
        An environment to capture in the lambda's closure.

    Returns
    -------
    func : callable
        The created lambda function.
    params : list of str
        The recognied parameters withint the expression.
    vars_ : list of str
        The recognied variables withint the expression.

    Examples
    --------
    >>> from Orange.data import Table
    >>> data = Table("housing")
    >>> sfun = "a * exp(-b * CRIM * LSTAT) + c"
    >>> names = [a.name for a in data.domain.attributes]
    >>> func, par, var = _create_lambda(sfun, available_feature_names=names,
    ...                                 functions=["exp"], env={"exp": np.exp})
    >>> y = func(data.X, 1, 2, 3)
    >>> par
    ['a', 'b', 'c']
    >>> var
    ['CRIM', 'LSTAT']

    """
    if sanitizer is None:
        sanitizer = lambda n: n
    if env is None:
        env = {name: getattr(np, name) for name in functions}

    exp = ast.parse(expression, mode="eval")
    search = _ParametersSearch(
        [sanitizer(name) for name in available_feature_names],
        functions
    )
    search.visit(exp)
    params = search.parameters
    used_sanitized_feature_names = search.variables

    name = get_unique_names(params, "x")
    feature_mapper = {n: i for i, n in enumerate(used_sanitized_feature_names)}
    exp = _ReplaceVars(name, feature_mapper, functions).visit(exp)

    lambda_ = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=arg) for arg in [name] + params],
            varargs=None,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=exp.body
    )
    exp = ast.Expression(body=lambda_)
    ast.fix_missing_locations(exp)
    vars_ = [name for name in available_feature_names
             if sanitizer(name) in used_sanitized_feature_names]

    # pylint: disable=eval-used
    return eval(compile(exp, "<lambda>", mode="eval"), env), params, vars_


class _ParametersSearch(ast.NodeVisitor):
    """
    Find features and parameters:
     - feature: if node is instance of ast.Name and is included in vars_names
     - parameters: if node is instance of ast.Name and is not included
     in functions

    Parameters
    ----------
    vars_names : list of str
        List of all available features names.
        Needed to distinguish between parameters, features and functions.
    functions : list of str
        List of all available functions.
        Needed to distinguish between parameters, features and functions.

    Attributes
    ----------
    parameters : list of str
        List of used parameters.
    variables : list of str
        List of used features.

    """

    def __init__(self, vars_names: List[str], functions: List[str]):
        super().__init__()
        self.__vars_names = vars_names
        self.__functions = functions
        self.__parameters: List[str] = []
        self.__variables: List[str] = []

    @property
    def parameters(self) -> List[str]:
        return self.__parameters

    @property
    def variables(self) -> List[str]:
        return self.__variables

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.__vars_names:
            # don't use Set in order to preserve parameters order
            if node.id not in self.__variables:
                self.__variables.append(node.id)
        elif node.id not in self.__functions:
            # don't use Set in order to preserve parameters order
            if node.id not in self.__parameters:
                self.__parameters.append(node.id)
        return node


class _ReplaceVars(ast.NodeTransformer):
    """
    Replace feature names with X[:, i], where i is index of feature.

    Parameters
    ----------
    name : str
        List of all available features names.
        Needed to distinguish between parameters, features and functions.
    vars_mapper : dict
        Dictionary of used features names and the belonging index from domain.
    functions : list of str
        List of all available functions.

    """

    def __init__(self, name: str, vars_mapper: Dict, functions: List):
        super().__init__()
        self.__name = name
        self.__vars_mapper = vars_mapper
        self.__functions = functions

    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Subscript]:
        if node.id not in self.__vars_mapper or node.id in self.__functions:
            return node
        else:
            n = self.__vars_mapper[node.id]
            return ast.Subscript(
                value=ast.Name(id=self.__name, ctx=ast.Load()),
                slice=ast.ExtSlice(
                    dims=[ast.Slice(lower=None, upper=None, step=None),
                          ast.Index(value=ast.Num(n=n))]),
                ctx=node.ctx
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    housing = Table("housing")
    xdata = housing.X
    ydata = housing.Y

    func = lambda x, a, b, c: a * np.exp(-b * x[:, 0]) + c
    pred = CurveFitLearner(func, ["a", "b", "c"], ["LSTAT"])(housing)(housing)

    plt.plot(xdata[:, 12], ydata, "o")
    indices = np.argsort(xdata[:, 12])
    plt.plot(xdata[indices, 12], pred[indices])
    plt.show()
