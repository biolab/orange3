import operator
from copy import copy
from hashlib import sha1
from collections import namedtuple
import bottleneck as bn
import numpy as np
from scipy.stats import chi2
from Orange.data import Table
from Orange.statistics import contingency
from Orange.classification import Learner, Model
from Orange.preprocess.discretize import EntropyMDL
from Orange.preprocess import RemoveNaNClasses, Impute, Average

__all__ = ["CN2Learner", "CN2UnorderedLearner"]


def argmaxrnd(a, random_seed=None):
    """
    Find the index of the maximum value for a given 1-D numpy array.

    In case of multiple indices corresponding to the maximum value, the
    result is chosen randomly among those. The random number generator
    can be seeded by forwarding a seed value: see function 'hash_dist'.

    Parameters
    ----------
    a : ndarray
        The source array.
    random_seed : int
        RNG seed.

    Returns
    -------
    index : int
        Index of the maximum value.

    Raises
    ------
    ValueError : shape mismatch
        If 'a' has got more than 2 dimensions.

    Notes
    -----
    2-D arrays are also supported to avoid multiple RNG initialisation.
    An array of indices corresponding to the maximum value of each row
    is then returned.
    """
    if a.ndim > 2:
        raise ValueError("argmaxrnd only accepts arrays of up to 2 dim")

    def rndc(x):
        return random.choice((x == bn.nanmax(x)).nonzero()[0])

    random = (np.random if random_seed is None
              else np.random.RandomState(random_seed))

    return rndc(a) if a.ndim == 1 else np.apply_along_axis(rndc, axis=1, arr=a)


def entropy(x):
    """
    Calculate information-theoretic entropy measure for a given
    distribution.

    Parameters
    ----------
    x : ndarray
        Input distribution.

    Returns
    -------
    res : float
        Entropy measure result.
    """
    x = x[x != 0]
    x /= x.sum()
    x *= -np.log2(x)
    return x.sum()


def likelihood_ratio_statistic(x, y):
    """
    Calculate likelihood ratio statistic for given distributions.

    Parameters
    ----------
    x : ndarray
        Observed distribution.
    y : ndarray
        Expected distribution.

    Returns
    -------
    lrs : float
        Likelihood ratio statistic result.
    """
    x[x == 0] = 1e-5
    y[y == 0] = 1e-5
    y *= x.sum() / y.sum()
    lrs = 2 * (x * np.log(x/y)).sum()
    return lrs


def get_dist(Y, domain):
    """
    Determine the class distribution for a given array of
    classifications. Identify the number of classes from the data
    domain.

    Parameters
    ----------
    Y : ndarray, int
        Array of classifications.
    domain : Orange.data.domain.Domain
        Data domain.

    Returns
    -------
    dist : ndarray, int
        Class distribution.
    """
    return np.bincount(Y, minlength=len(domain.class_var.values))


def hash_dist(x):
    """
    For a given distribution, calculate a hash value that can be used to
    seed the RNG.

    Parameters
    ----------
    x : ndarray
        Input distribution.

    Returns
    -------
    hash : int
        Hash function result.
    """
    return int(sha1(bytes(x)).hexdigest(), base=16) & 0xffffffff


class Evaluator:
    def evaluate_rule(self, rule):
        """
        Characterise a search heuristic.

        If lower values indicate better results, return negatives to
        correctly integrate the sorting procedure.

        Parameters
        ----------
        rule : Rule
            Evaluate this rule.

        Returns
        -------
        res : float
            Evaluation function result.
        """
        raise NotImplementedError


class EntropyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        tc = rule.target_class
        dist = rule.curr_class_dist
        x = (np.array([dist[tc], dist.sum() - dist[tc]], dtype=float)
             if tc is not None else dist.astype(float))
        return -entropy(x)


class LaplaceAccuracyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        # as an exception, when target class is not set,
        # the majority class is chosen to stand against
        # all others
        tc = rule.target_class
        dist = rule.curr_class_dist
        if tc is not None:
            k = 2
            target = dist[tc]
        else:
            k = len(dist)
            target = bn.nanmax(dist)
        return (target + 1) / (dist.sum() + k)


class LengthEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        return -rule.length


class Validator:
    def validate_rule(self, rule):
        """
        Characterise a heuristic to avoid the over-fitting of noisy data
        and to reduce computation time.

        Parameters
        ----------
        rule : Rule
            Validate this rule.

        Returns
        -------
        res : bool
            Validation function result.
        """
        raise NotImplementedError


class CustomGeneralValidator(Validator):
    """
    Discard rules that

    - cover less than the minimum required number of examples,
    - offer no additional advantage compared to their parent rule,
    - are too complex.
    """
    def __init__(self, max_rule_length=5, minimum_covered_examples=1):
        self.max_rule_length = max_rule_length
        self.minimum_covered_examples = minimum_covered_examples

    def validate_rule(self, rule):
        num_target_covered = (rule.curr_class_dist[rule.target_class]
                              if rule.target_class is not None
                              else rule.curr_class_dist.sum())

        return (num_target_covered >= self.minimum_covered_examples and
                rule.length <= self.max_rule_length and
                (not np.array_equal(rule.curr_class_dist,
                                    rule.parent_rule.curr_class_dist)
                 if rule.parent_rule is not None else True))


class LRSValidator(Validator):
    """
    To test significance, calculate the likelihood ratio statistic. It
    provides an information-theoretic measure of the (non-commutative)
    distance between two distributions. Under suitable assumptions, it
    can be shown that the statistic is distributed approximately as
    Chi^2 probability distribution with n-1 degrees of freedom. As the
    score lowers, the apparent regularity is more likely observed due to
    chance.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def validate_rule(self, rule):
        if self.alpha >= 1.0 or rule.parent_rule is None:
            return True

        tc = rule.target_class
        dist = rule.curr_class_dist
        p_dist = rule.parent_rule.curr_class_dist

        if tc is not None:
            x = np.array([dist[tc], dist.sum() - dist[tc]], dtype=float)
            y = np.array([p_dist[tc], p_dist.sum() - p_dist[tc]], dtype=float)
        else:
            x = dist.astype(float)
            y = dist.astype(float)

        lrs = likelihood_ratio_statistic(x, y)
        df = len(dist) - 1
        return lrs > 0 and chi2.sf(lrs, df) <= self.alpha


class SearchAlgorithm:
    """
    Implement an algorithm to maneuver through the search space towards
    a better solution, guided by the search heuristics.
    """
    def select_candidates(self, rules):
        """
        Select candidate rules to be further specialised.

        Parameters
        ----------
        rules : Rule list
            An ordered list of rules (best come first).

        Returns
        -------
        candidate_rules : Rule list
            Chosen rules.
        rules : Rule list
            Rules not chosen, i.e. the remainder.
        """
        raise NotImplementedError

    def filter_rules(self, rules):
        """
        Filter rules to be considered in the next iteration.

        Parameters
        ----------
        rules : Rule list
            An ordered list of rules (best come first).

        Returns
        -------
        rules : Rule list
            Rules kept in play.
        """
        raise NotImplementedError


class BeamSearchAlgorithm(SearchAlgorithm):
    """
    Remember the best rule found thus far and monitor a fixed number of
    alternatives (the beam).
    """
    def __init__(self, beam_width=10):
        self.beam_width = beam_width

    def select_candidates(self, rules):
        return rules, []

    def filter_rules(self, rules):
        return rules[:self.beam_width]


class SearchStrategy:
    def initialise_rule(self, X, Y, target_class, base_rules, domain,
                        prior_class_dist, quality_evaluator,
                        complexity_evaluator, significance_validator,
                        general_validator):
        """
        Develop a starting rule.

        Parameters
        ----------
        X, Y : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        base_rules : Rule list
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        prior_class_dist : ndarray
            Data class distribution just before a rule is developed.
        quality_evaluator : Evaluator
            Evaluation algorithm.
        complexity_evaluator : Evaluator
            Evaluation algorithm.
        significance_validator : Validator
            Validation algorithm.
        general_validator : Validator
            Validation algorithm.

        Returns
        -------
        rules : Rule list
            First rules developed in the process of learning a single
            rule.
        """
        raise NotImplementedError

    def refine_rule(self, X, Y, candidate_rule):
        """
        Refine rule.

        Parameters
        ----------
        X, Y : ndarray
            Learning data.
        candidate_rule : Rule
            Refine this rule.

        Returns
        -------
        rules : Rule list
            Descendant rules of 'candidate_rule'.
        """
        raise NotImplementedError


class TopDownSearchStrategy(SearchStrategy):
    """
    An empty starting rule that covers all instances is developed and
    added to the list of candidate rules. The hypothesis space of
    possible rules is searched repeatedly by specialising candidate
    rules.
    """
    def __init__(self, discretise_continuous=False):
        self.discretise_continuous = discretise_continuous
        self.storage = None

    def initialise_rule(self, X, Y, target_class, base_rules, domain,
                        prior_class_dist, quality_evaluator,
                        complexity_evaluator, significance_validator,
                        general_validator):
        rules = []
        default_rule = Rule(domain=domain,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

        default_rule.filter_and_store(X, Y, target_class)
        default_rule.do_evaluate()
        rules.append(default_rule)

        for base_rule in base_rules:
            temp_rule = Rule(selectors=copy(base_rule.selectors),
                             parent_rule=default_rule,
                             domain=domain,
                             prior_class_dist=prior_class_dist,
                             quality_evaluator=quality_evaluator,
                             complexity_evaluator=complexity_evaluator,
                             significance_validator=significance_validator,
                             general_validator=general_validator)

            temp_rule.filter_and_store(X, Y, target_class)
            if temp_rule.is_valid():
                temp_rule.do_evaluate()
                rules.append(temp_rule)

        # optimisation: store covered examples when a selector is found
        self.storage = {}
        return rules

    def refine_rule(self, X, Y, candidate_rule):
        (target_class, candidate_rule_covered_examples,
         candidate_rule_selectors, domain, prior_class_dist,
         quality_evaluator, complexity_evaluator, significance_validator,
         general_validator) = candidate_rule.seed()

        # optimisation: to develop further rules is futile
        if candidate_rule.length == general_validator.max_rule_length:
            return []

        possible_selectors = self.find_new_selectors(
            X[candidate_rule_covered_examples],
            Y[candidate_rule_covered_examples],
            domain, candidate_rule_selectors)

        new_rules = []
        for curr_selector in possible_selectors:
            copied_selectors = copy(candidate_rule_selectors)
            copied_selectors.append(curr_selector)

            new_rule = Rule(selectors=copied_selectors,
                            parent_rule=candidate_rule,
                            domain=domain,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

            if curr_selector not in self.storage:
                self.storage[curr_selector] = curr_selector.filter_data(X)
            # optimisation: faster calc. of covered examples
            smh = candidate_rule_covered_examples & self.storage[curr_selector]
            # to ensure that the covered_examples matrices are of the
            # same size throughout the rule_finder iteration
            new_rule.filter_and_store(X, Y, target_class, smh)
            if new_rule.is_valid():
                new_rule.do_evaluate()
                new_rules.append(new_rule)

        return new_rules

    def find_new_selectors(self, X, Y, domain, existing_selectors):
        existing_selectors = (existing_selectors if existing_selectors is not
                              None else [])

        possible_selectors = []
        # examine covered examples, for each variable
        for i, attribute in enumerate(domain.attributes):
            # if discrete variable
            if attribute.is_discrete:
                # for each unique value, generate all possible selectors
                for val in np.unique(X[:, i]):
                    s1 = Selector(column=i, op="==", value=val)
                    s2 = Selector(column=i, op="!=", value=val)
                    possible_selectors.extend([s1, s2])
            # if continuous variable
            elif attribute.is_continuous:
                # discretise if True
                values = (self.discretise(X, Y, domain, attribute)
                          if self.discretise_continuous
                          else np.unique(X[:, i]))
                # for each unique value, generate all possible selectors
                for val in values:
                    s1 = Selector(column=i, op="<=", value=val)
                    s2 = Selector(column=i, op=">=", value=val)
                    possible_selectors.extend([s1, s2])

        # remove redundant selectors
        possible_selectors = [smh for smh in possible_selectors if
                              smh not in existing_selectors]

        return possible_selectors

    @staticmethod
    def discretise(X, Y, domain, attribute):
        data = Table.from_numpy(domain, X, Y)
        cont = contingency.get_contingency(data, attribute)
        values, counts = cont.values, cont.counts.T
        cut_ind = np.array(EntropyMDL._entropy_discretize_sorted(counts, True))
        return [values[smh] for smh in cut_ind]


class Selector(namedtuple('Selector', 'column, op, value')):
    # define a single rule condition

    OPERATORS = {
        # discrete, nominal variables
        '==': operator.eq,
        '!=': operator.ne,

        # continuous variables
        '<=': operator.le,
        '>=': operator.ge
    }

    def filter_instance(self, x):
        return Selector.OPERATORS[self.op](x[self.column], self.value)

    def filter_data(self, X):
        return Selector.OPERATORS[self[1]](X[:, self[0]], self[2])


class Rule:
    """
    Represent a single rule and keep a reference to its parent. Taking
    into account numpy slicing and memory management, instance
    references are strictly not kept.

    Those can be easily gathered however, by following the trail of
    covered examples from rule to rule, provided that the original
    learning data reference is still known.
    """
    def __init__(self, selectors=None, parent_rule=None, domain=None,
                 prior_class_dist=None, quality_evaluator=None,
                 complexity_evaluator=None, significance_validator=None,
                 general_validator=None):
        """
        Initialise a Rule.

        Parameters
        ----------
        selectors : Selector list
            Rule conditions.
        parent_rule : Rule
            Reference to the parent rule.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        prior_class_dist : ndarray
            Data class distribution just before a rule is developed.
        quality_evaluator : Evaluator
            Evaluation algorithm.
        complexity_evaluator : Evaluator
            Evaluation algorithm.
        significance_validator : Validator
            Validation algorithm.
        general_validator : Validator
            Validation algorithm.
        """
        self.selectors = selectors if selectors is not None else []
        self.parent_rule = parent_rule
        self.domain = domain
        self.prior_class_dist = prior_class_dist
        self.quality_evaluator = quality_evaluator
        self.complexity_evaluator = complexity_evaluator
        self.significance_validator = significance_validator
        self.general_validator = general_validator

        self.target_class = None
        self.covered_examples = None
        self.curr_class_dist = None
        self.quality = None
        self.complexity = None
        self.prediction = None
        self.probabilities = None
        self.length = len(self.selectors)

    def filter_and_store(self, X, Y, target_class, predef_covered=None):
        """
        Apply data and target class to a rule.

        Parameters
        ----------
        X, Y : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        predef_covered : ndarray
            Built-in optimisation variable to enable external
            computation of covered examples.
        """
        self.target_class = target_class
        if predef_covered is not None:
            self.covered_examples = predef_covered
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=bool)
            for selector in self.selectors:
                self.covered_examples &= selector.filter_data(X)
        self.curr_class_dist = get_dist(Y[self.covered_examples], self.domain)

    def is_valid(self):
        return self.general_validator.validate_rule(self)

    def is_significant(self):
        return self.significance_validator.validate_rule(self)

    def do_evaluate(self):
        self.quality = self.quality_evaluator.evaluate_rule(self)
        self.complexity = self.complexity_evaluator.evaluate_rule(self)

    def evaluate_instance(self, x):
        """
        Evaluate a single instance.

        Parameters
        ----------
        x : ndarray
            Evaluate this instance.

        Returns
        -------
        res : bool
            True, if the rule covers 'x'.
        """
        return all(selector.filter_instance(x) for selector in self.selectors)

    def evaluate_data(self, X):
        """
        Evaluate several instances concurrently.

        Parameters
        ----------
        X : ndarray
            Evaluate this data.

        Returns
        -------
        res : ndarray, bool
            Array of evaluations, size of ''X.shape[0]''.
        """
        curr_covered = np.ones(X.shape[0], dtype=bool)
        for selector in self.selectors:
            curr_covered &= selector.filter_data(X)
        return curr_covered

    def create_model(self):
        """
        For unordered rules, the prediction is set to be the target
        class. In contrast, ordered rules predict the majority class of
        the covered examples.
        """
        # laplace class probabilities
        self.probabilities = ((self.curr_class_dist + 1) /
                              (self.curr_class_dist.sum() +
                               len(self.curr_class_dist)))
        # predicted class
        self.prediction = (self.target_class if self.target_class is not None
                           else argmaxrnd(self.curr_class_dist))

    def seed(self):
        """
        Forward relevant information to develop new rules.
        """
        return (self.target_class, self.covered_examples, self.selectors,
                self.domain, self.prior_class_dist, self.quality_evaluator,
                self.complexity_evaluator, self.significance_validator,
                self.general_validator)

    def __str__(self):
        attributes = self.domain.attributes
        class_var = self.domain.class_var

        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                                 (str(attributes[s.column].values[int(s.value)])
                                  if attributes[s.column].is_discrete
                                  else str(s.value)) for s in self.selectors])
        else:
            cond = "TRUE"

        outcome = class_var.name + "=" + class_var.values[self.prediction]
        return "IF {} THEN {} ".format(cond, outcome)


class RuleFinder:
    def __init__(self):
        self.search_algorithm = BeamSearchAlgorithm()
        self.search_strategy = TopDownSearchStrategy()

        # search heuristics
        self.quality_evaluator = EntropyEvaluator()
        self.complexity_evaluator = LengthEvaluator()
        # heuristics to avoid the over-fitting of noisy data
        self.significance_validator = LRSValidator()
        self.general_validator = CustomGeneralValidator()

    def __call__(self, X, Y, target_class, base_rules, domain):
        """
        Return a single rule.

        Parameters
        ----------
        X, Y : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        base_rules : Rule list
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.

        Returns
        -------
        best_rule : Rule
            Highest quality rule discovered.
        """
        def rcmp(rule):
            return rule.quality, rule.complexity

        prior_class_dist = get_dist(Y, domain)
        rules = self.search_strategy.initialise_rule(
            X, Y, target_class, base_rules, domain, prior_class_dist,
            self.quality_evaluator, self.complexity_evaluator,
            self.significance_validator, self.general_validator)

        rules = sorted(rules, key=rcmp, reverse=True)
        best_rule = rules[0]

        while len(rules) > 0:
            cand_rules, rules = self.search_algorithm.select_candidates(rules)
            for cand_rule in cand_rules:
                new_rules = self.search_strategy.refine_rule(X, Y, cand_rule)
                rules.extend(new_rules)
                for new_rule in new_rules:
                    if (new_rule.quality > best_rule.quality and
                            new_rule.is_significant()):
                        best_rule = new_rule

            rules = sorted(rules, key=rcmp, reverse=True)
            rules = self.search_algorithm.filter_rules(rules)

        best_rule.create_model()
        return best_rule


class _RuleLearner(Learner):
    """
    A base rule induction learner. Returns a rule classifier if called
    with data.

    Separate and conquer strategy is applied, allowing for different
    rule learning algorithms to be easily implemented by connecting
    together predefined components. In essence, learning instances are
    covered and removed following a chosen rule. The process is repeated
    while learning set examples remain. To evaluate found hypotheses and
    to choose the best rule in each iteration, search heuristics are
    used. Primarily, rule class distribution is the decisive
    determinant.

    The over-fitting of noisy data is avoided by preferring simpler,
    shorter rules even if the accuracy of more complex rules is higher.

    References
    ----------
    .. [1] "Separate-and-Conquer Rule Learning", Johannes Fürnkranz,
           Artificial Intelligence Review 13, 3-54, 1999
    """
    preprocessors = [RemoveNaNClasses(), Impute(Average())]

    def __init__(self, preprocessors=None, base_rules=None):
        """
        Initialise a RuleLearner object.

        Constrain the algorithm with a list of base rules. Also create
        a RuleFinder object. Set search bias and over-fitting avoidance
        bias parameters by selecting its components.

        Parameters
        ----------
        base_rules : Rule list
            An optional list of initial rules to constrain the search.
        """
        super().__init__(preprocessors=preprocessors)
        self.base_rules = base_rules if base_rules is not None else []
        self.rule_finder = RuleFinder()

    def fit(self, X, Y, W=None):
        raise NotImplementedError

    def find_rules(self, X, Y, target_class, base_rules, domain):
        rule_list = []
        while not self.data_stopping(X, Y, target_class):
            new_rule = self.rule_finder(X, Y, target_class, base_rules, domain)
            if self.rule_stopping(new_rule):
                break
            X, Y = self.cover_and_remove(X, Y, new_rule)
            rule_list.append(new_rule)
        return rule_list

    @staticmethod
    def rule_stopping(new_rule, alpha=1.0):
        if alpha >= 1.0:
            return False

        tc = new_rule.target_class
        dist = new_rule.curr_class_dist
        p_dist = new_rule.prior_class_dist

        if tc is not None:
            x = np.array([dist[tc], dist.sum() - dist[tc]], dtype=float)
            y = np.array([p_dist[tc], p_dist.sum() - p_dist[tc]], dtype=float)
        else:
            x = dist.astype(float)
            y = dist.astype(float)

        lrs = likelihood_ratio_statistic(x, y)
        df = len(dist) - 1
        return not (lrs > 0 and chi2.sf(lrs, df) <= alpha)

    @staticmethod
    def data_stopping(X, Y, target_class):
        # stop if no positive examples
        return (Y.size == 0 or (target_class is not None and
                                target_class not in Y))

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = np.logical_not(new_rule.covered_examples)
        return X[examples_to_keep], Y[examples_to_keep]


class _RuleClassifier(Model):
    """
    A rule induction classifier. Instances are classified following
    either an unordered set of rules or a decision list.
    """
    def __init__(self, domain=None, rule_list=None):
        super().__init__(domain)
        self.domain = domain
        self.rule_list = rule_list if rule_list is not None else []

    def predict(self, X):
        raise NotImplementedError


class CN2Learner(_RuleLearner):
    """
    Classic CN2 inducer that constructs a list of ordered rules. To
    evaluate found hypotheses, entropy measure is used. Returns a
    CN2Classifier if called with data.

    References
    ----------
    .. [1] "The CN2 Induction Algorithm", Peter Clark and Tim Niblett,
           Machine Learning Journal, 3 (4), pp261-283, (1989)
    """
    name = 'CN2 inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = EntropyEvaluator()

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = self.find_rules(X, Y, None, self.base_rules, self.domain)
        return CN2Classifier(domain=self.domain, rule_list=rule_list)


class CN2Classifier(_RuleClassifier):
    def predict(self, X):
        """
        Following a decision list, for each instance, rules are tried in
        order until one fires.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, int
            Array of classifications.
        """
        num_classes = len(self.domain.class_var.values)
        probabilities = np.array([np.zeros(num_classes, dtype=float)
                                  for _ in range(X.shape[0])])

        status = np.ones(X.shape[0], dtype=bool)
        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            probabilities[curr_covered] = rule.probabilities
        return probabilities


class CN2UnorderedLearner(_RuleLearner):
    """
    Unordered CN2 inducer that constructs a set of unordered rules. To
    evaluate found hypotheses, Laplace accuracy measure is used. Returns
    a CN2UnorderedClassifier if called with data.

    Notes
    -----
    Rules are learnt for each class (target class) independently, in
    regard to the original learning data. When a rule has been found,
    only covered examples of that class are removed. This is because now
    each rule must independently stand against all negatives.

    References
    ----------
    .. [1] "Rule Induction with CN2: Some Recent Improvements", Peter
           Clark and Robin Boswell, Machine Learning - Proceedings of
           the 5th European Conference (EWSL-91), pp151-163, 1991
    """
    name = 'CN2Unordered inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = []
        for curr_class in range(len(self.domain.class_var.values)):
            r = self.find_rules(X, Y, curr_class, self.base_rules, self.domain)
            rule_list.extend(r)
        return CN2UnorderedClassifier(domain=self.domain, rule_list=rule_list)

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = Y == new_rule.target_class
        examples_to_keep &= new_rule.covered_examples
        examples_to_keep = np.logical_not(examples_to_keep)
        return X[examples_to_keep], Y[examples_to_keep]


class CN2UnorderedClassifier(_RuleClassifier):
    def predict(self, X):
        """
        Following an unordered set of rules, for each instance, all
        rules are tried and those that fire are collected. If a clash
        occurs (i.e. more than one class is predicted), class
        distributions of all collected rules are summed and the most
        probable class is predicted.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, int
            Array of classifications.
        """
        num_classes = len(self.domain.class_var.values)
        probabilities = np.array([np.zeros(num_classes, dtype=float)
                                  for _ in range(X.shape[0])])

        resolve_clash = np.zeros(X.shape[0], dtype=int)
        clashes_total_weight = np.vstack(np.zeros(X.shape[0], dtype=int))
        clashes = np.copy(probabilities)

        status = np.ones(X.shape[0], dtype=bool)
        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            resolve_clash += curr_covered
            temp = rule.curr_class_dist.sum()
            clashes[curr_covered] += rule.probabilities * temp
            clashes_total_weight[curr_covered] += temp
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            probabilities[curr_covered] = rule.probabilities

        resolve_clash = resolve_clash > 1
        clashes[resolve_clash] /= clashes_total_weight[resolve_clash]
        probabilities[resolve_clash] = clashes[resolve_clash]
        return probabilities


def main():
    data = Table('titanic')
    learner = CN2Learner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule)

    print()

    data = Table('adult')
    learner = CN2UnorderedLearner()
    learner.rule_finder.general_validator.max_rule_length = 5
    learner.rule_finder.general_validator.minimum_covered_examples = 100
    # learner.rule_finder.significance_validator.alpha = 0.9
    learner.rule_finder.search_algorithm.beam_width = 10
    learner.rule_finder.search_strategy.discretise_continuous = True

    import time
    start = time.time()
    classifier = learner(data)
    end = time.time()

    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist(), rule.quality)

    print(len(classifier.rule_list))
    print(end - start)

if __name__ == "__main__":
    main()
