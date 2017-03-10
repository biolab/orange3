"""
Induction of rules works by finding a rule that covers some learning instances,
removing these instances, and repeating this until all instances are covered.
Rules are scored by heuristics such as impurity of class distribution of
covered instances. The module includes common rule-learning algorithms,
and allows for replacing rule search strategies, scoring and other
components.
"""

import operator
from copy import copy
from hashlib import sha1
from collections import namedtuple
import bottleneck as bn
import numpy as np
from scipy.stats import chi2

from Orange.data import Table, _contingency
from Orange.classification import Learner, Model
from Orange.preprocess.discretize import EntropyMDL
from Orange.preprocess import RemoveNaNColumns, RemoveNaNClasses, Impute

__all__ = ["CN2Learner", "CN2UnorderedLearner", "CN2SDLearner",
           "CN2SDUnorderedLearner"]


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


def get_dist(Y, W, domain):
    """
    Determine the class distribution for a given array of
    classifications. Identify the number of classes from the data
    domain.

    Parameters
    ----------
    Y : ndarray, int
        Array of classifications.
    W : ndarray, float
        Weights.
    domain : Orange.data.domain.Domain
        Data domain.

    Returns
    -------
    dist : ndarray
        Class distribution.
    """
    return np.bincount(Y, weights=W, minlength=len(domain.class_var.values))


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
    return int(sha1(np.ascontiguousarray(x).data)
               .hexdigest(), base=16) & 0xffffffff


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


class WeightedRelativeAccuracyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        tc = rule.target_class
        dist = rule.curr_class_dist
        p_dist = rule.prior_class_dist
        dist_sum, p_dist_sum = dist.sum(), p_dist.sum()
        d_modus = argmaxrnd(dist)

        if tc is not None:
            p_cond = dist_sum / p_dist_sum
            # p_cond = dist[tc] / p_dist[tc]
            p_true_positive = dist[tc] / dist_sum
            p_class = p_dist[tc] / p_dist_sum
        else:
            # generality of the rule
            p_cond = dist_sum / p_dist_sum
            # true positives of class c
            p_true_positive = dist[d_modus] / dist_sum
            # prior probability of class c
            p_class = p_dist[d_modus] / p_dist_sum

        return (p_cond * (p_true_positive - p_class) if
                p_true_positive > p_class
                else (p_true_positive - p_class) / max(p_cond, 1e-6))


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


class GuardianValidator(Validator):
    """
    Discard rules that

    - cover less than the minimum required number of examples,
    - offer no additional advantage compared to their parent rule,
    - are too complex.
    """
    def __init__(self, max_rule_length=5, min_covered_examples=1):
        self.max_rule_length = max_rule_length
        self.min_covered_examples = min_covered_examples

    def validate_rule(self, rule):
        num_target_covered = (rule.curr_class_dist[rule.target_class]
                              if rule.target_class is not None
                              else rule.curr_class_dist.sum())

        return (num_target_covered >= self.min_covered_examples and
                rule.length <= self.max_rule_length and
                (True if rule.parent_rule is None
                 else not np.array_equal(rule.curr_class_dist,
                                         rule.parent_rule.curr_class_dist)))


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
    def __init__(self, parent_alpha=1.0, default_alpha=1.0):
        self.parent_alpha = parent_alpha
        self.default_alpha = default_alpha

    def validate_rule(self, rule, _default=False):
        tc = rule.target_class
        dist = rule.curr_class_dist

        if self.default_alpha < 1.0:
            sig = self.test_sig(dist, rule.initial_class_dist, tc, self.default_alpha)
            if not sig:
                return False
        if self.parent_alpha < 1.0 and rule.parent_rule is not None:
            expdist = rule.parent_rule.curr_class_dist
            alpha = self.parent_alpha
            return self.test_sig(dist, expdist, tc, alpha)
        return True

    def test_sig(self, obsdist, expdist, tc, alpha):
        if tc is not None:
            x = np.array([obsdist[tc], obsdist.sum() - obsdist[tc]], dtype=float)
            y = np.array([expdist[tc], expdist.sum() - expdist[tc]], dtype=float)
        else:
            x = obsdist.astype(float)
            y = expdist.astype(float)

        lrs = likelihood_ratio_statistic(x, y)
        df = len(obsdist) - 1
        return lrs > 0 and chi2.sf(lrs, df) <= alpha


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
        rules : list of Rule
            An ordered list of rules (best come first).

        Returns
        -------
        candidate_rules : list of Rule
            Chosen rules.
        rules : list of Rule
            Rules not chosen, i.e. the remainder.
        """
        raise NotImplementedError

    def filter_rules(self, rules):
        """
        Filter rules to be considered in the next iteration.

        Parameters
        ----------
        rules : list of Rule
            An ordered list of rules (best come first).

        Returns
        -------
        rules : list of Rule
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
    def initialise_rule(self, X, Y, W, target_class, base_rules, domain,
                        initial_class_dist, prior_class_dist,
                        quality_evaluator, complexity_evaluator,
                        significance_validator, general_validator):
        """
        Develop a starting rule.

        Parameters
        ----------
        X, Y : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        base_rules : list of Rule
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        initial_class_dist : ndarray
            Data class distribution in regard to the whole learning set.
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
        rules : list of Rule
            First rules developed in the process of learning a single
            rule.
        """
        raise NotImplementedError

    def refine_rule(self, X, Y, W, candidate_rule):
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
        rules : list of Rule
            Descendant rules of 'candidate_rule'.
        """
        raise NotImplementedError


class TopDownSearchStrategy(SearchStrategy):
    """
    If no base rules are given, an empty starting rule that covers all
    instances is developed. The hypothesis space of possible rules is
    then searched repeatedly by specialising candidate rules.
    """
    def __init__(self, constrain_continuous=True, evaluate=True):
        self.constrain_continuous = constrain_continuous
        self.storage = None
        self.evaluate = evaluate

    def initialise_rule(self, X, Y, W, target_class, base_rules, domain,
                        initial_class_dist, prior_class_dist,
                        quality_evaluator, complexity_evaluator,
                        significance_validator, general_validator):
        rules = []
        default_rule = Rule(domain=domain,
                            initial_class_dist=initial_class_dist,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

        default_rule.filter_and_store(X, Y, W, target_class)
        if not base_rules and default_rule.is_valid():
            if self.evaluate:
                default_rule.do_evaluate()
            rules.append(default_rule)

        for base_rule in base_rules:
            temp_rule = Rule(selectors=copy(base_rule.selectors),
                             domain=domain,
                             initial_class_dist=initial_class_dist,
                             prior_class_dist=prior_class_dist,
                             quality_evaluator=quality_evaluator,
                             complexity_evaluator=complexity_evaluator,
                             significance_validator=significance_validator,
                             general_validator=general_validator)

            temp_rule.filter_and_store(X, Y, W, target_class)
            if temp_rule.is_valid():
                if self.evaluate:
                    temp_rule.do_evaluate()
                rules.append(temp_rule)

        # optimisation: store covered examples when a selector is found
        self.storage = {}
        return rules

    def refine_rule(self, X, Y, W, candidate_rule):
        (target_class, candidate_rule_covered_examples,
         candidate_rule_selectors, domain, initial_class_dist,
         prior_class_dist, quality_evaluator, complexity_evaluator,
         significance_validator, general_validator) = candidate_rule.seed()

        # optimisation: to develop further rules is futile
        if candidate_rule.length == general_validator.max_rule_length:
            return []

        possible_selectors = self.find_new_selectors(
            X[candidate_rule_covered_examples],
            Y[candidate_rule_covered_examples],
            W[candidate_rule_covered_examples]
            if W is not None else None,
            domain, candidate_rule_selectors)

        new_rules = []
        for curr_selector in possible_selectors:
            copied_selectors = copy(candidate_rule_selectors)
            copied_selectors.append(curr_selector)

            new_rule = Rule(selectors=copied_selectors,
                            parent_rule=candidate_rule,
                            domain=domain,
                            initial_class_dist=initial_class_dist,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

            if curr_selector not in self.storage:
                self.storage[curr_selector] = curr_selector.filter_data(X)
            # optimisation: faster calc. of covered examples
            pdc = candidate_rule_covered_examples & self.storage[curr_selector]
            # to ensure that the covered_examples matrices are of
            # the same size throughout the rule_finder iteration
            new_rule.filter_and_store(X, Y, W, target_class, predef_covered=pdc)
            if new_rule.is_valid():
                if self.evaluate:
                    new_rule.do_evaluate()
                new_rules.append(new_rule)

        return new_rules

    def find_new_selectors(self, X, Y, W, domain, existing_selectors):
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
                # choose best thresholds if constrain_continuous is True
                values = (self.discretize(X[:, i], Y, W, domain)
                          if self.constrain_continuous
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
    def discretize(X, Y, W, domain):
        values, counts, _ = _contingency.contingency_floatarray(
            X, Y.astype(dtype=np.int8), len(domain.class_var.values), W)
        cut_ind = np.array(EntropyMDL._entropy_discretize_sorted(counts.T, True))
        return [values[smh] for smh in cut_ind]


class Selector(namedtuple('Selector', 'column, op, value')):
    """
    Define a single rule condition.
    """
    OPERATORS = {
        # discrete, nominal variables
        '==': operator.eq,
        '!=': operator.ne,

        # continuous variables
        '<=': operator.le,
        '>=': operator.ge
    }

    def filter_instance(self, x):
        """
        Filter a single instance.

        Parameters
        ----------
        x : ndarray
            Evaluate this instance.

        Returns
        -------
        res : bool
            True, if the rule covers 'x'.
        """
        return Selector.OPERATORS[self[1]](x[self[0]], self[2])

    def filter_data(self, X):
        """
        Filter several instances concurrently.

        Parameters
        ----------
        X : ndarray
            Evaluate this data.

        Returns
        -------
        res : ndarray, bool
            Array of evaluations.
        """
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
                 initial_class_dist=None, prior_class_dist=None,
                 quality_evaluator=None, complexity_evaluator=None,
                 significance_validator=None, general_validator=None):
        """
        Initialise a Rule.

        Parameters
        ----------
        selectors : list of Selector
            Rule conditions.
        parent_rule : Rule
            Reference to the parent rule.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        initial_class_dist : ndarray
            Data class distribution in regard to the whole learning set.
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
        self.initial_class_dist = initial_class_dist
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

    def filter_and_store(self, X, Y, W, target_class, predef_covered=None):
        """
        Apply data and target class to a rule.

        Parameters
        ----------
        X, Y, W : ndarray
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

        self.curr_class_dist = get_dist(Y[self.covered_examples],
                                        W[self.covered_examples]
                                        if W is not None else None,
                                        self.domain)

    def is_valid(self):
        """
        Return True if the rule passes the general validator's
        requirements.
        """
        return self.general_validator.validate_rule(self)

    def is_significant(self, *args, **kwargs):
        """
        Return True if the rule passes the significance validator's
        requirements (is significant in regard to its parent).
        """
        return self.significance_validator.validate_rule(self, *args, **kwargs)

    def do_evaluate(self):
        """
        Evaluate the rule's quality and complexity.
        """
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
            Array of evaluations.
        """
        curr_covered = np.ones(X.shape[0], dtype=bool)
        for selector in self.selectors:
            curr_covered &= selector.filter_data(X)
        return curr_covered

    def create_model(self):
        """
        Determine prediction class probabilities.
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
                self.domain, self.initial_class_dist, self.prior_class_dist,
                self.quality_evaluator, self.complexity_evaluator,
                self.significance_validator, self.general_validator)

    def __eq__(self, other):
        # return self.selectors == other.selectors
        return np.array_equal(self.covered_examples, other.covered_examples)

    def __len__(self):
        return len(self.selectors)

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


class RuleHuntress:
    """
    An experimental implementation of the CN2-R algorithm.
    """
    pass


class RuleHunter:
    def __init__(self):
        self.search_algorithm = BeamSearchAlgorithm()
        self.search_strategy = TopDownSearchStrategy()

        # search heuristics
        self.quality_evaluator = EntropyEvaluator()
        self.complexity_evaluator = LengthEvaluator()
        # heuristics to avoid the over-fitting of noisy data
        self.general_validator = GuardianValidator()
        self.significance_validator = LRSValidator()

    def __call__(self, X, Y, W, target_class, base_rules, domain,
                 initial_class_dist, existing_rules):
        """
        Return a single rule.

        The search is guided by search heuristics (evaluators) and
        controlled by over-fitting avoidance heuristics (validators).
        Search strategy creates and refines rules, whereas search
        algorithm maneuvers through the search space towards a better
        solution.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        base_rules : list of Rule
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        initial_class_dist : ndarray
            Data class distribution in regard to the whole learning set.
        existing_rules : list of Rule
            Rules found in previous iterations (to avoid duplicates).

        Returns
        -------
        best_rule : Rule
            Highest quality rule discovered.
        """
        def rcmp(rule):
            return rule.quality, rule.complexity

        prior_class_dist = get_dist(Y, W, domain)
        rules = self.search_strategy.initialise_rule(
            X, Y, W, target_class, base_rules, domain,
            initial_class_dist, prior_class_dist,
            self.quality_evaluator, self.complexity_evaluator,
            self.significance_validator, self.general_validator)

        if not rules:
            return None

        rules = sorted(rules, key=rcmp, reverse=True)
        best_rule = rules[0]

        while len(rules) > 0:
            candidates, rules = self.search_algorithm.select_candidates(rules)
            for candidate_rule in candidates:
                new_rules = self.search_strategy.refine_rule(
                    X, Y, W, candidate_rule)
                rules.extend(new_rules)
                for new_rule in new_rules:
                    if (new_rule.quality > best_rule.quality and
                            new_rule.is_significant() and
                            new_rule not in existing_rules):
                        best_rule = new_rule

            rules = sorted(rules, key=rcmp, reverse=True)
            rules = self.search_algorithm.filter_rules(rules)

        best_rule.create_model()
        return best_rule if best_rule not in existing_rules else None


class _RuleLearner(Learner):
    """
    A base rule induction learner. Descendants should return a relevant
    classifier if called with data.

    Separate and conquer strategy is applied, allowing for different
    rule learning algorithms to be easily implemented by connecting
    together predefined components. In essence, learning instances are
    covered and removed following a chosen rule. The process is repeated
    while learning set examples remain.

    To evaluate found hypotheses and to choose the best rule in each
    iteration, search heuristics are used. Primarily, rule class
    distribution is the decisive determinant. The over-fitting of noisy
    data is avoided by preferring simpler, shorter rules even if the
    accuracy of more complex rules is higher.

    References
    ----------
    .. [1] "Separate-and-Conquer Rule Learning", Johannes Fürnkranz,
           Artificial Intelligence Review 13, 3-54, 1999
    """
    preprocessors = [RemoveNaNColumns(), RemoveNaNClasses(), Impute()]

    def __init__(self, preprocessors=None, base_rules=None):
        """
        Constrain the search algorithm with a list of base rules.

        Assign relevant functions to regulate the top-level control
        procedure (find_rules). Specify when the algorithm should stop
        the search (data_stopping, rule_stopping) and how instances
        covered are removed/adjusted (cover_and_remove) after finding a
        single rule.

        Also initialise a rule finder (RuleHunter being one possible
        implementation) to control the bottom-level search procedure.
        Set search bias and over-fitting avoidance bias parameters by
        selecting its components.

        Parameters
        ----------
        preprocessors : list of Preprocess
            A sequence of data preprocessors to apply on data prior to
            fitting the model.
        base_rules : list of Rule
            An optional list of initial rules to constrain the search.
        """
        super().__init__(preprocessors=preprocessors)
        self.base_rules = base_rules if base_rules is not None else []
        self.rule_finder = RuleHunter()

        self.data_stopping = self.positive_remaining_data_stopping
        self.cover_and_remove = self.exclusive_cover_and_remove
        self.rule_stopping = self.lrs_significance_rule_stopping

    # base_rules and domain not accessed using self to avoid
    # possible crashes and to enable quick use of the algorithm
    def find_rules(self, X, Y, W, target_class, base_rules, domain):
        """
        The top-level control procedure of the separate-and-conquer
        algorithm. For given data and target class (may be None), return
        a list of rules which all must strictly adhere to the
        requirements of rule finder's validators.

        To induce decision lists (ordered rules), set target class to
        None. Best rule conditions are found and the majority class is
        assigned in the rule head.

        To induce rule sets (unordered rules), learn rules for each
        class individually, in regard to the original learning data.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        base_rules : list of Rule
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.

        Returns
        -------
        rule_list : list of Rule
            Induced rules.
        """
        initial_class_dist = get_dist(Y, W, domain)
        rule_list = []

        # while data allows, continuously find new rules,
        # break the loop if min. requirements cannot be met,
        # after finding a rule, remove the instances covered
        while not self.data_stopping(X, Y, W, target_class):

            # generate a new rule that has not been seen before
            new_rule = self.rule_finder(X, Y, W, target_class, base_rules,
                                        domain, initial_class_dist, rule_list)

            # None when no new, unique rules that pass
            # the general requirements can be found
            if new_rule is None or self.rule_stopping(new_rule):
                break

            # exclusive or weighted
            X, Y, W = self.cover_and_remove(X, Y, W, new_rule)
            rule_list.append(new_rule)

        return rule_list

    def positive_remaining_data_stopping(self, X, Y, W, target_class):
        """
        Data stopping.

        If the minimum number of required covered examples can no longer
        be met, return True and conclude rule induction.

        Important
        ---------
        This function is inherently linked to the GuardianValidator. If
        a different general validator is used, it is imperative that the
        minimum number of required covered examples is enforced or this
        function overridden. The parameter min_covered_examples must be
        accessible, else an exception is raised.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        target_class : int
            Index of the class to model.

        Returns
        -------
        res : bool
            Whether or not rule induction will be stopped.
        """
        tc = target_class
        dist = get_dist(Y, W, self.domain)
        general_validator = self.rule_finder.general_validator
        num_possible = dist[tc] if tc is not None else dist.sum()
        return num_possible < general_validator.min_covered_examples

    @staticmethod
    def lrs_significance_rule_stopping(new_rule):
        """
        Rule stopping.

        If the latest rule is not found to be relevant in regard to the
        initial class distribution, return True and conclude rule
        induction.

        Important
        ---------
        This function is inherently linked to the LRSValidator. If a
        different general validator is used, it is imperative that this
        function is overridden.

        Parameters
        ----------
        new_rule : Rule
            Evaluate this rule.

        Returns
        -------
        res : bool
            Whether or not rule induction will be stopped.
        """
        return not new_rule.is_significant(_default=True)

    @staticmethod
    def exclusive_cover_and_remove(X, Y, W, new_rule):
        """
        Cover and remove.

        After covering a learning instance, remove it from further
        consideration.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        new_rule : Rule
            Remove learning instances covered by this rule.

        Returns
        -------
        X, Y, W : ndarray
            Learning data subset.
        """
        examples_to_keep = new_rule.covered_examples.copy()
        if new_rule.target_class is not None:
            examples_to_keep &= Y == new_rule.target_class
        examples_to_keep = np.logical_not(examples_to_keep)
        W = W[examples_to_keep] if W is not None else None
        return X[examples_to_keep], Y[examples_to_keep], W

    def weighted_cover_and_remove(self, X, Y, W, new_rule):
        """
        Cover and remove.

        After covering a learning instance, decrease its weight and
        in-turn decrease its impact on further iterations of the
        algorithm.

        Important
        ---------
        To use this function, learner parameter gamma must be set.
        Weights are adjusted inplace.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        new_rule : Rule
            Adjust learning instances covered by this rule.

        Returns
        -------
        X, Y, W : ndarray
            Adjusted learning data.
        """
        examples_to_weigh = new_rule.covered_examples.copy()
        if new_rule.target_class is not None:
            examples_to_weigh &= Y == new_rule.target_class
        if W is None:
            W = np.ones(X.shape[0])
        W[examples_to_weigh] *= self.gamma
        return X, Y, W

    def generate_default_rule(self, X, Y, W, domain):
        """
        Generate a default rule, which mimics a majority classifier.
        Specific to each individual rule inducer, the application of the
        default rule varies. To predict an instance, a single default
        rule should be considered.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.

        Returns
        -------
        default_rule : Rule
            Default rule with no selectors that covers all learning
            instances and mimics a majority classifier.
        """
        rf = self.rule_finder
        dist = get_dist(Y, W, domain)

        default_rule = Rule(None, None, self.domain, dist, dist,
                            rf.quality_evaluator, rf.complexity_evaluator,
                            rf.significance_validator, rf.general_validator)

        default_rule.filter_and_store(X, Y, W, None)
        default_rule.do_evaluate()
        default_rule.create_model()
        return default_rule


class _RuleClassifier(Model):
    """
    A rule induction classifier.

    Descendants classify instances following either an unordered set of
    rules or a decision list.
    """
    def __init__(self, domain=None, rule_list=None):
        super().__init__(domain)
        self.domain = domain
        self.rule_list = rule_list if rule_list is not None else []

    def predict(self, X):
        raise NotImplementedError

    def ordered_predict(self, X):
        """
        Following a decision list, for each instance, rules are tried in
        order until one fires.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, float
            Probabilistic classification.
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

    def unordered_predict(self, X):
        """
        Following an unordered set of rules, for each instance, all
        rules are tried and those that fire are collected. If a clash
        occurs, class probabilities (predictions) of all collected rules
        are summed and weighted.

        Notes
        -----
        If no other rules fire, the default rule prediction is used.
        Any other empty rules (a fallout most common in the domain of
        weighted covering algorithms) are ignored.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, float
            Probabilistic classification.
        """
        num_classes = len(self.domain.class_var.values)
        probabilities = np.array([np.zeros(num_classes, dtype=float)
                                  for _ in range(X.shape[0])])

        num_hits = np.zeros(X.shape[0], dtype=int)
        total_weight = np.vstack(np.zeros(X.shape[0], dtype=float))
        for rule in self.rule_list[:-1]:
            if rule.length > 0:
                curr_covered = rule.evaluate_data(X)
                num_hits += curr_covered
                temp = rule.curr_class_dist.sum()
                probabilities[curr_covered] += rule.probabilities * temp
                total_weight[curr_covered] += temp

        default_rule = self.rule_list[-1]
        weigh_down = num_hits > 0
        apply_default = num_hits == 0

        probabilities[weigh_down] /= total_weight[weigh_down]
        probabilities[apply_default] = default_rule.probabilities
        return probabilities


class _BaseCN2Learner(_RuleLearner):
    """
    Base CN2 Learner used to extend CN2 rule induction algorithms.
    """
    def __init__(self, preprocessors=None, base_rules=None, beam_width=5,
                 constrain_continuous=True, min_covered_examples=1,
                 max_rule_length=5, default_alpha=1.0, parent_alpha=1.0):
        super().__init__(preprocessors, base_rules)
        rf = self.rule_finder
        rf.search_algorithm.beam_width = beam_width
        rf.search_strategy.constrain_continuous = constrain_continuous
        rf.general_validator.min_covered_examples = min_covered_examples
        rf.general_validator.max_rule_length = max_rule_length
        rf.significance_validator.default_alpha = default_alpha
        rf.significance_validator.parent_alpha = parent_alpha

    def fit(self, X, Y, W=None):
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
    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = EntropyEvaluator()

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = self.find_rules(X, Y, W, None, self.base_rules, self.domain)
        # add the default rule, if required
        if not rule_list or rule_list and rule_list[-1].length > 0:
            rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
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
        res : ndarray, float
            Probabilistic classification.
        """
        return self.ordered_predict(X)


class CN2UnorderedLearner(_RuleLearner):
    """
    Construct a set of unordered rules.

    Rules are learnt for each class individually and scored
    by the relative frequency of the class corrected by the Laplace correction.
    After adding a rule, only the covered examples of that class are removed.

    The code below loads the *iris* data set (four continuous attributes
    and a discrete class) and fits the learner.

    .. literalinclude:: code/classification-cn2ruleinduction1.py

    References
    ----------
    .. [1] "Rule Induction with CN2: Some Recent Improvements", Peter
           Clark and Robin Boswell, Machine Learning - Proceedings of
           the 5th European Conference (EWSL-91), pp151-163, 1991
    """
    name = 'CN2 unordered inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = []
        for curr_class in range(len(self.domain.class_var.values)):
            rule_list.extend(self.find_rules(X, Y, W, curr_class,
                                             self.base_rules, self.domain))
        # add the default rule
        rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
        return CN2UnorderedClassifier(domain=self.domain, rule_list=rule_list)


class CN2UnorderedClassifier(_RuleClassifier):
    def predict(self, X):
        """
        Following an unordered set of rules, for each instance, all
        rules are tried and those that fire are collected. If a clash
        occurs, class probabilities (predictions) of all collected rules
        are summed and weighted.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, float
            Probabilistic classification.
        """
        return self.unordered_predict(X)


class CN2SDLearner(_RuleLearner):
    """
    Ordered CN2SD inducer that constructs a list of ordered rules. To
    evaluate found hypotheses, Weighted relative accuracy measure is
    used. Returns a CN2SDClassifier if called with data.

    In this setting, ordered rule induction refers exclusively to
    finding best rule conditions and assigning the majority class in the
    rule head (target class is set to None). To later predict instances,
    rules will be regarded as unordered.

    Notes
    -----
    A weighted covering algorithm is applied, in which subsequently
    induced rules also represent interesting and sufficiently large
    subgroups of the population. Covered positive examples are not
    deleted from the learning set, rather their weight is reduced.

    The algorithm demonstrates how classification rule learning
    (predictive induction) can be adapted to subgroup discovery, a task
    at the intersection of predictive and descriptive induction.

    References
    ----------
    .. [1] "Subgroup Discovery with CN2-SD", Nada Lavrač et al., Journal
           of Machine Learning Research 5 (2004), 153-188, 2004
    """
    name = 'CN2-SD inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = WeightedRelativeAccuracyEvaluator()
        self.cover_and_remove = self.weighted_cover_and_remove
        self.gamma = 0.7

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = self.find_rules(X, Y, np.copy(W) if W is not None else None,
                                    None, self.base_rules, self.domain)
        # add the default rule, other
        # TRUE rules are insufficient
        rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
        return CN2SDClassifier(domain=self.domain, rule_list=rule_list)


class CN2SDClassifier(_RuleClassifier):
    def predict(self, X):
        """
        In contrast to the classic CN2 algorithm, all applicable rules
        are taken into account even though CN2SD induces ordered rules.

        For each instance, all rules are tried and those that fire are
        collected. If a clash occurs, class probabilities (predictions)
        of all collected rules are summed and weighted.

        Notes
        -----
        If no other rules fire, the default rule prediction is used.
        Any other empty rules (a fallout most common in the domain of
        weighted covering algorithms) are ignored.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, float
            Probabilistic classification.
        """
        return self.unordered_predict(X)


class CN2SDUnorderedLearner(_RuleLearner):
    """
    Unordered CN2SD inducer that constructs a set of unordered rules. To
    evaluate found hypotheses, Weighted relative accuracy measure is
    used. Returns a CN2SDUnorderedClassifier if called with data.

    Notes
    -----
    A weighted covering algorithm is applied, in which subsequently
    induced rules also represent interesting and sufficiently large
    subgroups of the population. Covered positive examples are not
    deleted from the learning set, rather their weight is reduced.

    The algorithm demonstrates how classification rule learning
    (predictive induction) can be adapted to subgroup discovery, a task
    at the intersection of predictive and descriptive induction.

    References
    ----------
    .. [1] "Subgroup Discovery with CN2-SD", Nada Lavrač et al., Journal
           of Machine Learning Research 5 (2004), 153-188, 2004
    """
    name = 'CN2-SD unordered inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = WeightedRelativeAccuracyEvaluator()
        self.cover_and_remove = self.weighted_cover_and_remove
        self.gamma = 0.7

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = []
        for curr_class in range(len(self.domain.class_var.values)):
            rule_list.extend(self.find_rules(
                X, Y, np.copy(W) if W is not None else None,
                curr_class, self.base_rules, self.domain))
        # add the default rule
        rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
        return CN2SDUnorderedClassifier(domain=self.domain, rule_list=rule_list)


class CN2SDUnorderedClassifier(_RuleClassifier):
    def predict(self, X):
        """
        Following an unordered set of rules, for each instance, all
        rules are tried and those that fire are collected. If a clash
        occurs, class probabilities (predictions) of all collected rules
        are summed and weighted.

        Notes
        -----
        If no other rules fire, the default rule prediction is used.
        Any other empty rules (a fallout most common in the domain of
        weighted covering algorithms) are ignored.

        Parameters
        ----------
        X : ndarray
            Classify this data.

        Returns
        -------
        res : ndarray, float
            Probabilistic classification.
        """
        return self.unordered_predict(X)


def main():
    data = Table('titanic')
    learner = CN2Learner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule, rule.quality)
    print()

    data = Table('iris.tab')
    learner = CN2UnorderedLearner()
    learner.rule_finder.general_validator.max_rule_length = 2
    learner.rule_finder.general_validator.min_covered_examples = 10
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist())
    print()

if __name__ == "__main__":
    main()
