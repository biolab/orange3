import operator
from copy import copy
from hashlib import sha1
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import chisqprob
import Orange
from Orange.classification import Learner, Model

__all__ = ["CN2Learner", "CN2UnorderedLearner"]


def argmaxrnd(a, random_seed=None):
    """
    Find the index of the maximum value for a given 1-D numpy array.

    In case of multiple indices corresponding to the maximum value, the
    result is chosen randomly among those. The random number generator
    can be seeded by forwarding a seed value: see function 'hash_dist'.

    Parameters
    ----------
    a : array_like
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

    random = (np.random if random_seed is None
              else np.random.RandomState(random_seed))

    def f(x): return random.choice((x == np.nanmax(x)).nonzero()[0])
    return f(a) if a.ndim == 1 else np.apply_along_axis(f, axis=1, arr=a)


def entropy(x):
    x = x[x != 0]
    x = x / np.sum(x, axis=0)
    x = -x * np.log2(x)
    return np.sum(x)


def likelihood_ratio_statistic(x, y):
    x[x == 0] = 1e-5
    y[y == 0] = 1e-5
    lrs = np.sum(x * np.log(x / y))
    lrs = 2 * (lrs - np.sum(x) * np.log(np.sum(x) / np.sum(y)))
    return lrs


def get_dist(Y, domain):
    """
    Determine the class distribution for a given array. Gather the
    number of classes from the data domain.

    Parameters
    ----------
    Y : array_like
        The source array (observed classification).
    domain : Orange.data.domain.Domain
        Data domain.

    Returns
    -------
    dist : ndarray, int
        Distribution of classes.
    """
    return np.bincount(Y.astype(dtype=np.int),
                       minlength=len(domain.class_var.values))


def hash_dist(a):
    """
    For a given class distribution, calculate a hash value that can be
    used to seed the RNG.

    Parameters
    ----------
    a : array_like
        The source array.

    Returns
    -------
    hash : int
        Hash function result.
    """
    return int(sha1(bytes(a)).hexdigest(), base=16) & 0xffffffff


def rule_length(rule):
    return len(rule.selectors)


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
        raise NotImplementedError("descendants of Evaluator must "
                                  "overload method 'evaluate_rule'")


class EntropyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        x = rule.curr_class_dist.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class],
                          np.sum(x) - x[rule.target_class]])
        return -entropy(x)


class LaplaceAccuracyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        if rule.target_class is not None:
            target = rule.curr_class_dist[rule.target_class]
            combined = np.sum(rule.curr_class_dist)
            k = 2
        else:
            target = np.argmax(rule.curr_class_dist)
            combined = np.sum(rule.curr_class_dist)
            k = len(rule.curr_class_dist)
        return (target + 1) / (combined + k)


class LengthEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        return -rule_length(rule)


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
        raise NotImplementedError("descendants of Validator must "
                                  "overload method 'validate_rule'")


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
        rule_target_covered = (rule.curr_class_dist[rule.target_class]
                               if rule.target_class is not None
                               else np.sum(rule.curr_class_dist))

        return (rule_target_covered >= self.minimum_covered_examples and
                rule_length(rule) <= self.max_rule_length and
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

        x = rule.curr_class_dist.astype(dtype=np.float)
        y = rule.parent_rule.curr_class_dist.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class],
                          np.sum(x) - x[rule.target_class]])
            y = np.array([y[rule.target_class],
                          np.sum(y) - y[rule.target_class]])

        lrs = likelihood_ratio_statistic(x, y)
        return (lrs > 0 and
                chisqprob(lrs, len(rule.curr_class_dist) - 1) <= self.alpha)


class SearchAlgorithm:
    """
    Implement a procedure to maneuver through the search space towards a
    better solution, guided by the search heuristics.
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
        raise NotImplementedError("descendants of SearchAlgorithm must "
                                  "overload method 'select_candidates'")

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
        raise NotImplementedError("descendants of SearchAlgorithm must "
                                  "overload method 'filter_rules'")


class BeamSearchAlgorithm(SearchAlgorithm):
    """
    Remember the best rule found thus far and monitor a fixed number of
    alternatives (the beam).
    """
    def __init__(self, beam_width=5):
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
        X, Y : array_like
            Learning data.
        target_class : int or NoneType
            Index of the class to model.
        base_rules : Rule list
            An optional list of initial rules to constrain the search.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        prior_class_dist : array_like
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
        raise NotImplementedError("descendants of SearchStrategy must "
                                  "overload method 'initialise_rule'")

    def refine_rule(self, X, Y, candidate_rule):
        """
        Refine rule.

        Parameters
        ----------
        X, Y : array_like
            Learning data.
        candidate_rule : Rule
            Refine this rule.

        Returns
        -------
        rules : Rule list
            Descendant rules of 'candidate_rule'.
        """
        raise NotImplementedError("descendants of SearchStrategy must "
                                  "overload method 'refine_rule'")


class TopDownSearchStrategy(SearchStrategy):
    """
    An empty starting rule that covers all instances is developed and
    added to the list of candidate rules. The hypothesis space of
    possible rules is searched by repeatedly specialising candidate
    rules.
    """
    def initialise_rule(self, X, Y, target_class, base_rules, domain,
                        prior_class_dist, quality_evaluator,
                        complexity_evaluator, significance_validator,
                        general_validator):
        rules = []
        default_rule = Rule(domain=domain, prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

        default_rule.filter_and_store(X, Y, target_class)
        rules.append(default_rule)

        for base_rule in base_rules:
            temp_rule = Rule(selectors=copy(base_rule.selectors),
                             parent_rule=default_rule, domain=domain,
                             prior_class_dist=prior_class_dist,
                             quality_evaluator=quality_evaluator,
                             complexity_evaluator=complexity_evaluator,
                             significance_validator=significance_validator,
                             general_validator=general_validator)

            temp_rule.filter_and_store(X, Y, target_class)
            if temp_rule.validity:
                rules.append(temp_rule)

        return rules

    def refine_rule(self, X, Y, candidate_rule):
        covered_X = X[candidate_rule.covered_examples]
        possible_selectors = []

        for i, attribute in enumerate(candidate_rule.domain.attributes):
            column = covered_X[:, i]
            if attribute.is_discrete:
                for val in [int(x) for x in set(column)]:
                    s1 = Selector(column=i, op="==", value=val)
                    s2 = Selector(column=i, op="!=", value=val)
                    possible_selectors.extend([s1, s2])
            elif attribute.is_continuous:  # TODO: component based
                column = np.unique(column)
                dividers = np.array_split(column, min(10, column.shape[0]))
                dividers = [np.median(smh) for smh in dividers]
                for val in dividers:
                    s1 = Selector(column=i, op="<=", value=val)
                    s2 = Selector(column=i, op=">=", value=val)
                    possible_selectors.extend([s1, s2])

        possible_selectors = [smh for smh in possible_selectors if
                              smh not in candidate_rule.selectors]

        new_rules = []
        (target_class, selectors, domain, prior_class_dist,
         quality_evaluator, complexity_evaluator, significance_validator,
         general_validator) = candidate_rule.seed()

        for curr_selector in possible_selectors:
            copied_selectors = copy(selectors)
            copied_selectors.append(curr_selector)

            new_rule = Rule(selectors=copied_selectors,
                            parent_rule=candidate_rule, domain=domain,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

            new_rule.filter_and_store(X, Y, target_class, td_optimisation=True)
            # to ensure that the covered_examples matrices are of the
            # same size throughout the rule_finder iteration

            if new_rule.validity:
                new_rules.append(new_rule)

        return new_rules


class Selector:
    """
    Define a single rule condition.
    """
    operators = {
        # discrete, nominal variables
        '==': operator.eq,
        '!=': operator.ne,

        # continuous variables
        '<=': operator.le,
        '>=': operator.ge
    }

    def __init__(self, column, op, value):
        self.column = column
        self.op = op
        self.value = value

    def filter_instance(self, x):
        return self.operators[self.op](x[self.column], self.value)

    def filter_data(self, X):
        return self.operators[self.op](X[:, self.column], self.value)

    def __eq__(self, other):
        return all((self.op == other.op, self.value == other.value,
                    self.column == other.column))


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
        prior_class_dist : array_like
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
        self.prediction = None
        self.quality = None
        self.complexity = None
        self.significance = None
        self.validity = None

    def filter_and_store(self, X, Y, target_class, td_optimisation=False):
        """
        Apply data and target class to a rule.

        Parameters
        ----------
        X, Y : array_like
            Learning data.
        target_class : int or NoneType
            Index of the class to model.
        td_optimisation : bool
            Built-in 'top down' search strategy optimisation.

        Notes
        -----
        Most rules will only ever make 1 call to this method before
        they are discarded. Only rules returned by the rule_finder will
        have ever accessed any of the other class methods.
        """
        self.target_class = target_class
        if td_optimisation and self.parent_rule is not None:
            self.covered_examples = np.copy(self.parent_rule.covered_examples)
            start = len(self.parent_rule.selectors)
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=np.bool)
            start = 0
        for selector in self.selectors[start:]:
            self.covered_examples &= selector.filter_data(X)

        self.curr_class_dist = get_dist(Y[self.covered_examples], self.domain)
        self.validity = self.general_validator.validate_rule(self)
        if self.validity:
            self.quality = self.quality_evaluator.evaluate_rule(self)
            self.complexity = self.complexity_evaluator.evaluate_rule(self)
            self.significance = self.significance_validator.validate_rule(self)

    def evaluate_instance(self, x):
        """
        Evaluate a single instance.

        Parameters
        ----------
        x : array_like
            Evaluate this instance.

        Returns
        -------
        res : bool
            True, if the rule covers 'x'.
        """
        # return True if the given instance matches the rule condition
        return all((selector.filter_instance(x) for selector in self.selectors))

    def evaluate_data(self, X):
        """
        Evaluate several instances concurrently.

        Parameters
        ----------
        X : array_like
            Evaluate this data.

        Returns
        -------
        res : ndarray, bool
            Array of evaluations, size of ''X.shape[0]''.
        """
        curr_covered = np.ones(X.shape[0], dtype=np.bool)
        for selector in self.selectors:
            curr_covered &= selector.filter_data(X)
        return curr_covered

    def create_model(self):
        """
        For unordered rules, the prediction is set to be the target
        class. In contrast, ordered rules predict the majority class of
        the covered examples.
        """
        self.prediction = (self.target_class if self.target_class is not None
                           else argmaxrnd(self.curr_class_dist))

    def seed(self):
        """
        Forward relevant information to develop new rules.
        """
        return (self.target_class, self.selectors, self.domain,
                self.prior_class_dist, self.quality_evaluator,
                self.complexity_evaluator, self.significance_validator,
                self.general_validator)

    def __str__(self):
        attributes = self.domain.attributes
        class_var = self.domain.class_var

        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                                 (str(attributes[s.column].values[s.value])
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
        X, Y : array_like
            Learning data.
        target_class : int or NoneType
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
        def rcmp(rule): return rule.quality, rule.complexity
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
                            new_rule.significance):
                        best_rule = new_rule

            rules = sorted(rules, key=rcmp, reverse=True)
            rules = self.search_algorithm.filter_rules(rules)

        best_rule.create_model()
        return best_rule


class RuleLearner(Learner):
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
    __metaclass__ = ABCMeta

    def __init__(self, preprocessors=None, base_rules=None):
        """
        Initialise a RuleLearner object.

        Constrain the algorithm with a list of base rules. Also create
        a RuleFinder object. Set search bias and over-fitting avoidance
        bias parameters by setting its components.

        Parameters
        ----------
        base_rules : Rule list
            An optional list of initial rules to constrain the search.
        """
        super().__init__(preprocessors=preprocessors)
        self.base_rules = base_rules if base_rules is not None else []
        self.rule_finder = RuleFinder()

    @abstractmethod
    def fit(self, X, Y, W=None):
        rule_list = self.find_rules(X, Y, None, self.base_rules, self.domain)
        return RuleClassifier(domain=self.domain, rule_list=rule_list)

    def find_rules(self, X, Y, target_class, base_rules, domain):
        rule_list = []
        while not self.data_stopping(X, Y, target_class):
            new_rule = self.rule_finder(X, Y, target_class, base_rules, domain)
            if self.rule_stopping(X, Y, new_rule):
                break
            X, Y = self.cover_and_remove(X, Y, new_rule)
            rule_list.append(new_rule)
        return rule_list

    @staticmethod
    def rule_stopping(X, Y, new_rule):
        return False

    @staticmethod
    def data_stopping(X, Y, target_class):
        # stop if no positive examples
        return (Y.size == 0 or (target_class is not None and
                                target_class not in Y))

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = np.logical_not(new_rule.covered_examples)
        return X[examples_to_keep], Y[examples_to_keep]


class RuleClassifier(Model):
    """
    A rule induction classifier. Instances are classified following
    either an unordered set of rules or a decision list.
    """
    __metaclass__ = ABCMeta

    def __init__(self, domain=None, rule_list=None):
        super().__init__(domain)
        self.domain = domain
        self.rule_list = rule_list if rule_list is not None else []

    @abstractmethod
    def predict(self, X):
        # decision list (ordered) example
        classifications = []
        status = np.ones(X.shape[0], dtype=np.bool)
        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            curr_covered[curr_covered] = rule.prediction
            classifications.append(curr_covered)
        return np.sum(np.row_stack(classifications), axis=0)


class CN2Learner(RuleLearner):
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
        self.rule_finder.search_algorithm.beam_width = 10

    def fit(self, X, Y, W=None):
        rule_list = self.find_rules(X, Y, None, self.base_rules, self.domain)
        return CN2Classifier(domain=self.domain, rule_list=rule_list)


class CN2Classifier(RuleClassifier):
    def predict(self, X):
        """
        Following a decision list, for each instance, rules are tried in
        order until one fires.

        Parameters
        ----------
        X : array_like
            Classify this data.

        Returns
        -------
        res : ndarray, int
            Array of classifications, size of ''X.shape[0]''.
        """
        classifications = []
        status = np.ones(X.shape[0], dtype=np.bool)
        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            curr_covered[curr_covered] = rule.prediction
            classifications.append(curr_covered)
        return np.sum(np.row_stack(classifications), axis=0)


class CN2UnorderedLearner(RuleLearner):
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
    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.search_algorithm.beam_width = 10
        self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

    def fit(self, X, Y, W=None):
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


class CN2UnorderedClassifier(RuleClassifier):
    def predict(self, X):
        """
        Following an unordered set of rules, for each instance, all
        rules are tried and those that fired are collected. If a clash
        occurs (i.e. more than one class is predicted), class
        distributions of all collected rules are summed and the most
        probable class is predicted.

        Parameters
        ----------
        X : array_like
            Classify this data.

        Returns
        -------
        res : ndarray, int
            Array of classifications, size of ''X.shape[0]''.
        """
        num_classes = len(self.domain.class_var.values)
        classifications = []
        status = np.ones(X.shape[0], dtype=np.bool)
        resolve_clash = np.zeros(X.shape[0], dtype=np.int)
        clashes = np.zeros((X.shape[0], num_classes), dtype=np.int)

        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            resolve_clash += curr_covered
            temp = np.zeros((X.shape[0], num_classes), dtype=np.int)
            temp[curr_covered] = rule.curr_class_dist
            clashes += temp
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            curr_covered[curr_covered] = rule.prediction
            classifications.append(curr_covered)

        resolve_clash = resolve_clash > 1
        no_clash = np.logical_not(resolve_clash)
        classifications = np.sum(np.row_stack(classifications), axis=0)
        result = np.zeros(X.shape[0], dtype=np.int)
        result[no_clash] = classifications[no_clash]
        result[resolve_clash] = argmaxrnd(clashes[resolve_clash])
        return result


def main():
    data = Orange.data.Table('titanic')
    row_defined = ~np.isnan(data.X).any(axis=1)
    data.Y = data.Y[row_defined]
    data.X = data.X[row_defined]

    learner = CN2Learner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule)

    print()

    learner = CN2UnorderedLearner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist())

if __name__ == "__main__":
    main()

# WEIGHTS tok casa dokler pokrijes primere = 5
