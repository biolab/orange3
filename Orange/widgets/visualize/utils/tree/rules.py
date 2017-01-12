r"""Rules for classification and regression trees.

Tree visualisations usually need to show the rules of nodes, these classes make
merging these rules simple (otherwise you have repeating rules e.g. `age < 3`
and `age < 2` which can be merged into `age < 2`.

Subclasses of the `Rule` class should provide a nice interface to merge rules
together through the `merge_with` method. Of course, this should not be forced
where it doesn't make sense e.g. merging a discrete rule (e.g.
:math:`x \in \{red, blue, green\}`) and a continuous rule (e.g.
:math:`x \leq 5`).

"""
import warnings


class Rule:
    """The base Rule class for tree rules."""

    def merge_with(self, rule):
        """Merge the current rule with the given rule.

        Parameters
        ----------
        rule : Rule

        Returns
        -------
        Rule

        """
        raise NotImplementedError()

    @property
    def description(self):
        return str(self)


class DiscreteRule(Rule):
    """Discrete rule class for handling Indicator rules.

    Parameters
    ----------
    attr_name : str
    equals : bool
        Should indicate whether or not the rule equals the value or not.
    value : object

    Examples
    --------
    >>> print(DiscreteRule('age', True, 30))
    age = 30

    >>> print(DiscreteRule('name', False, 'John'))
    name ≠ John

    Notes
    -----
    .. note:: Merging discrete rules is currently not implemented, the new rule
        is simply returned and a warning is issued.

    """

    def __init__(self, attr_name, equals, value):
        self.attr_name = attr_name
        self.equals = equals
        self.value = value

    def merge_with(self, rule):
        # It does not make sense to merge discrete rules, since they can only
        # be eq or not eq.
        warnings.warn('Merged two discrete rules `%s` and `%s`' % (self, rule))
        return rule

    @property
    def description(self):
        return '{} {}'.format('=' if self.equals else '≠', self.value)

    def __str__(self):
        return '{} {} {}'.format(
            self.attr_name, '=' if self.equals else '≠', self.value)

    def __repr__(self):
        return "DiscreteRule(attr_name='%s', equals=%s, value=%s)" % (
            self.attr_name, self.equals, self.value)


class ContinuousRule(Rule):
    """Continuous rule class for handling numeric rules.

    Parameters
    ----------
    attr_name : str
    greater : bool
        Should indicate whether the variable must be greater than the value.
    value : int
    inclusive : bool, optional
        Should the variable range include the value or not
        (LT <> LTE | GT <> GTE). Default is False.

    Examples
    --------
    >>> print(ContinuousRule('age', False, 30, inclusive=True))
    age ≤ 30.000

    >>> print(ContinuousRule('age', True, 30))
    age > 30.000

    Notes
    -----
    .. note:: Continuous rules can currently only be merged with other
        continuous rules.

    """

    def __init__(self, attr_name, greater, value, inclusive=False):
        self.attr_name = attr_name
        self.greater = greater
        self.value = value
        self.inclusive = inclusive

    def merge_with(self, rule):
        if not isinstance(rule, ContinuousRule):
            raise NotImplementedError('Continuous rules can currently only be '
                                      'merged with other continuous rules')
        # Handle when both have same sign
        if self.greater == rule.greater:
            # When both are GT
            if self.greater is True:
                larger = max(self.value, rule.value)
                return ContinuousRule(self.attr_name, self.greater, larger)
            # When both are LT
            else:
                smaller = min(self.value, rule.value)
                return ContinuousRule(self.attr_name, self.greater, smaller)
        # When they have different signs we need to return an interval rule
        else:
            lt_rule, gt_rule = (rule, self) if self.greater else (self, rule)
            return IntervalRule(self.attr_name, gt_rule, lt_rule)

    @property
    def description(self):
        return '%s %.3f' % ('>' if self.greater else '≤', self.value)

    def __str__(self):
        return '%s %s %.3f' % (
            self.attr_name, '>' if self.greater else '≤', self.value)

    def __repr__(self):
        return "ContinuousRule(attr_name='%s', greater=%s, value=%s, " \
               "inclusive=%s)" % (self.attr_name, self.greater, self.value,
                                  self.inclusive)


class IntervalRule(Rule):
    """Interval rule class for ranges of continuous values.

    Parameters
    ----------
    attr_name : str
    left_rule : ContinuousRule
        The smaller (left) part of the interval.
    right_rule : ContinuousRule
        The larger (right) part of the interval.

    Examples
    --------
    >>> print(IntervalRule('Rule',
    >>>                    ContinuousRule('Rule', True, 1, inclusive=True),
    >>>                    ContinuousRule('Rule', False, 3)))
    Rule ∈ [1.000, 3.000)

    Notes
    -----
    .. note:: Currently, only cases which appear in classification and
        regression trees are implemented. An interval can not be made up of two
        parts (e.g. (-∞, -1) ∪ (1, ∞)).

    """

    def __init__(self, attr_name, left_rule, right_rule):
        if not isinstance(left_rule, ContinuousRule):
            raise AttributeError(
                'The left rule must be an instance of the `ContinuousRule` '
                'class.')
        if not isinstance(right_rule, ContinuousRule):
            raise AttributeError(
                'The right rule must be an instance of the `ContinuousRule` '
                'class.')

        self.attr_name = attr_name
        self.left_rule = left_rule
        self.right_rule = right_rule

    def merge_with(self, rule):
        if isinstance(rule, ContinuousRule):
            if rule.greater:
                return IntervalRule(
                    self.attr_name, self.left_rule.merge_with(rule),
                    self.right_rule)
            else:
                return IntervalRule(
                    self.attr_name, self.left_rule,
                    self.right_rule.merge_with(rule))

        elif isinstance(rule, IntervalRule):
            return IntervalRule(
                self.attr_name,
                self.left_rule.merge_with(rule.left_rule),
                self.right_rule.merge_with(rule.right_rule))

    @property
    def description(self):
        return '∈ %s%.3f, %.3f%s' % (
            '[' if self.left_rule.inclusive else '(',
            self.left_rule.value,
            self.right_rule.value,
            ']' if self.right_rule.inclusive else ')'
        )

    def __str__(self):
        return '%s ∈ %s%.3f, %.3f%s' % (
            self.attr_name,
            '[' if self.left_rule.inclusive else '(',
            self.left_rule.value,
            self.right_rule.value,
            ']' if self.right_rule.inclusive else ')'
        )

    def __repr__(self):
        return "IntervalRule(attr_name='%s', left_rule=%s, right_rule=%s)" % (
            self.attr_name, repr(self.left_rule), repr(self.right_rule))
