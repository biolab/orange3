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
        raise NotImplemented()


class DiscreteRule(Rule):
    """Discrete rule class for handling Indicator rules.

    Parameters
    ----------
    attr_name : str
    eq : bool
        Should indicate whether or not the rule equals the value or not.
    value : object

    Examples
    --------
    Age = 30
    >>> rule = DiscreteRule('age', True, 30)
    Name ≠ John
    >>> rule = DiscreteRule('name', False, 'John')

    Notes
    -----
      - Merging discrete rules is currently not implemented, the new rule is
        simply returned and a warning is printed to stderr.

    """

    def __init__(self, attr_name, eq, value):
        self.attr_name = attr_name
        self.sign = eq
        self.value = value

    def merge_with(self, rule):
        # It does not make sense to merge discrete rules, since they can only
        # be eq or not eq.
        from sys import stderr
        print('WARNING: Merged two discrete rules `%s` and `%s`'
              % (self, rule), file=stderr)
        return rule

    def __str__(self):
        return '{} {} {}'.format(
            self.attr_name, '=' if self.sign else '≠', self.value)


class ContinuousRule(Rule):
    """Continuous rule class for handling numeric rules.

    Parameters
    ----------
    attr_name : str
    gt : bool
        Should indicate whether the variable must be greater than the value.
    value : int
    inclusive : bool, optional
        Should the variable range include the value or not
        (LT <> LTE | GT <> GTE). Default is False.

    Examples
    --------
    x ≤ 30
    >>> rule = ContinuousRule('age', False, 30, inclusive=True)
    x > 30
    >>> rule = ContinuousRule('age', True, 30)

    Notes
    -----
      - Continuous rules can currently only be merged with other continuous
        rules.

    """

    def __init__(self, attr_name, gt, value, inclusive=False):
        self.attr_name = attr_name
        self.sign = gt
        self.value = value
        self.inclusive = inclusive

    def merge_with(self, rule):
        if not isinstance(rule, ContinuousRule):
            raise NotImplemented('Continuous rules can currently only be '
                                 'merged with other continuous rules')
        # Handle when both have same sign
        if self.sign == rule.sign:
            # When both are GT
            if self.sign is True:
                larger = self.value if self.value > rule.value else rule.value
                return ContinuousRule(self.attr_name, self.sign, larger)
            # When both are LT
            else:
                smaller = self.value if self.value < rule.value else rule.value
                return ContinuousRule(self.attr_name, self.sign, smaller)
        # When they have different signs we need to return an interval rule
        else:
            lt_rule = self if self.sign is False else rule
            gt_rule = self if lt_rule != self else rule
            return IntervalRule(self.attr_name, gt_rule, lt_rule)

    def __str__(self):
        return '%s %s %.3f' % (
            self.attr_name, '>' if self.sign else '≤', self.value)


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
    1 ≤ x < 3
    >>> rule = IntervalRule('Rule',
    >>>                     ContinuousRule('Rule', True, 1, inclusive=True),
    >>>                     ContinuousRule('Rule', False, 3))

    Notes
    -----
      - Currently, only cases which appear in classification and regression
        trees are implemented. An interval can not be made up of two parts
        (e.g. (-∞, -1) ∪ (1, ∞)).

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
            if rule.sign:
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

    def __str__(self):
        return '{} ∈ {}{:.3}, {:.3}{}'.format(
            self.attr_name,
            '[' if self.left_rule.inclusive else '(', self.left_rule.value,
            self.right_rule.value, ']' if self.right_rule.inclusive else ')'
        )
