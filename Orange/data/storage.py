class Storage:
    domain = None

    def _filter_is_defined(self, columns=None, negate=False):
        raise NotImplementedError

    def _filter_has_class(self, negate=False):
        raise NotImplementedError

    def _filter_random(self, prob, negate=False):
        raise NotImplementedError

    def _filter_same_value(self, column, value, negate=False):
        raise NotImplementedError

    def _filter_values(self, filter):
        raise NotImplementedError

    def _compute_distributions(self, columns=None):
        raise NotImplementedError

    def _compute_contingency(self, col_vars=None, row_var=None):
        raise NotImplementedError