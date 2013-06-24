class Storage:

    domain = None

    MISSING, DENSE, SPARSE, SPARSE_BOOL = range(4)

    def X_density(self):
        return Storage.DENSE

    def Y_density(self):
        return Storage.DENSE

    def metas_density(self):
        return Storage.DENSE

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

    def _compute_basic_stats(self, columns=None):
        """
        :param columns: columns to calculate stats for. None = all of them
        :return: tuple(min, max, mean, 0, #nans, #non-nans)
        """
        raise NotImplementedError

    def _compute_distributions(self, columns=None):
        raise NotImplementedError

    def _compute_contingency(self, col_vars=None, row_var=None):
        raise NotImplementedError
