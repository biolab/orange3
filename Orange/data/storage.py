class Storage:
    domain = None

    MISSING, DENSE, SPARSE, SPARSE_BOOL = range(4)

    def X_density(self):
        """
        Indicates whether the attributes are dense (DENSE) or sparse (SPARSE).
        If they are sparse and all values are 0 or 1, it is marked as
        SPARSE_BOOL. The Storage class provides a default DENSE.
        """
        return Storage.DENSE


    def Y_density(self):
        """
        Indicates whether the class attribute(s) are dense (DENSE) or sparse
        (SPARSE). If they are sparse and all values are 0 or 1, it is marked as
        SPARSE_BOOL. The Storage class provides a default DENSE.
        """
        return Storage.DENSE


    def metas_density(self):
        """
        Indicates whether the meta attributes are dense (DENSE) or sparse
        (SPARSE). If they are sparse and all values are 0 or 1, it is marked as
        SPARSE_BOOL. The Storage class provides a default DENSE.
        """
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

    def _compute_distributions(self, columns=None):
        raise NotImplementedError

    def _compute_contingency(self, col_vars=None, row_var=None):
        raise NotImplementedError