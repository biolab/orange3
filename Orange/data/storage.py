class Storage:
    domain = None

    @property
    def X_is_sparse(self):
        """
        Indicates whether the attributes are sparse. The default property in
        Storage class assumes that sparse data is not supported.
        """
        return False


    @property
    def Y_is_sparse(self):
        """
        Indicates whether the class attributes are sparse. The default property
        in Storage class assumes that sparse data is not supported.
        """
        return False


    @property
    def metas_is_sparse(self):
        """
        Indicates whether the meta attributes are sparse. The default property
        in Storage class assumes that sparse data is not supported.
        """
        return False

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