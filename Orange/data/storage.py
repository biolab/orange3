class Storage:

    domain = None

    name = ""

    MISSING, DENSE, SPARSE, SPARSE_BOOL = range(4)

    def approx_len(self):
        return len(self)

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
        """Compute basic stats for each of the columns.

        :param columns: columns to calculate stats for. None = all of them
        :return: tuple(min, max, mean, 0, #nans, #non-nans)
        """
        raise NotImplementedError

    def _compute_distributions(self, columns=None):
        """Compute distribution of values for the given columns.

        :param columns: columns to calculate distributions for
        :return: a list of distributions. Type of distribution depends on the
                 type of the column:
                   - for discrete, distribution is a 1d np.array containing the
                     occurrence counts for each of the values.
                   - for continuous, distribution is a 2d np.array with
                     distinct (ordered) values of the variable in the first row
                     and their counts in second.
        """
        raise NotImplementedError

    def _compute_contingency(self, col_vars=None, row_var=None):
        """
        Compute contingency matrices for one or more discrete or
        continuous variables against the specified discrete variable.

        The resulting list  contains a pair for each column variable.
        The first element contains the contingencies and the second
        elements gives the distribution of the row variables for instances
        in which the value of the column variable is missing.

        The format of contingencies returned depends on the variable type:

        - for discrete variables, it is a numpy array, where
          element (i, j) contains count of rows with i-th value of the
          row variable and j-th value of the column variable.

        - for continuous variables, contingency is a list of two arrays,
          where the first array contains ordered distinct values of the
          column_variable and the element (i,j) of the second array
          contains count of rows with i-th value of the row variable
          and j-th value of the ordered column variable.

        :param col_vars: variables whose values will correspond to columns of
            contingency matrices
        :type col_vars: list of ints, variable names or descriptors of type
            :obj:`Orange.data.Variable`
        :param row_var: a discrete variable whose values will correspond to the
            rows of contingency matrices
        :type row_var: int, variable name or :obj:`Orange.data.DiscreteVariable`
        """
        raise NotImplementedError
