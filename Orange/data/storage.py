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
        """Compute contingency tables with row_var values in rows and column_var
         values in columns.

        :param col_vars: variables to compute contingency tables for
        :param row_var: discrete variable in rows of the contingency tables
        :return: a list of tuples (contingency_table, unknown), one for each
                 var in col_vars. Structure of the contingency_table depends
                 on the type of the column variable:
                    - for discrete, contingency_table is a 2d numpy array, where
                      element (i, j) contains count of rows with i-th value
                      of the row variable and j-th value of the column variable.
                    - for continuous, contingency is a list of two arrays,
                      where the first array contains ordered distinct values 
                      of the column_variable and the element (i,j) of the second 
                      arrays contains count of rows with i-th value                                          of the row variable and j-th value of the ordered column 
                      variable.
        """
        raise NotImplementedError
