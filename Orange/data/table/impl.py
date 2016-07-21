from .base import *
import pandas as pd


class Table(TableBase, pd.DataFrame):
    KNOWN_PANDAS_KWARGS = {"data", "index", "columns", "dtype", "copy"}

    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return Table

    @property
    def _constructor_sliced(self):
        """
        An ugly workaround for the fact that pandas doesn't transfer _metadata to Series objects.
        Where this property should return a constructor callable, we instead return a
        proxy function, which sets the necessary properties from _metadata using a closure
        to ensure thread-safety.

        This enables TableSeries to use .X/.Y/.metas because it has a Domain.
        """
        attrs = {k: getattr(self, k, None) for k in Table._metadata}

        class _transferer:
            # this is a class and not a function because sometimes, pandas
            # wants _constructor_sliced.from_array
            def from_array(self, *args, **kwargs):
                return _transferer._attr_setter(TableSeries.from_array(*args, **kwargs))

            def __call__(self, *args, **kwargs):
                return _transferer._attr_setter(TableSeries(*args, **kwargs))

            @staticmethod
            def _attr_setter(target):
                for k, v in attrs.items():
                    setattr(target, k, v)
                return target

        return _transferer()

    @property
    def _constructor_expanddim(self):
        return TablePanel

    def density(self):
        """
        Compute the table density.
        Return the ratio of null values (pandas interpretation of null)
        """
        return 1 - self.isnull().sum().sum() / self.size

    def X_density(self):
        return TableBase.DENSE

    def Y_density(self):
        return TableBase.DENSE

    def metas_density(self):
        return TableBase.DENSE

    def is_sparse(self):
        return False


class TableSeries(SeriesBase, pd.Series):
    @property
    def _constructor(self):
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return Table


class TablePanel(PanelBase, pd.Panel):
    @property
    def _constructor(self):
        return TablePanel

    @property
    def _constructor_sliced(self):
        return Table


class SparseTable(TableBase, pd.SparseDataFrame):
    # this differs from Table.KNOWN_PANDAS_KWARGS by default_kind and default_fill_value
    KNOWN_PANDAS_KWARGS = {"data", "index", "columns", "dtype", "copy", "default_kind", "default_fill_value"}

    def density(self):
        """
        Compute the table density.
        Return the density as reported by pd.SparseDataFrame
        """
        return pd.SparseDataFrame.density(self)

    def X_density(self):
        return TableBase.SPARSE

    def Y_density(self):
        return TableBase.SPARSE

    def metas_density(self):
        return TableBase.SPARSE

    def is_sparse(self):
        return True


class SparseTableSeries(SeriesBase, pd.SparseSeries):
    @property
    def _constructor(self):
        return SparseTableSeries

    @property
    def _constructor_expanddim(self):
        return SparseTable


class SparseTablePanel(PanelBase, pd.SparsePanel):
    @property
    def _constructor(self):
        return SparseTablePanel

    @property
    def _constructor_sliced(self):
        return SparseTable
