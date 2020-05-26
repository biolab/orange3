"""Pandas DataFrame↔Table conversion helpers"""
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype, is_object_dtype,
    is_datetime64_any_dtype, is_numeric_dtype,
)

from Orange.data import (
    Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
    ContinuousVariable,
)

__all__ = ['table_from_frame', 'table_to_frame']


def table_from_frame(df, *, force_nominal=False):
    """
    Convert pandas.DataFrame to Orange.data.Table

    Parameters
    ----------
    df : pandas.DataFrame
    force_nominal : boolean
        If True, interpret ALL string columns as nominal (DiscreteVariable).

    Returns
    -------
    Table
    """

    def _is_discrete(s):
        return (is_categorical_dtype(s) or
                is_object_dtype(s) and (force_nominal or
                                        s.nunique() < s.size**.666))

    def _is_datetime(s):
        if is_datetime64_any_dtype(s):
            return True
        try:
            if is_object_dtype(s):
                pd.to_datetime(s, infer_datetime_format=True)
                return True
        except Exception:  # pylint: disable=broad-except
            pass
        return False

    # If df index is not a simple RangeIndex (or similar), put it into data
    if not (df.index.is_integer() and (df.index.is_monotonic_increasing or
                                       df.index.is_monotonic_decreasing)):
        df = df.reset_index()

    attrs, metas = [], []
    X, M = [], []

    # Iter over columns
    for name, s in df.items():
        name = str(name)
        if _is_discrete(s):
            discrete = s.astype('category').cat
            attrs.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
            X.append(discrete.codes.replace(-1, np.nan).values)
        elif _is_datetime(s):
            tvar = TimeVariable(name)
            attrs.append(tvar)
            s = pd.to_datetime(s, infer_datetime_format=True)
            X.append(s.astype('str').replace('NaT', np.nan).map(tvar.parse).values)
        elif is_numeric_dtype(s):
            attrs.append(ContinuousVariable(name))
            X.append(s.values)
        else:
            metas.append(StringVariable(name))
            M.append(s.values.astype(object))

    return Table.from_numpy(Domain(attrs, None, metas),
                            np.column_stack(X) if X else np.empty((df.shape[0], 0)),
                            None,
                            np.column_stack(M) if M else None)


def table_to_frame(tab, include_metas=False):
    """
    Convert Orange.data.Table to pandas.DataFrame

    Parameters
    ----------
    tab : Table

    include_metas : bool, (default=False)
        Include table metas into dataframe.

    Returns
    -------
    pandas.DataFrame
    """
    def _column_to_series(col, vals):
        result = ()
        if col.is_discrete:
            codes = pd.Series(vals).fillna(-1).astype(int)
            result = (col.name, pd.Categorical.from_codes(
                codes=codes, categories=col.values, ordered=True
            ))
        elif col.is_time:
            result = (col.name, pd.to_datetime(vals, unit='s').to_series().reset_index()[0])
        elif col.is_continuous:
            dt = float
            # np.nan are not compatible with int column
            nan_values_in_column = [t for t in vals if np.isnan(t)]
            if col.number_of_decimals == 0 and len(nan_values_in_column) == 0:
                dt = int
            result = (col.name, pd.Series(vals).astype(dt))
        elif col.is_string:
            result = (col.name, pd.Series(vals))
        return result

    def _columns_to_series(cols, vals):
        return [_column_to_series(col, vals[:, i]) for i, col in enumerate(cols)]

    x, y, metas = [], [], []
    domain = tab.domain
    if domain.attributes:
        x = _columns_to_series(domain.attributes, tab.X)
    if domain.class_vars:
        y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
        y = _columns_to_series(domain.class_vars, y_values)
    if domain.metas:
        metas = _columns_to_series(domain.metas, tab.metas)
    all_series = dict(x + y + metas)
    all_vars = tab.domain.variables
    if include_metas:
        all_vars += tab.domain.metas
    original_column_order = [var.name for var in all_vars]
    unsorted_columns_df = pd.DataFrame(all_series)
    return unsorted_columns_df[original_column_order]
