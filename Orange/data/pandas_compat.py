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

__all__ = ['table_from_frame']


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
