"""Pandas DataFrame↔Table conversion helpers"""
from unittest.mock import patch

import numpy as np
from pandas.core.dtypes.common import is_string_dtype
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from pandas.core.arrays import SparseArray
from pandas.core.arrays.sparse.dtype import SparseDtype
from pandas.api.types import (
    is_categorical_dtype, is_object_dtype,
    is_datetime64_any_dtype, is_numeric_dtype, is_integer_dtype
)

from Orange.data import (
    Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
    ContinuousVariable,
)
from Orange.data.table import Role

__all__ = ['table_from_frame', 'table_to_frame']


class OrangeDataFrame(pd.DataFrame):
    _metadata = ["orange_variables", "orange_weights",
                 "orange_attributes", "orange_role"]

    def __init__(self, *args, **kwargs):
        """
        A pandas DataFrame wrapper for one of Table's numpy arrays:
            - sets index values corresponding to Orange's global row indices
              e.g. ['_o1', '_o2'] (allows Orange to handle selection)
            - remembers the array's role in the Table (attribute, class var, meta)
            - keeps the Variable objects, and uses them in back-to-table conversion,
              should a column name match a variable's name
            - stores weight values (legacy)

        Parameters
        ----------
        table : Table
        orange_role : Role, (default=Role.Attribute)
            When converting back to an orange table, the DataFrame will
            convert to the right role (attrs, class vars, or metas)
        """
        if len(args) <= 0 or not isinstance(args[0], Table):
            super().__init__(*args, **kwargs)
            return
        table = args[0]
        if 'orange_role' in kwargs:
            role = kwargs.pop('orange_role')
        elif len(args) >= 2:
            role = args[1]
        else:
            role = Role.Attribute

        if role == Role.Attribute:
            data = table.X
            vars_ = table.domain.attributes
        elif role == Role.ClassAttribute:
            data = table.Y
            vars_ = table.domain.class_vars
        else:  # if role == Role.Meta:
            data = table.metas
            vars_ = table.domain.metas

        index = ['_o' + str(id_) for id_ in table.ids]
        varsdict = {var._name: var for var in vars_}
        columns = varsdict.keys()

        if sp.issparse(data):
            data = data.asformat('csc')
            sparrays = [SparseArray.from_spmatrix(data[:, i]) for i in range(data.shape[1])]
            data = dict(enumerate(sparrays))
            super().__init__(data, index=index, **kwargs)
            self.columns = columns
            # a hack to keep Orange df _metadata in sparse->dense conversion
            self.sparse.to_dense = self.__patch_constructor(self.sparse.to_dense)
        else:
            super().__init__(data=data, index=index, columns=columns, **kwargs)

        self.orange_role = role
        self.orange_variables = varsdict
        self.orange_weights = (dict(zip(index, table.W))
                               if table.W.size > 0 else {})
        self.orange_attributes = table.attributes

    def __patch_constructor(self, method):
        def new_method(*args, **kwargs):
            with patch(
                    'pandas.DataFrame',
                    OrangeDataFrame
            ):
                df = method(*args, **kwargs)
            df.__finalize__(self)
            return df

        return new_method

    @property
    def _constructor(self):
        return OrangeDataFrame

    def to_orange_table(self):
        return table_from_frame(self)

    def __finalize__(self, other, method=None, **_):
        """
        propagate metadata from other to self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : optional, a passed method name ; possibly to take different
            types of propagation actions based on this

        """
        if method == 'concat':
            objs = other.objs
        elif method == 'merge':
            objs = other.left, other.right
        else:
            objs = [other]

        orange_role = getattr(self, 'orange_role', None)
        dicts = {dname: getattr(self, dname, {})
                 for dname in ('orange_variables',
                               'orange_weights',
                               'orange_attributes')}
        for obj in objs:
            other_role = getattr(obj, 'orange_role', None)
            if other_role is not None:
                orange_role = other_role

            for dname, dict_ in dicts.items():
                other_dict = getattr(obj, dname, {})
                dict_.update(other_dict)

        object.__setattr__(self, 'orange_role', orange_role)
        for dname, dict_ in dicts.items():
            object.__setattr__(self, dname, dict_)

        return self

    pd.DataFrame.__finalize__ = __finalize__


def _reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """If df index is not a simple RangeIndex (or similar), include it into a table"""
    if (
        # not range-like index - test first to skip slow startswith(_o) check
        not (
            df.index.is_integer()
            and (df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing)
        )
        # check that it does not contain Orange index
        and (
            # startswith is slow (for long dfs) - firs check if col has strings
            isinstance(df.index, pd.MultiIndex)
            or not is_string_dtype(df.index)
            or not any(str(i).startswith("_o") for i in df.index)
        )
    ):
        df = df.reset_index()
    return df


def _is_discrete(s, force_nominal):
    return (is_categorical_dtype(s) or
            is_object_dtype(s) and (force_nominal or
                                    s.nunique() < s.size ** .666))


def _is_datetime(s):
    if is_datetime64_any_dtype(s):
        return True
    try:
        if is_object_dtype(s):
            # pd.to_datetime would sucessfuly parse column of numbers to datetime
            # but for column of object dtype with numbers we want to be either
            # discret or string - following code try to parse column to numeric
            # if connversion to numeric is sucessful return False
            try:
                pd.to_numeric(s)
                return False
            except (ValueError, TypeError):
                pass

            # utc=True - to allow different timezones in a series object
            pd.to_datetime(s, infer_datetime_format=True, utc=True)
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    return False


def _convert_datetime(series, var):
    def col_type(dt):
        """Test if is date, time or datetime"""
        dt_nonnat = dt[~pd.isnull(dt)]  # nat == nat is False
        if (dt_nonnat.dt.floor("d") == dt_nonnat).all():
            # all times are 00:00:00.0 - pure date
            return 1, 0
        elif (dt_nonnat.dt.date == pd.Timestamp("now").date()).all():
            # all dates are today's date - pure time
            return 0, 1  # pure time
        else:
            # else datetime
            return 1, 1

    try:
        dt = pd.to_datetime(series)
    except ValueError:
        # series with type object and different timezones will raise a
        # ValueError - normalizing to utc
        dt = pd.to_datetime(series, utc=True)

    # set variable type to date, time or datetime
    var.have_date, var.have_time = col_type(dt)

    if dt.dt.tz is not None:
        # set timezone if available and convert to utc
        var.timezone = dt.dt.tz
        dt = dt.dt.tz_convert("UTC")

    if var.have_time and not var.have_date:
        # if time only measure seconds from midnight - equal to setting date
        # to unix epoch
        return (
            (dt.dt.tz_localize(None) - pd.Timestamp("now").normalize())
            / pd.Timedelta("1s")
        ).values

    return (
        (dt.dt.tz_localize(None) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    ).values


def to_categorical(s, _):
    x = s.astype("category").cat.codes
    # it is same than x.replace(-1, np.nan), but much faster
    x = x.where(x != -1, np.nan)
    return np.asarray(x)


def vars_from_df(df, role=None, force_nominal=False):
    if role is None and hasattr(df, 'orange_role'):
        role = df.orange_role
    df = _reset_index(df)

    cols = [], [], []
    exprs = [], [], []
    vars_ = [], [], []

    for column in df.columns:
        s = df[column]
        _role = Role.Attribute if role is None else role
        if hasattr(df, 'orange_variables') and column in df.orange_variables:
            original_var = df.orange_variables[column]
            var = original_var.copy(compute_value=None)
            expr = None
        elif _is_datetime(s):
            var = TimeVariable(str(column))
            expr = _convert_datetime
        elif _is_discrete(s, force_nominal):
            discrete = s.astype("category").cat
            var = DiscreteVariable(
                str(column), discrete.categories.astype(str).tolist()
            )
            expr = to_categorical
        elif is_numeric_dtype(s):
            var = ContinuousVariable(
                # set number of decimals to 0 if int else keeps default behaviour
                str(column), number_of_decimals=(0 if is_integer_dtype(s) else None)
            )
            expr = None
        else:
            if role is not None and role != Role.Meta:
                raise ValueError("String variable must be in metas.")
            _role = Role.Meta
            var = StringVariable(str(column))
            expr = lambda s, _: np.asarray(s, dtype=object)

        cols[_role].append(column)
        exprs[_role].append(expr)
        vars_[_role].append(var)

    xym = []
    for a_vars, a_cols, a_expr in zip(vars_, cols, exprs):
        if not a_cols:
            arr = None if a_cols != cols[0] else np.empty((df.shape[0], 0))
        elif not any(a_expr):
            # if all c in columns table will share memory with dataframe
            a_df = df if all(c in a_cols for c in df.columns) else df[a_cols]
            if all(isinstance(a, SparseDtype) for a in a_df.dtypes):
                arr = csr_matrix(a_df.sparse.to_coo())
            else:
                arr = np.asarray(a_df)
        else:
            # we'll have to copy the table to resolve any expressions
            arr = np.array(
                [
                    expr(df[col], var) if expr else np.asarray(df[col])
                    for var, col, expr in zip(a_vars, a_cols, a_expr)
                ]
            ).T
        xym.append(arr)

    # Let the tables share memory with pandas frame
    if xym[1] is not None and xym[1].ndim == 2 and xym[1].shape[1] == 1:
        xym[1] = xym[1][:, 0]

    return xym, Domain(*vars_)


def table_from_frame(df, *, force_nominal=False):
    XYM, domain = vars_from_df(df, force_nominal=force_nominal)

    if hasattr(df, 'orange_weights') and hasattr(df, 'orange_attributes'):
        W = [df.orange_weights[i] for i in df.index if i in df.orange_weights]
        if len(W) != len(df.index):
            W = None
        attributes = df.orange_attributes
        if isinstance(df.index, pd.MultiIndex) or not is_string_dtype(df.index):
            # we can skip checking for Orange indices when MultiIndex an when
            # not string dtype and so speedup the conversion
            ids = None
        else:
            ids = [
                int(i[2:])
                if str(i).startswith("_o") and i[2:].isdigit()
                else Table.new_id()
                for i in df.index
            ]
    else:
        W = None
        attributes = None
        ids = None

    return Table.from_numpy(
        domain,
        *XYM,
        W=W,
        attributes=attributes,
        ids=ids
    )


def table_from_frames(xdf, ydf, mdf):
    if not (xdf.index.equals(ydf.index) and xdf.index.equals(mdf.index)):
        raise ValueError(
            "Indexes not equal. Make sure that all three dataframes have equal index"
        )

    # drop index from x and y - it makes sure that index if not range will be
    # placed in metas
    xdf = xdf.reset_index(drop=True)
    ydf = ydf.reset_index(drop=True)
    dfs = xdf, ydf, mdf

    if not all(df.shape[0] == xdf.shape[0] for df in dfs):
        raise ValueError(f"Leading dimension mismatch "
                         f"(not {xdf.shape[0]} == {ydf.shape[0]} == {mdf.shape[0]})")

    xXYM, xDomain = vars_from_df(xdf, role=Role.Attribute)
    yXYM, yDomain = vars_from_df(ydf, role=Role.ClassAttribute)
    mXYM, mDomain = vars_from_df(mdf, role=Role.Meta)

    XYM = (xXYM[0], yXYM[1], mXYM[2])
    domain = Domain(xDomain.attributes, yDomain.class_vars, mDomain.metas)

    ids = [
        int(idx[2:])
        if str(idx).startswith("_o") and idx[2:].isdigit()
        else Table.new_id()
        for idx in mdf.index
    ]

    attributes = {}
    W = None
    for df in dfs:
        if isinstance(df, OrangeDataFrame):
            W = [df.orange_weights[i] for i in df.index if i in df.orange_weights]
            if len(W) != len(df.index):
                W = None
            attributes.update(df.orange_attributes)
        else:
            W = None

    return Table.from_numpy(
        domain,
        *XYM,
        W=W,
        attributes=attributes,
        ids=ids
    )


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
            # using pd.isnull since np.isnan fails on array with dtype object
            # which can happen when metas contain column with strings
            if col.number_of_decimals == 0 and not np.any(pd.isnull(vals)):
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


def table_to_frames(table):
    xdf = OrangeDataFrame(table, Role.Attribute)
    ydf = OrangeDataFrame(table, Role.ClassAttribute)
    mdf = OrangeDataFrame(table, Role.Meta)

    return xdf, ydf, mdf


def amend_table_with_frame(table, df, role):
    arr = Role.get_arr(role, table)
    if arr.shape[0] != df.shape[0]:
        raise ValueError(f"Leading dimension mismatch "
                         f"(not {arr.shape[0]} == {df.shape[0]})")

    XYM, domain = vars_from_df(df, role=role)

    if role == Role.Attribute:
        table.domain = Domain(domain.attributes,
                              table.domain.class_vars,
                              table.domain.metas)
        table.X = XYM[0]
    elif role == Role.ClassAttribute:
        table.domain = Domain(table.domain.attributes,
                              domain.class_vars,
                              table.domain.metas)
        table.Y = XYM[1]
    else:  # if role == Role.Meta:
        table.domain = Domain(table.domain.attributes,
                              table.domain.class_vars,
                              domain.metas)
        table.metas = XYM[2]

    if isinstance(df, OrangeDataFrame):
        table.attributes.update(df.orange_attributes)
