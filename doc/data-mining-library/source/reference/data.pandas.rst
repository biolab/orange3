.. currentmodule:: Orange.data.pandas_compat

###########################################
Pandas interoperability (``pandas_compat``)
###########################################

:obj:`Orange.data.pandas_compat` module provides functions to convert between :class:`pandas.DataFrame` and :class:`Orange.data.Table`. These functions enable integration of Orange's data structures with the pandas library, enabling users to shift between the frameworks.

.. method::`table_from_frame`
.. autofunction:: table_from_frame

.. method::`table_to_frame`
.. autofunction:: table_to_frame

Example
=======

>>> import pandas as pd
>>> from Orange.data import Table
>>> from Orange.data.pandas_compat import table_from_frame, table_to_frame
>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0], 'C': ['a', 'b', 'c']})
>>> df
   A    B  C
0  1  4.0  a
1  2  5.0  b
2  3  6.0  c
>>> table = table_from_frame(df)
>>> table
[[1, 4] {a},
 [2, 5] {b},
 [3, 6] {c}
]

Note that the non-numeric column 'C' becomes a meta attribute in the resulting table. To set it to a categorical variable, use ``force_nominal=True``:

>>> table = table_from_frame(df, force_nominal=True)
[[1, 4, a],
 [2, 5, b],
 [3, 6, c]
]

To convert back to a pandas DataFrame, use :func:`table_to_frame`:

>>> frame = table_to_frame(table)
>>> frame
   A    B
0  1  4.0
1  2  5.0
2  3  6.0
