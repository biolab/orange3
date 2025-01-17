from functools import lru_cache
from typing import Callable, Dict, List, Tuple, Union, Type

import pandas as pd

from Orange.data import Domain, Table, Variable, table_from_frame, table_to_frame
from Orange.util import dummy_callback


class OrangeTableGroupBy:
    """
    A class representing the result of the groupby operation on Orange's
    Table and offers aggregation functionality on groupby object. It wraps
    Panda's GroupBy object.

    Attributes
    ----------
    table
        Table to be grouped
    by
        Variable used for grouping. Resulting groups are defined with unique
        combinations of those values.

    Examples
    --------
    from Orange.data import Table

    table = Table("iris")
    gb = table.groupby([table.domain["iris"]])
    aggregated_table = gb.aggregate(
        {table.domain["sepal length"]: ["mean", "median"],
         table.domain["petal length"]: ["mean"]}
    )
    """

    def __init__(self, table: Table, by: List[Variable]):
        self.table = table

        df = table_to_frame(table, include_metas=True)
        # observed=True keeps only groups with at leas one instance
        self.group_by = df.groupby([a.name for a in by], observed=True)
        self.by = tuple(by)

        # lru_cache that is caches on the object level
        self.compute_aggregation = lru_cache()(self._compute_aggregation)

    AggDescType = Union[str,
                    Callable,
                    Tuple[str, Union[str, Callable]],
                    Tuple[str, Union[str, Callable], Union[Type[Variable], bool]]
    ]

    def aggregate(
        self,
        aggregations: Dict[Variable, List[AggDescType]],
        callback: Callable = dummy_callback,
    ) -> Table:
        """
        Compute aggregations for each group

        Parameters
        ----------
        aggregations
            The dictionary that defines aggregations that need to be computed
            for variables. We support three formats:
            - {variable name: [agg function 1, agg function 2]}
            - {variable name: [(agg name 1, agg function 1),  (agg name 1, agg function 1)]}
            - {variable name: [(agg name 1, agg function 1, output_variable_type1), ...]}
            Where agg name is the aggregation name used in the output column name.
            Aggregation function can be either function or string that defines
            aggregation in Pandas (e.g. mean).
            output_variable_type can be a type for a new variable, True to copy
            the input variable, or False to create a new variable of the same type
            as the input
        callback
            Callback function to report the progress

        Returns
        -------
        Table that includes aggregation columns. Variables that are used for
        grouping are in metas.
        """
        num_aggs = sum(len(aggs) for aggs in aggregations.values())
        count = 0

        result_agg = []
        output_variables = []
        for col, aggs in aggregations.items():
            for agg in aggs:
                res, var = self._compute_aggregation(col, agg)
                result_agg.append(res)
                output_variables.append(var)
                count += 1
                callback(count / num_aggs * 0.8)

        agg_table = self._aggregations_to_table(result_agg, output_variables)
        callback(1)
        return agg_table

    def _compute_aggregation(
            self, col: Variable, agg: AggDescType) -> Tuple[pd.Series, Variable]:
        # use named aggregation to avoid issues with same column names when reset_index
        if isinstance(agg, tuple):
            name, agg, var_type, *_ = (*agg, None)
        else:
            name = agg if isinstance(agg, str) else agg.__name__
            var_type = None
        col_name = f"{col.name} - {name}"
        agg_col = self.group_by[col.name].agg(**{col_name: agg})
        if col.is_discrete and var_type is True:
            dtype = pd.CategoricalDtype(categories=col.values, ordered=True)
            agg_col = agg_col.astype(dtype)
        if var_type is True:
            var = col.copy(name=col_name)
        elif var_type is False:
            var = col.make(name=col_name)
        elif var_type is None:
            var = None
        else:
            assert issubclass(var_type, Variable)
            var = var_type.make(name=col_name)
        return agg_col, var

    def _aggregations_to_table(
            self,
            aggregations: List[pd.Series],
            output_variables: List[Union[Variable, None]]) -> Table:
        """Concatenate aggregation series and convert back to Table"""
        if aggregations:
            df = pd.concat(aggregations, axis=1)
        else:
            # when no aggregation is computed return a table with gropby columns
            df = self.group_by.first()
            df = df.drop(columns=df.columns)
        gb_attributes = df.index.names
        df = df.reset_index()  # move group by var that are in index to columns
        table = table_from_frame(df, variables=(*self.by, *output_variables))

        # group by variables should be last two columns in metas in the output
        metas = table.domain.metas
        new_metas = [m for m in metas if m.name not in gb_attributes] + [
            table.domain[n] for n in gb_attributes
        ]
        new_domain = Domain(
            [var for var in table.domain.attributes if var.name not in gb_attributes],
            metas=new_metas,
        )
        # keeps input table's type - e.g. output is Corpus if input Corpus
        return self.table.from_table(new_domain, table)
