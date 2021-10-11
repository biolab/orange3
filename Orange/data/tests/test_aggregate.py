import unittest

import numpy as np
import pandas as pd

from Orange.data import (
    DiscreteVariable,
    ContinuousVariable,
    Domain,
    StringVariable,
    Table,
    table_to_frame,
    TimeVariable
)


def create_sample_data():
    domain = Domain(
        [
            ContinuousVariable("a"),
            ContinuousVariable("b"),
            ContinuousVariable("cvar"),
            DiscreteVariable("dvar", values=["val1", "val2"]),
        ],
        metas=[StringVariable("svar")],
    )
    return Table.from_numpy(
        domain,
        np.array(
            [
                [1, 1, 0.1, 0],
                [1, 1, 0.2, 1],
                [1, 2, np.nan, np.nan],
                [1, 2, 0.3, 1],
                [1, 3, 0.3, 0],
                [1, 3, 0.4, 1],
                [1, 3, 0.6, 0],
                [2, 1, 1.0, 1],
                [2, 1, 2.0, 0],
                [2, 2, 3.0, 1],
                [2, 2, -4.0, 0],
                [2, 3, 5.0, 1],
                [2, 3, 5.0, 0],
            ]
        ),
        metas=np.array(
            [
                ["sval1"],
                ["sval2"],
                [""],
                ["sval2"],
                ["sval1"],
                ["sval2"],
                ["sval1"],
                ["sval2"],
                ["sval1"],
                ["sval2"],
                ["sval1"],
                ["sval2"],
                ["sval1"],
            ]
        ),
    )


# pylint: disable=abstract-method
class AlternativeTable(Table):
    pass


class DomainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = create_sample_data()

    def test_simple_aggregation(self):
        """Test aggregation results"""
        d = self.data.domain
        gb = self.data.groupby([d["a"]])
        output = gb.aggregate({d["a"]: ["mean"], d["b"]: ["mean"]})

        np.testing.assert_array_almost_equal(output.X, [[1, 2.143], [2, 2]], decimal=3)
        np.testing.assert_array_almost_equal(output.metas, [[1], [2]], decimal=3)
        self.assertListEqual(
            ["a - mean", "b - mean"], [d.name for d in output.domain.attributes]
        )
        self.assertListEqual(["a"], [d.name for d in output.domain.metas])

    def test_aggregation(self):
        d = self.data.domain
        gb = self.data.groupby([self.data.domain["a"], self.data.domain["b"]])
        output = gb.aggregate(
            {
                d["cvar"]: [("Mean", "mean"), ("Median", "median"), ("Mean1", np.mean)],
                d["dvar"]: [("Count defined", "count"), ("Count", "size")],
                d["svar"]: [("Concatenate", "".join)],
            }
        )

        expected_columns = [
            "cvar - Mean",
            "cvar - Median",
            "cvar - Mean1",
            "dvar - Count defined",
            "dvar - Count",
            "svar - Concatenate",
            "a",  # groupby variables are last two in metas
            "b",
        ]

        exp_df = pd.DataFrame(
            [
                [0.15, 0.15, 0.15, 2, 2, "sval1sval2", 1, 1],
                [0.3, 0.3, 0.3, 1, 2, "sval2", 1, 2],
                [0.433, 0.4, 0.433, 3, 3, "sval1sval2sval1", 1, 3],
                [1.5, 1.5, 1.5, 2, 2, "sval2sval1", 2, 1],
                [-0.5, -0.5, -0.5, 2, 2, "sval2sval1", 2, 2],
                [5, 5, 5, 2, 2, "sval2sval1", 2, 3],
            ],
            columns=expected_columns,
        )

        out_df = table_to_frame(output, include_metas=True)

        pd.testing.assert_frame_equal(
            out_df,
            exp_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
            atol=1e-3,
        )

    def test_preserve_table_class(self):
        """
        Test whether result table has the same type than the imnput table,
        e.g. if input table corpus the resutlitn table must be corpus too.
        """
        data = AlternativeTable.from_table(self.data.domain, self.data)
        gb = data.groupby([data.domain["a"]])
        output = gb.aggregate({data.domain["a"]: ["mean"]})
        self.assertIsInstance(output, AlternativeTable)

    def test_range_selection(self):
        gb = self.data.groupby([self.data.domain["a"]], ('head', 2, None))
        output = gb.aggregate({self.data.domain["cvar"]: [("Mean", "mean")]})

        np.testing.assert_array_almost_equal(output.X, [[0.15], [1.5]], decimal=3)
        np.testing.assert_array_almost_equal(output.metas, [[1], [2]], decimal=3)

    def test_time_range_selection(self):
        time = TimeVariable("time")
        new_domain = Domain(self.data.domain.attributes + (time, ))
        table = self.data.transform(new_domain)
        time_values = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
        table[:, time] = time_values.reshape(-1, 1)

        gb = table.groupby([table.domain["a"]], ('last', 1.5, 'time'))
        output = gb.aggregate({table.domain["cvar"]: [("Mean", "mean")]})

        np.testing.assert_array_almost_equal(output.X, [[0.433], [2.0]], decimal=3)
        np.testing.assert_array_almost_equal(output.metas, [[1], [2]], decimal=3)


if __name__ == "__main__":
    unittest.main()
