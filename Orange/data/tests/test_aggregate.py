import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from Orange.data import (
    DiscreteVariable,
    ContinuousVariable,
    Domain,
    StringVariable,
    Table,
    table_to_frame,
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
        e.g. if input table corpus the resulting table must be corpus too.
        """
        data = AlternativeTable.from_table(self.data.domain, self.data)
        gb = data.groupby([data.domain["a"]])
        output = gb.aggregate({data.domain["a"]: ["mean"]})
        self.assertIsInstance(output, AlternativeTable)

    def test_preserve_variables(self):
        a, _, _, dvar = self.data.domain.attributes
        gb = self.data.groupby([a])

        a.attributes = {"foo": "bar"}
        dvar.attributes = {"foo": "baz"}

        a.copy = Mock(side_effect=a.copy)
        a.make = Mock(side_effect=a.make)

        def f(*_):
            return 0

        output = gb.aggregate(
            {a: [("copy", f, True),
                 ("make", f, False),
                 ("auto", f, None),
                 ("string", f, StringVariable),
                 ("number", f, ContinuousVariable)],
             dvar: [("copy", f, True),
                    ("make", f, False),
                    ("auto", f, None),
                    ("string", f, StringVariable),
                    ("discrete", f, DiscreteVariable)]}
        )
        self.assertIsInstance(output.domain["a - copy"], ContinuousVariable)
        a.copy.assert_called_once()
        self.assertEqual(output.domain["a - copy"].attributes, {"foo": "bar"})

        self.assertIsInstance(output.domain["a - make"], ContinuousVariable)
        a.make.assert_called_once()
        self.assertNotEqual(output.domain["a - make"].attributes, {"foo": "bar"})

        self.assertIsInstance(output.domain["a - auto"], ContinuousVariable)
        self.assertNotEqual(output.domain["a - auto"].attributes, {"foo": "bar"})

        self.assertIsInstance(output.domain["a - string"], StringVariable)

        self.assertIsInstance(output.domain["a - number"], ContinuousVariable)
        self.assertNotEqual(output.domain["a - number"].attributes, {"foo": "bar"})

        self.assertIsInstance(output.domain["dvar - copy"], DiscreteVariable)
        self.assertEqual(output.domain["dvar - copy"].attributes, {"foo": "baz"})

        self.assertIsInstance(output.domain["dvar - make"], DiscreteVariable)
        self.assertNotEqual(output.domain["dvar - make"].attributes, {"foo": "baz"})

        # f returns 0, so the column looks numeric! Let's test that it is
        # converted to numeric.
        self.assertIsInstance(output.domain["dvar - auto"], ContinuousVariable)

        self.assertIsInstance(output.domain["dvar - string"], StringVariable)

        self.assertIsInstance(output.domain["dvar - discrete"], DiscreteVariable)
        self.assertNotEqual(output.domain["dvar - discrete"].attributes, {"foo": "baz"})


if __name__ == "__main__":
    unittest.main()
