import unittest
import datetime
from collections import namedtuple

import numpy as np

from Orange.data import Table, Domain, StringVariable, ContinuousVariable, \
    DiscreteVariable, TimeVariable
from Orange.widgets.utils.state_summary import format_summary_details

VarDataPair = namedtuple('VarDataPair', ['variable', 'data'])

# Continuous variable variations
continuous_full = VarDataPair(
    ContinuousVariable('continuous_full'),
    np.array([0, 1, 2, 3, 4], dtype=float),
)
continuous_missing = VarDataPair(
    ContinuousVariable('continuous_missing'),
    np.array([0, 1, 2, np.nan, 4], dtype=float),
)

# Unordered discrete variable variations
rgb_full = VarDataPair(
    DiscreteVariable('rgb_full', values=('r', 'g', 'b')),
    np.array([0, 1, 1, 1, 2], dtype=float),
)
rgb_missing = VarDataPair(
    DiscreteVariable('rgb_missing', values=('r', 'g', 'b')),
    np.array([0, 1, 1, np.nan, 2], dtype=float),
)

# Ordered discrete variable variations
ints_full = VarDataPair(
    DiscreteVariable('ints_full', values=('2', '3', '4'), ordered=True),
    np.array([0, 1, 1, 1, 2], dtype=float),
)
ints_missing = VarDataPair(
    DiscreteVariable('ints_missing', values=('2', '3', '4'), ordered=True),
    np.array([0, 1, 1, np.nan, 2], dtype=float),
)


def _to_timestamps(years):
    return [datetime.datetime(year, 1, 1).timestamp() if not np.isnan(year)
            else np.nan for year in years]


time_full = VarDataPair(
    TimeVariable('time_full'),
    np.array(_to_timestamps([2000, 2001, 2002, 2003, 2004]), dtype=float),
)
time_missing = VarDataPair(
    TimeVariable('time_missing'),
    np.array(_to_timestamps([2000, np.nan, 2001, 2003, 2004]), dtype=float),
)

# String variable variations
string_full = VarDataPair(
    StringVariable('string_full'),
    np.array(['a', 'b', 'c', 'd', 'e'], dtype=object),
)
string_missing = VarDataPair(
    StringVariable('string_missing'),
    np.array(['a', 'b', 'c', StringVariable.Unknown, 'e'], dtype=object),
)


def make_table(attributes, target=None, metas=None):
    """Build an instance of a table given various variables.

    Parameters
    ----------
    attributes : Iterable[Tuple[Variable, np.array]
    target : Optional[Iterable[Tuple[Variable, np.array]]
    metas : Optional[Iterable[Tuple[Variable, np.array]]

    Returns
    -------
    Table

    """
    attribute_vars, attribute_vals = list(zip(*attributes))
    attribute_vals = np.array(attribute_vals).T

    target_vars, target_vals = None, None
    if target is not None:
        target_vars, target_vals = list(zip(*target))
        target_vals = np.array(target_vals).T

    meta_vars, meta_vals = None, None
    if metas is not None:
        meta_vars, meta_vals = list(zip(*metas))
        meta_vals = np.array(meta_vals).T

    return Table.from_numpy(
        Domain(attribute_vars, class_vars=target_vars, metas=meta_vars),
        X=attribute_vals, Y=target_vals, metas=meta_vals,
    )


class TestUtils(unittest.TestCase):
    def test_details(self):
        """Check if details part of the summary is formatted correctly"""
        data = Table('zoo')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'{len(data)} instances, ' \
                  f'{n_features} variables\n' \
                  f'Features: {len(data.domain.attributes)} categorical\n' \
                  f'Target: categorical\n' \
                  f'Metas: string (not shown)'
        self.assertEqual(details, format_summary_details(data))

        data = Table('housing')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'{len(data)} instances, ' \
                  f'{n_features} variables\n' \
                  f'Features: {len(data.domain.attributes)} numeric\n' \
                  f'Target: numeric\n' \
                  f'Metas: —'
        self.assertEqual(details, format_summary_details(data))

        data = Table('heart_disease')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'{len(data)} instances, ' \
                  f'{n_features} variables\n' \
                  f'Features: {len(data.domain.attributes)} ' \
                  f'(7 categorical, 6 numeric)\n' \
                  f'Target: categorical\n' \
                  f'Metas: —'
        self.assertEqual(details, format_summary_details(data))

        data = make_table(
            [continuous_full, continuous_missing],
            target=[rgb_full, rgb_missing], metas=[ints_full, ints_missing]
        )
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'{len(data)} instances, ' \
                  f'{n_features} variables\n' \
                  f'Features: {len(data.domain.attributes)} numeric\n' \
                  f'Target: {len(data.domain.class_vars)} categorical\n' \
                  f'Metas: {len(data.domain.metas)} categorical'
        self.assertEqual(details, format_summary_details(data))

        data = make_table(
            [continuous_full, time_full, ints_full, rgb_missing],
            target=[rgb_full, continuous_missing],
            metas=[string_full, string_missing]
        )
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'{len(data)} instances, ' \
                  f'{n_features} variables\n' \
                  f'Features: {len(data.domain.attributes)} ' \
                  f'(2 categorical, 1 numeric, 1 time)\n' \
                  f'Target: {len(data.domain.class_vars)} ' \
                  f'(1 categorical, 1 numeric)\n' \
                  f'Metas: {len(data.domain.metas)} string (not shown)'
        self.assertEqual(details, format_summary_details(data))

        data = make_table([time_full, time_missing], target=[ints_missing],
                          metas=None)
        details = f'{len(data)} instances, ' \
                  f'{len(data.domain.variables)} variables\n' \
                  f'Features: {len(data.domain.attributes)} time\n' \
                  f'Target: categorical\n' \
                  f'Metas: —'
        self.assertEqual(details, format_summary_details(data))

        data = make_table([rgb_full, ints_full], target=None, metas=None)
        details = f'{len(data)} instances, ' \
                  f'{len(data.domain.variables)} variables\n' \
                  f'Features: {len(data.domain.variables)} categorical\n' \
                  f'Target: —\n' \
                  f'Metas: —'
        self.assertEqual(details, format_summary_details(data))

        data = make_table([rgb_full], target=None, metas=None)
        details = f'{len(data)} instances, ' \
                  f'{len(data.domain.variables)} variable\n' \
                  f'Features: categorical\n' \
                  f'Target: —\n' \
                  f'Metas: —'
        self.assertEqual(details, format_summary_details(data))

        data = None
        self.assertEqual('', format_summary_details(data))


if __name__ == "__main__":
    unittest.main()
