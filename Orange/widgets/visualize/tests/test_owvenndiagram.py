import unittest
import numpy as np

from Orange.data import (Table, Domain, StringVariable,
                         DiscreteVariable, ContinuousVariable)
from Orange.widgets.visualize.owvenndiagram import (reshape_wide,
                                                    table_concat,
                                                    varying_between,
                                                    drop_columns)


class TestOWVennDiagram(unittest.TestCase):
    def add_metas(self, table, meta_attrs, meta_data):
        domain = Domain(table.domain.attributes,
                        table.domain.class_vars,
                        table.domain.metas + meta_attrs)
        metas = np.hstack((table.metas, meta_data))
        return Table(domain, table.X, table.Y, metas)

    def test_reshape_wide(self):
        class_var = DiscreteVariable("c", values=["x"])
        item_id_var = StringVariable("item_id")
        source_var = StringVariable("source")
        c1, c, item_id, ca, cb = np.random.randint(10, size=5)
        data = Table(Domain([ContinuousVariable("c1")], [class_var],
                            [DiscreteVariable("c(a)", class_var.values),
                             DiscreteVariable("c(b)", class_var.values),
                             source_var, item_id_var]),
                     np.array([[c1], [c1]], dtype=object),
                     np.array([[c], [c]], dtype=object),
                     np.array([[ca, np.nan, "a", item_id],
                               [np.nan, cb, "b", item_id]], dtype=object))

        data = reshape_wide(data, [], [item_id_var], [source_var])
        self.assertFalse(any(np.isnan(data.metas.astype(np.float32)[0])))
        self.assertEqual(len(data), 1)
        np.testing.assert_equal(data.metas, np.array([[ca, cb, item_id]],
                                                     dtype=object))

    def test_venn_diagram(self):
        sources = ["SVM Learner", "Naive Bayes", "Random Forest"]
        item_id_var = StringVariable("item_id")
        source_var = StringVariable("source")
        table = Table("zoo")
        class_var = table.domain.class_var
        cv = np.random.randint(len(class_var.values), size=(3, len(sources)))

        tables = []
        for i in range(len(sources)):
            temp_table = Table.from_table(table.domain, table,
                                          [0 + i, 1 + i, 2 + i])
            temp_d = (DiscreteVariable("%s(%s)" % (class_var.name,
                                                   sources[0 + i]),
                                       class_var.values),
                      source_var, item_id_var)
            temp_m = np.array([[cv[0, i], sources[i], table.metas[0 + i, 0]],
                               [cv[1, i], sources[i], table.metas[1 + i, 0]],
                               [cv[2, i], sources[i], table.metas[2 + i, 0]]
                               ], dtype=object)
            temp_table = self.add_metas(temp_table, temp_d, temp_m)
            tables.append(temp_table)

        data = table_concat(tables)
        varying = varying_between(data, [item_id_var])
        if source_var in varying:
            varying.remove(source_var)
        data = reshape_wide(data, varying, [item_id_var], [source_var])
        data = drop_columns(data, [item_id_var])

        result = np.array([[table.metas[0, 0], cv[0, 0], np.nan, np.nan],
                           [table.metas[1, 0], cv[1, 0], cv[0, 1], np.nan],
                           [table.metas[2, 0], cv[2, 0], cv[1, 1], cv[0, 2]],
                           [table.metas[3, 0], np.nan, cv[2, 1], cv[1, 2]],
                           [table.metas[4, 0], np.nan, np.nan, cv[2, 2]]],
                          dtype=object)

        for i in range(len(result)):
            for j in range(len(result[0])):
                val = result[i][j]
                if isinstance(val, float) and np.isnan(val):
                    self.assertTrue(np.isnan(data.metas[i][j]))
                else:
                    np.testing.assert_equal(data.metas[i][j], result[i][j])
