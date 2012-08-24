from ..data.variable import *
from ..data import Domain
from scipy import sparse
import numpy as np
import bottleneck as bn

class FileReader:
    def prescan_file(self, f, delim, nvars, disc_cols, cont_cols):
        values = [set() for _ in range(nvars)]
        decimals = [-1]*nvars
        for lne in f:
            lne = lne.split(delim)
            for vs, col in zip(values, disc_cols):
                vs[col].add(lne[col])
            for col in cont_cols:
                val = lne[col]
                if not col in Variable.DefaultUnknownStr and "." in val:
                    decs = len(val) - val.find(".") - 1
                    if decs > decimals[col]:
                        decimals[col] = decs
        return values, decimals


class TabDelimReader:
    def read_header(self, filename):
        f = open(filename)
        names = f.readline().strip("\n\r").split("\t")
        types = f.readline().strip("\n\r").split("\t")
        flags = f.readline().strip("\n\r").split("\t")
        self.n_columns = len(names)
        if len(types) != self.n_columns:
            raise ValueError("File contains %i variable names and %i types" %
                             (len(names), len(types)))
        if len(flags) > self.n_columns:
            raise ValueError("There are more flags than variables")
        else:
            flags += [""] * self.n_columns

        attributes = []
        class_vars = []
        metas = []

        self.attribute_columns = []
        self.classvar_columns = []
        self.meta_columns = []
        self.weight_column = -1
        self.basket_column = -1

        for col, (name, tpe, flag) in enumerate(zip(names, types, flags)):
            tpe = tpe.strip()
            flag = flag.split()
            if "i" in flag or "ignore" in flag:
                continue
            if "b" in flag or "basket" in flag:
                self.basket_column = col
                continue
            is_class = "class" in flag
            is_meta = "m" in flag or "meta" in flag or \
                      "s" in tpe or "string" in tpe
            is_weight = "w" in tpe or "weight" in tpe or \
                        "w" in flag or "weight" in flag

            if is_weight:
                if is_class:
                    raise ValueError("Variable {} (column {}) is marked as "
                                     "class and weight".format(name, col))
                self.weight_column = col
                continue

            if tpe in ["c", "continuous"]:
                var = ContinuousVariable.make(name)
            elif tpe in ["w", "weight"]:
                var = None
            elif tpe in ["d", "discrete"]:
                var = DiscreteVariable.make(name)
            elif tpe in ["s", "string"]:
                var = StringVariable.make(name)
            else:
                var = DiscreteVariable.make(name, values=tpe.split())
            var.fix_order = isinstance(var, DiscreteVariable) and not var.values

            if is_class:
                if is_meta:
                    raise ValueError("Variable {} (column {}) is marked as "
                                     "class and meta attribute".format(name, col))
                class_vars.append(var)
                self.classvar_columns.append((col, var.val_from_str_add))
            elif is_meta:
                metas.append(var)
                self.meta_columns.append((col, var.val_from_str_add))
            else:
                attributes.append(var)
                self.attribute_columns.append((col, var.val_from_str_add))

        domain = Domain(attributes, class_vars)
        domain.metas = metas
        return domain


    def count_lines(self, filename):
        i = -3
        for _ in open(filename):
            i += 1
        return i


    def read_data(self, filename, table):
        _X, _Y = table._X, table._Y
        _W = getattr(table, "_W", None)
        f = open(filename)
        f.readline(); f.readline(); f.readline()
        padding = [""] * self.n_columns
        if self.basket_column >= 0:
            table._Xsparse = _Xsparse = sparse.lil_matrix(len(_X), 100) # TODO how many columns?!
        if self.meta_columns:
            table._metas = _metas = np.empty(
                (len(_X), len(self.meta_columns)), dtype=object)
        line_count = 0
        _Xr = None # To be able to delete it below even when there are no attributes
        for lne in f:
            values = lne.strip().split()
            if not values:
                continue
            if len(values) > self.n_columns:
                raise ValueError("Too many columns in line {}", 4+line_count)
            elif len(values) < self.n_columns:
                values += padding
            if self.attribute_columns:
                _Xr = _X[line_count]
                for i, (col, reader) in enumerate(self.attribute_columns):
                    _Xr[i] = reader(values[col])
            for i, (col, reader) in enumerate(self.classvar_columns):
                _Y[line_count, i] = reader(values[col])
            if _W is not None:
                _W[line_count] = float(values[self.weight_column])
            for i, (col, reader) in enumerate(self.meta_columns):
                _metas[line_count, i] = reader(values[col])
            line_count += 1
        if line_count != len(_X):
            del _Xr, _X, _Y, _W, _metas
            table._X.resize(line_count, len(table.domain.attributes))
            table._Y.resize(line_count, len(table.domain.class_vars))
            if table._W is not None:
                table._W.resize(line_count)
            if table._metas is not None:
                table._metas.resize(line_count)
        table.n_rows = line_count


    def reorder_values_array(self, arr, variables):
        for col, var in enumerate(variables):
            if var.fix_order and len(var.values) < 1000:
                new_order = var.order_values(var.values)
                if new_order == var.values:
                    continue
                arr[:, col] += 1000
                for i, val in enumerate(var.values):
                    bn.replace(arr[:,col], 1000+i, new_order.index(val))
                var.values = new_order
            delattr(var, "fix_order")


    def reorder_values(self, table):
        self.reorder_values_array(table._X, table.domain.attributes)
        self.reorder_values_array(table._Y, table.domain.class_vars)

    def read_file(self, filename):
        from ..data import Table
        domain = self.read_header(filename)
        nExamples = self.count_lines(filename)
        table = Table.new_from_domain(domain, nExamples, self.weight_column >= 0)
        self.read_data(filename, table)
        self.reorder_values(table)
        return table

