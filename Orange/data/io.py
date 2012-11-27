import re
import warnings
import collections
from ..data.variable import *
from ..data import Domain
from scipy import sparse
import numpy as np
import bottleneck as bn


class FileReader:
    def prescan_file(self, f, delim, nvars, disc_cols, cont_cols):
        values = [set() for _ in range(nvars)]
        decimals = [-1] * nvars
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
    def read_header(self, f):
        f.seek(0)
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
            is_meta = "m" in flag or "meta" in flag or\
                      "s" in tpe or "string" in tpe
            is_weight = "w" in tpe or "weight" in tpe or\
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
                var = DiscreteVariable.make(name, tpe.split(), True)
            var.fix_order = (isinstance(var, DiscreteVariable)
                             and not var.values)

            if is_class:
                if is_meta:
                    raise ValueError(
                        "Variable {} (column {}) is marked as "
                        "class and meta attribute".format(name, col))
                class_vars.append(var)
                self.classvar_columns.append((col, var.val_from_str_add))
            elif is_meta:
                metas.append(var)
                self.meta_columns.append((col, var.val_from_str_add))
            else:
                attributes.append(var)
                self.attribute_columns.append((col, var.val_from_str_add))

        domain = Domain(attributes, class_vars, metas)
        return domain

    def count_lines(self, file):
        file.seek(0)
        i = -3
        for _ in file:
            i += 1
        return i

    def read_data(self, f, table):
        _X, _Y = table._X, table._Y
        _W = table._W if table._W.shape[-1] else None
        f.seek(0)
        f.readline()
        f.readline()
        f.readline()
        padding = [""] * self.n_columns
        if self.basket_column >= 0:
            # TODO how many columns?!
            table._Xsparse = sparse.lil_matrix(len(_X), 100)
        table._metas = _metas = (
            np.empty((len(_X), len(self.meta_columns)), dtype=object))
        line_count = 0
        _Xr = None
        for lne in f:
            values = lne.strip().split()
            if not values:
                continue
            if len(values) > self.n_columns:
                raise ValueError("Too many columns in line {}", 4 + line_count)
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
            if table._W.ndim == 1:
                table._W.resize(line_count)
            else:
                table._W.resize((line_count, 0))
            table._metas.resize((line_count, len(self.meta_columns)))
        table.n_rows = line_count

    def reorder_values_array(self, arr, variables):
        for col, var in enumerate(variables):
            if var.fix_order and len(var.values) < 1000:
                new_order = var.ordered_values(var.values)
                if new_order == var.values:
                    continue
                arr[:, col] += 1000
                for i, val in enumerate(var.values):
                    bn.replace(arr[:, col], 1000 + i, new_order.index(val))
                var.values = new_order
            delattr(var, "fix_order")

    def reorder_values(self, table):
        self.reorder_values_array(table._X, table.domain.attributes)
        self.reorder_values_array(table._Y, table.domain.class_vars)

    def read_file(self, filename):
        with open(filename) as file:
            return self._read_file(file)

    def _read_file(self, file):
        from ..data import Table

        domain = self.read_header(file)
        nExamples = self.count_lines(file)
        table = Table.new_from_domain(domain, nExamples,
                                      self.weight_column >= 0)
        self.read_data(file, table)
        self.reorder_values(table)
        return table


class BasketReader():
    re_name = re.compile("([^,=\\n]+)(=((\d+\.?)|(\d*\.\d+)))?")

    def prescan_file(self, file):
        """Return a list of attributes that appear in the file"""
        names = set()
        n_elements = 0
        n_rows = 0
        for line in file:
            items = set(mo.group(1).strip()
                        for mo in self.re_name.finditer(line))
            names.update(items)
            n_elements += len(items)
            n_rows += 1
        return names, n_elements, n_rows

    def construct_domain(self, names):
        attributes = [ContinuousVariable.make(name) for name in sorted(names)]
        return Domain(attributes)

    def read_file(self, filename):
        with open(filename) as file:
            return self._read_file(file)

    def _read_file(self, file):
        names, n_elements, n_rows = self.prescan_file(file)
        domain = self.construct_domain(names)
        data = np.ones(n_elements)
        indices = np.empty(n_elements, dtype=int)
        indptr = np.empty(n_rows + 1, dtype=int)
        indptr[0] = curptr = 0

        file.seek(0)
        for row, line in enumerate(file):
            matches = [mo for mo in self.re_name.finditer(line)]
            items = {mo.group(1).strip(): float(mo.group(3) or 1)
                     for mo in matches}
            if len(matches) != len(items):
                counts = collections.Counter(mo.group(1).strip()
                                             for mo in matches)
                multiples = ["'%s'" % k for k, v in counts.items() if v > 1]
                if len(multiples) > 1:
                    multiples.sort()
                    attrs = "attributes %s and %s" % (
                        ", ".join(multiples[:-1]), multiples[-1])
                else:
                    attrs = "attribute " + multiples[0]
                warnings.warn(
                    "Ignoring multiple values for %s in row %i" %
                    (attrs, row + 1))
            nextptr = curptr + len(items)
            data[curptr:nextptr] = list(items.values())
            indices[curptr:nextptr] = [domain.index(name) for name in items]
            indptr[row + 1] = nextptr
            curptr = nextptr
        X = sparse.csr_matrix((data, indices, indptr),
                              (n_rows, len(domain.variables)))
        from ..data import Table

        return Table.new_from_numpy(domain, X)
