import csv
import re
import sys
from itertools import chain

import bottlechest as bn
import numpy as np
from scipy import sparse

from ..data import _io
from ..data import Domain
from ..data.variable import *


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
                if not col in Variable._DefaultUnknownStr and "." in val:
                    decs = len(val) - val.find(".") - 1
                    if decs > decimals[col]:
                        decimals[col] = decs
        return values, decimals


class TabDelimReader:
    non_escaped_spaces = re.compile(r"(?<!\\) +")

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
            flag = self.non_escaped_spaces.split(flag)
            flag = [f.replace("\\ ", " ") for f in flag]
            if "i" in flag or "ignore" in flag:
                continue
            if "b" in flag or "basket" in flag:
                self.basket_column = col
                continue
            is_class = "class" in flag
            is_meta = "m" in flag or "meta" in flag or tpe in ["s", "string"]
            is_weight = "w" in flag or "weight" in flag \
                or tpe in ["w", "weight"]

            attrs = [f.split("=", 1) for f in flag if "=" in f]

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
                values = [v.replace("\\ ", " ")
                          for v in self.non_escaped_spaces.split(tpe)]
                var = DiscreteVariable.make(name, values, True)
            var.fix_order = (isinstance(var, DiscreteVariable)
                             and not var.values)

            var.attributes.update(attrs)

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
        X, Y = table.X, table.Y
        W = table.W if table.W.shape[-1] else None
        f.seek(0)
        f.readline()
        f.readline()
        f.readline()
        padding = [""] * self.n_columns
        if self.basket_column >= 0:
            # TODO how many columns?!
            table._Xsparse = sparse.lil_matrix(len(X), 100)
        table.metas = metas = (
            np.empty((len(X), len(self.meta_columns)), dtype=object))
        line_count = 0
        Xr = None
        for lne in f:
            values = lne
            if not values.strip():
                continue
            values = values.split("\t")
            if len(values) > self.n_columns:
                raise ValueError("Too many columns in line {}".
                                 format(4 + line_count))
            elif len(values) < self.n_columns:
                values += padding
            if self.attribute_columns:
                Xr = X[line_count]
                for i, (col, reader) in enumerate(self.attribute_columns):
                    Xr[i] = reader(values[col].strip())
            for i, (col, reader) in enumerate(self.classvar_columns):
                Y[line_count, i] = reader(values[col].strip())
            if W is not None:
                W[line_count] = float(values[self.weight_column])
            for i, (col, reader) in enumerate(self.meta_columns):
                metas[line_count, i] = reader(values[col].strip())
            line_count += 1
        if line_count != len(X):
            del Xr, X, Y, W, metas
            table.X.resize(line_count, len(table.domain.attributes))
            table.Y.resize(line_count, len(table.domain.class_vars))
            if table.W.ndim == 1:
                table.W.resize(line_count)
            else:
                table.W.resize((line_count, 0))
            table.metas.resize((line_count, len(self.meta_columns)))
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
        self.reorder_values_array(table.X, table.domain.attributes)
        self.reorder_values_array(table.Y, table.domain.class_vars)

    def read_file(self, filename, cls=None):
        with open(filename) as file:
            return self._read_file(file, cls)

    def _read_file(self, file, cls=None):
        from ..data import Table
        if cls is None:
            cls = Table
        domain = self.read_header(file)
        nExamples = self.count_lines(file)
        table = cls.from_domain(domain, nExamples, self.weight_column >= 0)
        self.read_data(file, table)
        self.reorder_values(table)
        return table


class TxtReader:
    MISSING_VALUES = frozenset({"", "NA", "?"})

    @staticmethod
    def read_header(file, delimiter=None):
        first_line = file.readline()
        file.seek(0)
        if delimiter is None:
            for delimiter in "\t,; ":
                if delimiter in first_line:
                    break
            else:
                delimiter = None
        if delimiter == " ":
            delimiter = None
        atoms = first_line.split(delimiter)
        try:
            [float(atom) for atom in set(atoms) - TxtReader.MISSING_VALUES]
            header_lines = 0
            names = ["Var{:04}".format(i + 1) for i in range(len(atoms))]
        except ValueError:
            names = [atom.strip() for atom in atoms]
            header_lines = 1
        domain = Domain([ContinuousVariable.make(name) for name in names])
        return domain, header_lines, delimiter

    def read_file(self, filename, cls=None):
        from ..data import Table
        if cls is None:
            cls = Table
        with open(filename, "rt") as file:
            domain, header_lines, delimiter = self.read_header(file)
        with open(filename, "rb") as file:
            arr = np.genfromtxt(file, delimiter=delimiter,
                                skip_header=header_lines,
                                missing_values=self.MISSING_VALUES)
        table = cls.from_numpy(domain, arr)
        return table


class BasketReader():
    def read_file(self, filename, cls=None):
        if cls is None:
            from ..data import Table as cls
        def constr_vars(inds):
            if inds:
                return [ContinuousVariable(x.decode("utf-8")) for _, x in
                        sorted((ind, name) for name, ind in inds.items())]

        X, Y, metas, attr_indices, class_indices, meta_indices = \
            _io.sparse_read_float(filename.encode(sys.getdefaultencoding()))

        attrs = constr_vars(attr_indices)
        classes = constr_vars(class_indices)
        meta_attrs = constr_vars(meta_indices)
        domain = Domain(attrs, classes, meta_attrs)
        return cls.from_numpy(domain,
                              attrs and X, classes and Y, metas and meta_attrs)


def csv_saver(filename, data, delimiter='\t'):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        all_vars = data.domain.variables + data.domain.metas
        writer.writerow([v.name for v in all_vars])  # write variable names
        if delimiter == '\t':
            flags = ([''] * len(data.domain.attributes)) + \
                    (['class'] * len(data.domain.class_vars)) + \
                    (['m'] * len(data.domain.metas))

            for i, var in enumerate(all_vars):
                attrs = ["{0!s}={1!s}".format(*item).replace(" ", "\\ ")
                         for item in var.attributes.items()]
                if attrs:
                    flags[i] += (" " if flags[i] else "") + (" ".join(attrs))

            writer.writerow([type(v).__name__.replace("Variable", "").lower()
                             for v in all_vars])  # write variable types
            writer.writerow(flags) # write flags
        for ex in data: # write examples
            writer.writerow(ex)


def save_csv(filename, data):
    csv_saver(filename, data, ',')


def _save_tab_fast(f, data):
    wa = [var.repr_val for var in data.domain.variables + data.domain.metas]
    for Xi, Yi, Mi in zip(data.X, data.Y, data.metas):
        f.write("\t".join(w(val) for val, w in zip(chain(Xi, Yi, Mi), wa)))
        f.write("\n")


def save_tab_delimited(filename, data):
    """
    Save data to tab-delimited file.

    Function uses fast implementation in case of numpy data, and slower
    fall-back for general storage.

    :param filename: the name of the file
    :type filename: str
    :param data: the data to be saved
    :type data: Orange.data.Storage
    """
    f = open(filename, "w")
    domain_vars = data.domain.variables + data.domain.metas
    # first line
    f.write("\t".join([str(j.name) for j in domain_vars]))
    f.write("\n")

    # second line
    #TODO Basket column.
    t = {"ContinuousVariable": "c", "DiscreteVariable": "d",
         "StringVariable": "string", "Basket": "basket"}

    f.write("\t".join([t[type(j).__name__] for j in domain_vars]))
    f.write("\n")

    # third line
    m = list(data.domain.metas)
    c = list(data.domain.class_vars)
    r = []
    for i in domain_vars:
        if i in m:
            r.append("m")
        elif i in c:
            r.append("class")
        else:
            r.append("")
    f.write("\t".join(r))
    f.write("\n")

    # data
    # noinspection PyBroadException
    try:
        _save_tab_fast(f, data)
    except:
        domain_vars = [data.domain.index(var) for var in domain_vars]
        for i in data:
            f.write("\t".join(str(i[j]) for j in domain_vars) + "\n")
    f.close()
