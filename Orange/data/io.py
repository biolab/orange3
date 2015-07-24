import csv
import re
import sys
import pickle
from itertools import chain

import bottlechest as bn
import numpy as np
from scipy import sparse
# We are not loading openpyxl here since it takes some time

from Orange.data import Domain
from Orange.data.variable import *


# A singleton simulated with a class
class FileFormats:
    formats = []
    names = {}
    writers = {}
    readers = {}
    img_writers = {}
    graph_writers = {}

    @classmethod
    def register(cls, name, extension):
        def f(format):
            cls.NAME = name
            cls.formats.append(format)
            cls.names[extension] = name
            if hasattr(format, "write_file"):
                cls.writers[extension] = format
            if hasattr(format, "read_file"):
                cls.readers[extension] = format
            if hasattr(format, "write_image"):
                cls.img_writers[extension] = format
            if hasattr(format, "write_graph"):
                cls.graph_writers[extension] = format
            return format

        return f


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


@FileFormats.register("Tab-delimited file", ".tab")
class TabDelimFormat:
    non_escaped_spaces = re.compile(r"(?<!\\) +")

    def read_header(self, f):
        f.seek(0)
        names = [x.strip() for x in f.readline().strip("\n\r").split("\t")]
        types = [x.strip() for x in f.readline().strip("\n\r").split("\t")]
        flags = [x.strip() for x in f.readline().strip("\n\r").split("\t")]
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
                var = DiscreteVariable()  # no name to bypass caching
                var.name = name
                var.fix_order = True
            elif tpe in ["s", "string"]:
                var = StringVariable.make(name)
            else:
                values = [v.replace("\\ ", " ")
                          for v in self.non_escaped_spaces.split(tpe)]
                var = DiscreteVariable.make(name, values, True)
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
        X, Y = table.X, table._Y
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
            table._Y.resize(line_count, len(table.domain.class_vars))
            if table.W.ndim == 1:
                table.W.resize(line_count)
            else:
                table.W.resize((line_count, 0))
            table.metas.resize((line_count, len(self.meta_columns)))
        table.n_rows = line_count

    def reorder_values_array(self, arr, variables):
        newvars = []
        for col, var in enumerate(variables):
            if getattr(var, "fix_order", False):
                nvar = var.make(var.name, var.values, var.ordered)
                nvar.attributes = var.attributes
                move = len(var.values)
                if nvar.values != var.values:
                    arr[:, col] += move
                    for i, val in enumerate(var.values):
                        bn.replace(arr[:, col], move + i, nvar.values.index(val))
                var = nvar
            newvars.append(var)
        return newvars

    def reorder_values(self, table):
        attrs = self.reorder_values_array(table.X, table.domain.attributes)
        classes = self.reorder_values_array(table._Y, table.domain.class_vars)
        metas = self.reorder_values_array(table.metas, table.domain.metas)
        table.domain = Domain(attrs, classes, metas=metas)

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

    @classmethod
    def _write_fast(cls, f, data):
        wa = [var.str_val for var in data.domain.variables + data.domain.metas]
        for Xi, Yi, Mi in zip(data.X, data._Y, data.metas):
            f.write("\t".join(w(val) for val, w in zip(chain(Xi, Yi, Mi), wa)))
            f.write("\n")

    @classmethod
    def write_file(cls, filename, data):
        """
        Save data to file.

        Function uses fast implementation in case of numpy data, and slower
        fall-back for general storage.

        :param filename: the name of the file
        :type filename: str
        :param data: the data to be saved
        :type data: Orange.data.Storage
        """
        if isinstance(filename, str):
            f = open(filename, "w")
        else:
            f = filename
        domain_vars = data.domain.variables + data.domain.metas
        # first line
        f.write("\t".join([str(j.name) for j in domain_vars]))
        f.write("\n")

        # second line
        # TODO Basket column.
        t = {"ContinuousVariable": "c", "DiscreteVariable": "d",
             "StringVariable": "string", "Basket": "basket"}

        f.write("\t".join([t[type(j).__name__] for j in domain_vars]))
        f.write("\n")

        # third line
        m = list(data.domain.metas)
        c = list(data.domain.class_vars)
        r = []
        for i in domain_vars:
            r1 = ["{}={}".format(k, v).replace(" ", "\\ ")
                  for k, v in i.attributes.items()]
            if i in m:
                r1.append("m")
            elif i in c:
                r1.append("class")
            r.append(" ".join(r1))
        f.write("\t".join(r))
        f.write("\n")

        # data
        # noinspection PyBroadException
        try:
            cls._write_fast(f, data)
        except:
            domain_vars = [data.domain.index(var) for var in domain_vars]
            for i in data:
                f.write("\t".join(str(i[j]) for j in domain_vars) + "\n")
        f.close()

    def write(self, filename, data):
        self.write_file(filename, data)


@FileFormats.register("Comma-separated file", ".csv")
class TxtFormat:
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
            [float(atom) for atom in set(atoms) - TxtFormat.MISSING_VALUES]
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

    @classmethod
    def csv_saver(cls, filename, data, delimiter='\t'):
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
                writer.writerow(flags)  # write flags
            for ex in data:  # write examples
                writer.writerow(ex)

    @classmethod
    def write_file(cls, filename, data):
        cls.csv_saver(filename, data, ',')

    def write(self, filename, data):
        self.write_file(filename, data)


@FileFormats.register("Basket file", ".basket")
class BasketFormat:
    @classmethod
    def read_file(cls, filename, storage_class=None):
        from Orange.data import _io

        if storage_class is None:
            from ..data import Table as storage_class

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
        return storage_class.from_numpy(
            domain, attrs and X, classes and Y, metas and meta_attrs)


@FileFormats.register("Excel file", ".xlsx")
class ExcelFormat:
    non_escaped_spaces = re.compile(r"(?<!\\) +")

    def __init__(self):
        self.attribute_columns = []
        self.classvar_columns = []
        self.meta_columns = []
        self.weight_column = -1
        self.basket_column = -1

        self.n_columns = self.first_data_row = 0

    def open_workbook(self, f):
        from openpyxl import load_workbook

        if isinstance(f, str) and ":" in f[2:]:
            f, sheet = f.rsplit(":", 1)
        else:
            sheet = None
        wb = load_workbook(f, use_iterators=True,
                           read_only=True, data_only=True)
        ws = wb.get_sheet_by_name(sheet) if sheet else wb.get_active_sheet()
        self.n_columns = ws.get_highest_column()
        return ws

    # noinspection PyBroadException
    def read_header_3(self, worksheet):
        cols = self.n_columns
        try:
            names, types, flags = [
                [cell.value.strip() if cell.value is not None else ""
                 for cell in row]
                for row in worksheet.get_squared_range(1, 1, cols, 3)]
        except:
            return False
        if not (all(tpe in ("", "c", "d", "s", "continuous", "discrete",
                            "string", "w", "weight") or " " in tpe
                    for tpe in types) and
                    all(flg in ("", "i", "ignore", "m", "meta", "w", "weight",
                                "b", "basket", "class") or "=" in flg
                        for flg in flags)):
            return False
        attributes = []
        class_vars = []
        metas = []
        for col, (name, tpe, flag) in enumerate(zip(names, types, flags)):
            flag = self.non_escaped_spaces.split(flag)
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
                                     "class and weight".format(name, col + 1))
                self.weight_column = col
                continue
            if tpe in ["c", "continuous"]:
                var = ContinuousVariable.make(name)
            elif tpe in ["w", "weight"]:
                var = None
            elif tpe in ["d", "discrete"]:
                var = DiscreteVariable.make(name)
                var.fix_order = True
            elif tpe in ["s", "string"]:
                var = StringVariable.make(name)
            else:
                values = [v.replace("\\ ", " ")
                          for v in self.non_escaped_spaces.split(tpe)]
                var = DiscreteVariable.make(name, values, True)
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
        self.first_data_row = 4
        return Domain(attributes, class_vars, metas)

    # noinspection PyBroadException
    def read_header_0(self, worksheet):
        try:
            [float(cell.value) if cell.value is not None else None
             for cell in
             worksheet.get_squared_range(1, 1, self.n_columns, 3).__next__()]
        except:
            return False
        self.first_data_row = 1
        attrs = [ContinuousVariable.make("Var{:04}".format(i + 1))
                 for i in range(self.n_columns)]
        self.attribute_columns = [(i, var.val_from_str_add)
                                  for i, var in enumerate(attrs)]
        return Domain(attrs)

    def read_header_1(self, worksheet):
        import openpyxl.cell.cell

        if worksheet.get_highest_column() < 2 or \
                        worksheet.get_highest_row() < 2:
            return False
        cols = self.n_columns
        names = [cell.value.strip() if cell.value is not None else ""
                 for cell in
                 worksheet.get_squared_range(1, 1, cols, 3).__next__()]
        row2 = list(worksheet.get_squared_range(1, 2, cols, 3).__next__())
        attributes = []
        class_vars = []
        metas = []
        for col, name in enumerate(names):
            if "#" in name:
                flags, name = name.split("#", 1)
            else:
                flags = ""
            if "i" in flags:
                continue
            if "b" in flags:
                self.basket_column = col
                continue
            is_class = "c" in flags
            is_meta = "m" in flags or "s" in flags
            is_weight = "W" in flags or "w" in flags
            if is_weight:
                if is_class:
                    raise ValueError("Variable {} (column {}) is marked as "
                                     "class and weight".format(name, col))
                self.weight_column = col
                continue
            if "C" in flags:
                var = ContinuousVariable.make(name)
            elif is_weight:
                var = None
            elif "D" in flags:
                var = DiscreteVariable.make(name)
                var.fix_order = True
            elif "S" in flags:
                var = StringVariable.make(name)
            elif row2[col].data_type == "n":
                var = ContinuousVariable.make(name)
            else:
                if len(set(row[col].value for row in worksheet.rows)) > 20:
                    var = StringVariable.make(name)
                    is_meta = True
                else:
                    var = DiscreteVariable.make(name)
                    var.fix_order = True
            if is_class:
                if is_meta:
                    raise ValueError(
                        "Variable {} (column {}) is marked as "
                        "class and meta attribute".format(
                            name, openpyxl.cell.cell.get_column_letter(col + 1))
                    )
                class_vars.append(var)
                self.classvar_columns.append((col, var.val_from_str_add))
            elif is_meta:
                metas.append(var)
                self.meta_columns.append((col, var.val_from_str_add))
            else:
                attributes.append(var)
                self.attribute_columns.append((col, var.val_from_str_add))
        if attributes and not class_vars:
            class_vars.append(attributes.pop(-1))
            self.classvar_columns.append(self.attribute_columns.pop(-1))
        self.first_data_row = 2
        return Domain(attributes, class_vars, metas)

    def read_header(self, worksheet):
        domain = self.read_header_3(worksheet) or \
                 self.read_header_0(worksheet) or \
                 self.read_header_1(worksheet)
        if domain is False:
            raise ValueError("Invalid header")
        return domain

    # noinspection PyPep8Naming,PyProtectedMember
    def read_data(self, worksheet, table):
        X, Y = table.X, table._Y
        W = table.W if table.W.shape[-1] else None
        if self.basket_column >= 0:
            # TODO how many columns?!
            table._Xsparse = sparse.lil_matrix(len(X), 100)
        table.metas = metas = (
            np.empty((len(X), len(self.meta_columns)), dtype=object))
        sheet_rows = worksheet.rows
        for _ in range(1, self.first_data_row):
            sheet_rows.__next__()
        line_count = 0
        Xr = None
        for row in sheet_rows:
            values = [cell.value for cell in row]
            if all(value is None for value in values):
                continue
            if self.attribute_columns:
                Xr = X[line_count]
                for i, (col, reader) in enumerate(self.attribute_columns):
                    v = values[col]
                    Xr[i] = reader(v.strip() if isinstance(v, str) else v)
            for i, (col, reader) in enumerate(self.classvar_columns):
                v = values[col]
                Y[line_count, i] = reader(
                    v.strip() if isinstance(v, str) else v)
            if W is not None:
                W[line_count] = float(values[self.weight_column])
            for i, (col, reader) in enumerate(self.meta_columns):
                v = values[col]
                metas[line_count, i] = reader(
                    v.strip() if isinstance(v, str) else v)
            line_count += 1
        if line_count != len(X):
            del Xr, X, Y, W, metas
            table.X.resize(line_count, len(table.domain.attributes))
            table._Y.resize(line_count, len(table.domain.class_vars))
            if table.W.ndim == 1:
                table.W.resize(line_count)
            else:
                table.W.resize((line_count, 0))
            table.metas.resize((line_count, len(self.meta_columns)))
        table.n_rows = line_count

    # noinspection PyUnresolvedReferences
    @staticmethod
    def reorder_values_array(arr, variables):
        for col, var in enumerate(variables):
            if getattr(var, "fix_order", False) and len(var.values) < 1000:
                new_order = var.ordered_values(var.values)
                if new_order == var.values:
                    continue
                arr[:, col] += 1000
                for i, val in enumerate(var.values):
                    bn.replace(arr[:, col], 1000 + i, new_order.index(val))
                var.values = new_order
                delattr(var, "fix_order")

    # noinspection PyProtectedMember
    def reorder_values(self, table):
        self.reorder_values_array(table.X, table.domain.attributes)
        self.reorder_values_array(table._Y, table.domain.class_vars)
        self.reorder_values_array(table.metas, table.domain.metas)

    def read_file(self, file, cls=None):
        from Orange.data import Table

        if cls is None:
            cls = Table
        worksheet = self.open_workbook(file)
        domain = self.read_header(worksheet)
        table = cls.from_domain(
            domain,
            worksheet.get_highest_row() - self.first_data_row + 1,
            self.weight_column >= 0)
        self.read_data(worksheet, table)
        self.reorder_values(table)
        return table


@FileFormats.register("Pickled table", ".pickle")
class PickleFormat:
    @classmethod
    def read_file(cls, file, _=None):
        with open(file, "rb") as f:
            return pickle.load(f)

    @classmethod
    def write_file(cls, filename, table):
        with open(filename, "wb") as f:
            pickle.dump(table, f)

    def write(self, filename, table):
        self.write_file(filename, table)


@FileFormats.register("Dot Tree File", ".dot")
class DotFormat:
    @classmethod
    def write_graph(cls, filename, graph):
        from sklearn import tree

        tree.export_graphviz(graph, out_file=filename)

    def write(self, filename, tree):
        if type(tree) == dict:
            tree = tree['tree']
        self.write_graph(filename, tree)
