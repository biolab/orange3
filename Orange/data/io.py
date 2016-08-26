import contextlib
import csv
import locale
import os
import pickle
import re
import subprocess
import sys
import warnings
from tempfile import NamedTemporaryFile

from os import path, unlink
from ast import literal_eval
from math import isnan
from numbers import Number
from itertools import chain, repeat
from functools import lru_cache
from collections import OrderedDict
from urllib.parse import urlparse, unquote as urlunquote
from urllib.request import urlopen

import bottleneck as bn
import numpy as np
import pandas as pd
from chardet.universaldetector import UniversalDetector

from Orange.data import (
    _io, Table, Domain, Variable, DiscreteVariable,
    StringVariable, ContinuousVariable, TimeVariable,
)
from Orange.util import Registry, flatten, namegen


class Compression:
    """Supported compression extensions"""
    GZIP = '.gz'
    BZIP2 = '.bz2'
    XZ = '.xz'
    all = (GZIP, BZIP2, XZ)


def open_compressed(filename, *args, _open=open, **kwargs):
    """Return seamlessly decompressed open file handle for `filename`"""
    if isinstance(filename, str):
        if filename.endswith(Compression.GZIP):
            from gzip import open as _open
        elif filename.endswith(Compression.BZIP2):
            from bz2 import open as _open
        elif filename.endswith(Compression.XZ):
            from lzma import open as _open
        return _open(filename, *args, **kwargs)
    # Else already a file, just pass it through
    return filename


def detect_encoding(filename):
    """
    Detect encoding of `filename`, which can be a ``str`` filename, a
    ``file``-like object, or ``bytes``.
    """
    # Try with Unix file utility first because it's faster (~10ms vs 100ms)
    if isinstance(filename, str) and not filename.endswith(Compression.all):
        try:
            with subprocess.Popen(('file', '--brief', '--mime-encoding', filename),
                                  stdout=subprocess.PIPE) as process:
                process.wait()
                if process.returncode == 0:
                    encoding = process.stdout.read().strip()
                    # file only supports these encodings; for others it says
                    # unknown-8bit or binary. So we give chardet a chance to do
                    # better
                    if encoding in (b'utf-8', b'us-ascii', b'iso-8859-1',
                                    b'utf-7', b'utf-16le', b'utf-16be', b'ebcdic'):
                        return encoding.decode('us-ascii')
        except OSError: pass  # windoze

    # file not available or unable to guess the encoding, have chardet do it
    detector = UniversalDetector()
    # We examine only first N 4kB blocks of file because chardet is really slow
    MAX_BYTES = 4*1024*12

    def _from_file(f):
        detector.feed(f.read(MAX_BYTES))
        detector.close()
        return detector.result.get('encoding')

    if isinstance(filename, str):
        with open_compressed(filename, 'rb') as f:
            return _from_file(f)
    elif isinstance(filename, bytes):
        detector.feed(filename[:MAX_BYTES])
        detector.close()
        return detector.result.get('encoding')
    elif hasattr(filename, 'encoding'):
        return filename.encoding
    else:  # assume file-like object that you can iter through
        return _from_file(filename)


class Flags:
    """Parser for column flags (i.e. third header row)"""
    DELIMITER = ' '
    _RE_SPLIT = re.compile(r'(?<!\\)' + DELIMITER).split
    _RE_ATTR_UNQUOTED_STR = re.compile(r'^[a-zA-Z_]').match
    ALL = OrderedDict((
        ('class',     'c'),
        ('ignore',    'i'),
        ('meta',      'm'),
        ('weight',    'w'),
        ('.+?=.*?',    ''),  # general key=value attributes
    ))
    _RE_ALL = re.compile(r'^({})$'.format('|'.join(filter(None, flatten(ALL.items())))))

    def __init__(self, flags):
        for v in filter(None, self.ALL.values()):
            setattr(self, v, False)
        self.attributes = {}
        for flag in flags or []:
            flag = flag.strip()
            if self._RE_ALL.match(flag):
                if '=' in flag:
                    k, v = flag.split('=', 1)
                    self.attributes[k] = (v if Flags._RE_ATTR_UNQUOTED_STR(v) else
                                          literal_eval(v) if v else
                                          '')
                else:
                    setattr(self, flag, True)
                    setattr(self, self.ALL.get(flag, ''), True)
            elif flag:
                warnings.warn('Invalid attribute flag \'{}\''.format(flag))

    @staticmethod
    def join(iterable, *args):
        return Flags.DELIMITER.join(i.strip().replace(Flags.DELIMITER, '\\' + Flags.DELIMITER)
                                    for i in chain(iterable, args)).lstrip()

    @staticmethod
    def split(s):
        return [i.replace('\\' + Flags.DELIMITER, Flags.DELIMITER)
                for i in Flags._RE_SPLIT(s)]


# Matches discrete specification where all the values are listed, space-separated
_RE_DISCRETE_LIST = re.compile(r'^\s*[^\s]+(\s[^\s]+)+\s*$')
_RE_TYPES = re.compile(r'^\s*({}|{}|)\s*$'.format(_RE_DISCRETE_LIST.pattern,
                                                  '|'.join(flatten(getattr(vartype, 'TYPE_HEADERS')
                                                                   for vartype in Variable.registry.values()))))
_RE_FLAGS = re.compile(r'^\s*( |{}|)*\s*$'.format('|'.join(flatten(filter(None, i) for i in Flags.ALL.items()))))


class FileFormatMeta(Registry):
    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

        # Optionally add compressed versions of extensions as supported
        if getattr(newcls, 'SUPPORT_COMPRESSED', False):
            new_extensions = list(getattr(newcls, 'EXTENSIONS', ()))
            for compression in Compression.all:
                for ext in newcls.EXTENSIONS:
                    new_extensions.append(ext + compression)
            newcls.EXTENSIONS = tuple(new_extensions)

        return newcls

    @property
    def formats(self):
        return self.registry.values()

    @lru_cache(5)
    def _ext_to_attr_if_attr2(self, attr, attr2):
        """
        Return ``{ext: `attr`, ...}`` dict if ``cls`` has `attr2`.
        If `attr` is '', return ``{ext: cls, ...}`` instead.
        """
        return OrderedDict((ext, getattr(cls, attr, cls))
                            for cls in self.registry.values()
                            if hasattr(cls, attr2)
                            for ext in getattr(cls, 'EXTENSIONS', []))

    @property
    def names(self):
        return self._ext_to_attr_if_attr2('DESCRIPTION', '__class__')

    @property
    def writers(self):
        return self._ext_to_attr_if_attr2('', 'write_file')

    @property
    def readers(self):
        return self._ext_to_attr_if_attr2('', 'read')

    @property
    def img_writers(self):
        return self._ext_to_attr_if_attr2('', 'write_image')

    @property
    def graph_writers(self):
        return self._ext_to_attr_if_attr2('', 'write_graph')


class FileFormat(metaclass=FileFormatMeta):
    """
    Subclasses set the following attributes and override the following methods:

        EXTENSIONS = ('.ext1', '.ext2', ...)
        DESCRIPTION = 'human-readable file format description'
        SUPPORT_COMPRESSED = False

    --------------------------
    START CHOICE: Subclasses override either

        def read_header(self):
            # read only the first 3 rows into a raw pd.DataFrame
            # 3 because the header has {0, 1, 3} rows

    and

        def read_contents(self, skiprows):
            # read the whole file (with skipped rows) into a raw pd.DataFrame
            # raw means that no rows ar treated as columns, no columns as indices etc
            # skiprows determines how many rows to skip at the beginning of the file

    or, if the file format has no headers or is e.g. binary,

        def read(self):
            # return a complete, processed, pd.DataFrame/Table object

    END CHOICE
    --------------------------

        @classmethod
        def write_file(cls, filename, data):
            ...
            self.write_headers(writer.write, data)
            writer.writerows(data)
    """

    PRIORITY = 10000  # Sort order in OWSave widget combo box, lower is better

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            name of the file to open
        """
        self.filename = filename
        self.sheet = None

    @property
    def sheets(self):
        """FileFormats with a notion of sheets should override this property
        to return a list of sheet names in the file.

        Returns
        -------
        a list of sheet names
        """
        return ()

    def select_sheet(self, sheet):
        """Select sheet to be read

        Parameters
        ----------
        sheet : str
            sheet name
        """
        self.sheet = sheet

    @classmethod
    def get_reader(cls, filename):
        """Return reader instance that can be used to read the file

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """
        for ext, reader in cls.readers.items():
            if filename.endswith(ext):
                return reader(filename)

        raise IOError('No readers for file "{}"'.format(filename))

    @classmethod
    def write(cls, filename, data):
        return cls.write_file(filename, data)

    @classmethod
    def write_table_metadata(cls, filename, data):
        if isinstance(filename, str) and getattr(data, 'attributes', {}):
            with open(filename + '.metadata', 'wb') as f:
                pickle.dump(data.attributes, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def set_table_metadata(cls, filename, table):
        if isinstance(filename, str) and path.exists(filename + '.metadata'):
            with open(filename + '.metadata', 'rb') as f:
                table.attributes = pickle.load(f)

    @classmethod
    def locate(cls, filename, search_dirs=('.',)):
        """Locate a file with given filename that can be opened by one
        of the available readers.

        Parameters
        ----------
        filename : str
        search_dirs : Iterable[str]

        Returns
        -------
        str
            Absolute path to the file
        """
        if path.exists(filename):
            return filename

        for directory in search_dirs:
            absolute_filename = path.join(directory, filename)
            if path.exists(absolute_filename):
                break
            for ext in cls.readers:
                if filename.endswith(ext):
                    break
                if path.exists(absolute_filename + ext):
                    absolute_filename += ext
                    break
            if path.exists(absolute_filename):
                break
        else:
            absolute_filename = ""

        if not path.exists(absolute_filename):
            raise IOError('File "{}" was not found.'.format(filename))

        return absolute_filename

    @staticmethod
    def open(filename, *args, **kwargs):
        """
        Format handlers can use this method instead of the builtin ``open()``
        to transparently (de)compress files if requested (according to
        `filename` extension). Set ``SUPPORT_COMPRESSED=True`` if you use this.
        """
        return open_compressed(filename, *args, **kwargs)

    def read_header(self):
        """Subclasses may override this according to the details in FileFormat."""
        raise NotImplementedError()

    def read_contents(self, skiprows):
        """Subclasses may override this according to the details in FileFormat."""
        raise NotImplementedError()

    def read(self):
        """
        Subclasses may override this according to the details in FileFormat.
        If this is not overridden, it uses FileFormat.read_header and FileFormat.read_contents
        to construct a Table from headers and data.
        """
        def tryparse_int_float(item):
            try:
                return float(item)
            except ValueError:
                return None

        # Second row items are type identifiers
        def header_test2(items):
            return all(map(_RE_TYPES.match, items))

        # Third row items are flags and column attributes (attr=value)
        def header_test3(items):
            return all(map(_RE_FLAGS.match, items))

        # read and parse the header before reading the rest of the file
        header_df = self.read_header()
        is_small = len(header_df) < 3

        # the header can have 0, 1 or 3 rows
        # it is a three-row header if the second and third columns match the regex,
        # the first one can then be anything
        three_row_header = not is_small and header_test2(header_df.iloc[1].fillna('').astype(str)) \
                           and header_test3(header_df.iloc[2].fillna('').astype(str))
        # a one-row header has something that doesn't parse as a float in the first row
        one_row_header = not all(tryparse_int_float(i) is not None for i in header_df.iloc[0])
        if one_row_header or three_row_header:
            if three_row_header:
                names, types, flags = [list(header_df.iloc[i].fillna('')) for i in range(3)]
                contents = self.read_contents(skiprows=3)
            else:
                # one row header format either:
                #   1) delimited column names
                #   2) -||- with type and flags prepended, separated by #, e.g. d#sex,c#age,cC#IQ
                HEADER1_FLAG_SEP = '#'
                ft_combo, names = zip(*[i.split(HEADER1_FLAG_SEP, 1) if HEADER1_FLAG_SEP in i else ('', i)
                                        for i in header_df.iloc[0].fillna('')])
                names = list(names)
                types = [''.join(filter(str.isupper, flag)).lower() for flag in ft_combo]
                flags = [Flags.join(filter(str.islower, flag)) for flag in ft_combo]
                contents = self.read_contents(skiprows=1)

            # we may have no data, so construct an empty df to satisfy later needs
            if contents is None:
                contents = pd.DataFrame(data=np.empty((0, len(header_df.columns))), columns=header_df.columns)

            # transform any values we believe are null into actual null values
            contents.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)

            # data may be longer than the headers, extend headers with empty values
            names.extend([''] * (len(contents.columns) - len(names)))
            types.extend([''] * (len(contents.columns) - len(types)))
            flags.extend([''] * (len(contents.columns) - len(flags)))

            result = pd.DataFrame()
            weight_column = None  # separate, because it doesn't fit into the domain
            role_vars = {'x': [], 'y': [], 'meta': []}
            for col_idx, (name, typef, flag) in enumerate(zip(names, types, flags)):
                typef = typef.strip()
                flag = Flags(Flags.split(flag))
                col_type_kwargs = {}

                # some columns can be ignored
                if flag.i:
                    continue

                # determine column role
                if flag.m or typef in StringVariable.TYPE_HEADERS:
                    col_role = 'meta'
                elif flag.w:
                    # special execution path because this type is special
                    weight_column = contents[col_idx]
                    continue
                elif flag.c:
                    col_role = 'y'
                else:
                    col_role = 'x'

                # determine column type from header
                if typef in StringVariable.TYPE_HEADERS:
                    col_type = StringVariable
                elif typef in ContinuousVariable.TYPE_HEADERS:
                    col_type = ContinuousVariable
                elif typef in TimeVariable.TYPE_HEADERS:
                    col_type = TimeVariable
                elif typef in DiscreteVariable.TYPE_HEADERS:
                    col_type = DiscreteVariable
                elif _RE_DISCRETE_LIST.match(typef):
                    col_type = DiscreteVariable
                    # if possible, we want these to be numbers (as they will be in the table)
                    # but only do this if every value is parsed to a number, otherwise elements
                    # will be upcast to strings in the table and they won't actually be the same thing
                    raws = Flags.split(typef)
                    nums = [tryparse_int_float(a) for a in raws]
                    all_parsed = all(num is not None for num in nums)
                    col_type_kwargs.update(values=[num if num is not None and all_parsed else raw
                                                   for raw, num in zip(raws, nums)],
                                           ordered=True)
                else:
                    # infer from data
                    # if the initial role was x (not specified), allow this to modify it
                    col_type, col_role = Domain.infer_type_role(contents[col_idx],
                                                                force_role=col_role if col_role != 'x' else None)

                if col_type is DiscreteVariable and 'values' not in col_type_kwargs:
                    # for discrete variables that haven't specified their values in the header
                    col_type_kwargs.update(values=DiscreteVariable.generate_unique_values(contents[col_idx]))
                elif col_type is ContinuousVariable and typef in ContinuousVariable.TYPE_HEADERS:
                    # guard against reading non-continuous data into continuous columns
                    # (this happens when the column is specified manually, therefore typef check)
                    # this doesn't catch timevariable errors, even though they are continuous,
                    # even though timevariables are also continuous
                    if not np.issubdtype(contents[col_idx].dtype, np.number):
                        raise ValueError("Non-continuous data read into a continuous column "
                                         "(column {}: {}).".format(col_idx + 1, name))

                # use an existing variable if available, otherwise get a brand new one
                # with a brand new name
                new_name = Domain.infer_name(col_type, col_role, name,
                                             role_vars['x'], role_vars['y'], role_vars['meta'])
                if name == new_name:
                    var = col_type.make(new_name, **col_type_kwargs)
                else:
                    var = col_type(new_name, **col_type_kwargs)
                # also store any attributes in the third row (beside the role declaration)
                var.attributes.update(flag.attributes)

                role_vars[col_role].append(var)
                # strip whitespace from string/string-like columns (np.object_ in pandas)
                # important: not just object which is a superclass of everything probably,
                # object_ is the one that only encloses 'real' object-like types
                if np.issubdtype(contents[col_idx], np.object_):
                    result[var.name] = contents[col_idx].str.strip()
                else:
                    result[var.name] = contents[col_idx]

            domain = Domain(role_vars['x'], role_vars['y'], role_vars['meta'])
            result = Table.from_dataframe(domain, result, reindex=True, weights=weight_column)
        else:
            # there is no header, just read the file
            # and pass it to the proper constructor to infer columns
            result = Table.from_dataframe(None, self.read_contents(skiprows=0))
        result.consolidate(inplace=True)

        # TODO: Name can be set unconditionally when/if
        # self.filename will always be a string with the file name.
        # Currently, some tests pass StringIO instead of
        # the file name to a reader.
        if isinstance(self.filename, str):
            result.name = os.path.splitext(os.path.split(self.filename)[-1])[0]
        result.name = getattr(self, 'force_name', None) or result.name
        return result

    @staticmethod
    def header_names(data):
        return ([data._WEIGHTS_COLUMN] if data.has_weights else []) + \
               [v.name for v in chain(data.domain.attributes,
                                      data.domain.class_vars,
                                      data.domain.metas)]

    @staticmethod
    def header_types(data):
        def _vartype(var):
            if var.is_continuous or var.is_string:
                return var.TYPE_HEADERS[0]
            elif var.is_discrete:
                return Flags.join(var.values) if var.ordered else var.TYPE_HEADERS[0]
            raise NotImplementedError
        return (['continuous'] if data.has_weights else []) + \
               [_vartype(v) for v in chain(data.domain.attributes,
                                           data.domain.class_vars,
                                           data.domain.metas)]

    @staticmethod
    def header_flags(data):
        return list(chain(['weight'] if data.has_weights else [],
                          (Flags.join([flag], *('{}={}'.format(*a)
                                                for a in sorted(var.attributes.items())))
                           for flag, var in chain(zip(repeat(''),  data.domain.attributes),
                                                  zip(repeat('class'), data.domain.class_vars),
                                                  zip(repeat('meta'), data.domain.metas)))))

    @classmethod
    def write_headers(cls, write, data):
        """`write` is a callback that accepts an iterable"""
        write(cls.header_names(data))
        write(cls.header_types(data))
        write(cls.header_flags(data))

    @classmethod
    def write_data(cls, write, data):
        """`write` is a callback that accepts an iterable"""
        vars = list(chain((ContinuousVariable(data._WEIGHTS_COLUMN),) if data.has_weights else [],
                          data.domain.attributes,
                          data.domain.class_vars,
                          data.domain.metas))
        for idx, row in data.iterrows():
            row_filtered = row[vars]
            write(list(row_filtered))


class CSVReader(FileFormat):
    """Reader for comma separated files"""
    EXTENSIONS = ('.csv',)
    DESCRIPTION = 'Comma-separated values'
    DELIMITERS = ',;:\t$ '
    SUPPORT_COMPRESSED = True
    PRIORITY = 20

    def __init__(self, filename):
        super().__init__(filename)
        self.actual_delimiter = None

    def sniff_delimiter(self):
        """Sniff a delimiter from a limited sample of the data.

        Can cache already-sniffed delimiters and falls back to "\t" if unable to sniff.

        Returns
        -------
        str
            The sniffed delimiter.
        """
        if self.actual_delimiter is not None:
            return self.actual_delimiter

        # sniff the separator for efficiency (inferring with pandas forces the python engine)
        sniffer = csv.Sniffer()
        try:
            if hasattr(self.filename, 'read'):  # if a file-like object is passed
                self.filename.seek(0)
                sample = self.filename.read(4096)
                self.filename.seek(0)
                delimiter = sniffer.sniff(sample).delimiter
            else:
                with open(self.filename, encoding='utf-8') as f:
                    delimiter = sniffer.sniff(f.read(4096)).delimiter
        except csv.Error:
            # sometimes sniffing fails, fall back to the default delimiter
            # (pandas won't solve this as it uses the same internally)
            delimiter = self.DELIMITERS[0]

        # only allow a delimiter that is in the delimiters
        if delimiter not in self.DELIMITERS:
            delimiter = self.DELIMITERS[0]
        self.actual_delimiter = delimiter
        return delimiter

    def read_header(self):
        # restrict to cls.delimiters, this also stabilizes some weird behaviour
        # when there is not a lot of data to infer the delimiter
        # don't skip blank lines on case of an empty third header line
        return pd.read_table(self.filename,
                             sep=self.sniff_delimiter(), header=None, index_col=False, skipinitialspace=True,
                             skip_blank_lines=False, parse_dates=False,
                             compression='infer', engine='python', nrows=3)

    def read_contents(self, skiprows):
        # be sure
        if hasattr(self.filename, 'read'):
            self.filename.seek(0)
        try:
            # don't parse dates, we want more control over timezones
            # see TimeVariable.column_to_datetime
            # fix the sniffed delimiter,
            # skip blank lines here, we're not interested in them
            return pd.read_table(self.filename,
                                 sep=self.sniff_delimiter(), header=None, index_col=False, skipinitialspace=True,
                                 skip_blank_lines=True, parse_dates=False,
                                 compression='infer', skiprows=skiprows, na_values=Variable.MISSING_VALUES)
        except pd.io.common.EmptyDataError:
            # if there is only the header, signal no data
            return None

    def read(self):
        """Apart from reading the table, this sets metadata. """
        res = super(CSVReader, self).read()
        self.set_table_metadata(self.filename, res)
        return res


    @classmethod
    def write_file(cls, filename, data):
        with cls.open(filename, mode='wt', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=cls.DELIMITERS[0])
            cls.write_headers(writer.writerow, data)
            cls.write_data(writer.writerow, data)
        cls.write_table_metadata(filename, data)


class TabReader(CSVReader):
    """Reader for tab separated files"""
    EXTENSIONS = ('.tab', '.tsv')
    DESCRIPTION = 'Tab-separated values'
    DELIMITERS = '\t'
    PRIORITY = 10


class PickleReader(FileFormat):
    """Reader for pickled Table objects"""
    EXTENSIONS = ('.pickle', '.pkl')
    DESCRIPTION = 'Pickled Python object file'

    def read(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def write_file(filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class BasketReader(FileFormat):
    """Reader for basket (sparse) files"""
    EXTENSIONS = ('.basket', '.bsk')
    DESCRIPTION = 'Basket file'

    def read(self):
        def constr_vars(inds):
            if inds:
                return [ContinuousVariable(x.decode("utf-8")) for _, x in
                        sorted((ind, name) for name, ind in inds.items())]

        X, Y, metas, attr_indices, class_indices, meta_indices = \
            _io.sparse_read_float(self.filename.encode(sys.getdefaultencoding()))

        attrs = constr_vars(attr_indices)
        classes = constr_vars(class_indices)
        meta_attrs = constr_vars(meta_indices)
        domain = Domain(attrs, classes, meta_attrs)
        table = Table.from_numpy(
            domain, attrs and X, classes and Y, metas and meta_attrs)
        table.name = os.path.splitext(os.path.split(self.filename)[-1])[0]

        return table


class ExcelReader(FileFormat):
    """Reader for excel files"""
    EXTENSIONS = ('.xls', '.xlsx')
    DESCRIPTION = 'Microsoft Excel spreadsheet'

    def __init__(self, filename):
        super().__init__(filename)
        # still need to open this to get a list of sheets
        from xlrd import open_workbook
        self.workbook = open_workbook(self.filename)

    @property
    @lru_cache(1)
    def sheets(self):
        return self.workbook.sheet_names()

    def read_header(self):
        # we can't read just the header, we must read the entire file
        # pandas also doesn't automatically recognize the area the data is in
        # and starts reading from the top left: we need to specify the area manually
        try:
            shet = self.workbook.sheet_by_name(self.sheet) if self.sheet else self.workbook.sheet_by_index(0)
            self.first_row_idx = next(i for i in range(shet.nrows) if any(shet.row_values(i)))
            self.first_col_idx = next(i for i in range(shet.ncols) if shet.cell_value(self.first_row_idx, i))
            self.last_col_idx = shet.row_len(self.first_row_idx)
            self.force_name = os.path.splitext(os.path.split(self.filename)[-1])[0]
            if self.sheet:
                self.force_name = '-'.join((self.force_name, self.sheet))
        except Exception as e:
            raise IOError("Couldn't load spreadsheet from " + self.filename)

        raw_data = pd.read_excel(self.workbook, sheetname=self.sheet or 0,
                                 header=None, index_col=None, engine='xlrd', skiprows=self.first_row_idx,
                                 parse_cols=range(self.first_col_idx, self.last_col_idx))
        return raw_data.iloc[:3]

    def read_contents(self, skiprows):
        # reading again so pandas correctly determines data types of columns (without the header)
        raw_data = pd.read_excel(self.workbook, sheetname=self.sheet or 0,
                                 header=None, index_col=None, engine='xlrd', skiprows=self.first_row_idx + skiprows,
                                 parse_cols=range(self.first_col_idx, self.last_col_idx))
        return raw_data


class DotReader(FileFormat):
    """Writer for dot (graph) files"""
    EXTENSIONS = ('.dot', '.gv')
    DESCRIPTION = 'Dot graph description'
    SUPPORT_COMPRESSED = True

    @classmethod
    def write_graph(cls, filename, graph):
        from sklearn import tree
        tree.export_graphviz(graph, out_file=cls.open(filename, 'wt'))

    @classmethod
    def write(cls, filename, tree):
        if type(tree) == dict:
            tree = tree['tree']
        cls.write_graph(filename, tree)


class UrlReader(FileFormat):
    def read(self):
        self.filename = self._trim(self._resolve_redirects(self.filename))

        with contextlib.closing(urlopen(self.filename, timeout=10)) as response:
            name = self._suggest_filename(response.headers['content-disposition'])
            with NamedTemporaryFile(suffix=name, delete=False) as f:
                f.write(response.read())
                # delete=False is a workaround for https://bugs.python.org/issue14243

            reader = self.get_reader(f.name)
            data = reader.read()
            unlink(f.name)
        # Override name set in from_file() to avoid holding the temp prefix
        data.name = path.splitext(name)[0]
        data.origin = self.filename
        return data

    def _resolve_redirects(self, url):
        # Resolve (potential) redirects to a final URL
        with contextlib.closing(urlopen(url, timeout=10)) as response:
            return response.url

    def _trim(self, url):
        URL_TRIMMERS = (
            self._trim_googlesheet_url,
        )
        for trim in URL_TRIMMERS:
            try:
                url = trim(url)
            except ValueError:
                continue
            else:
                break
        return url

    def _trim_googlesheet_url(self, url):
        match = re.match(r'(?:https?://)?(?:www\.)?'
                         'docs\.google\.com/spreadsheets/d/'
                         '(?P<workbook_id>[-\w_]+)'
                         '(?:/.*?gid=(?P<sheet_id>\d+).*|.*)?',
                         url, re.IGNORECASE)
        try:
            workbook, sheet = match.group('workbook_id'), match.group('sheet_id')
            if not workbook:
                raise ValueError
        except (AttributeError, ValueError):
            raise ValueError
        url = 'https://docs.google.com/spreadsheets/d/{}/export?format=tsv'.format(workbook)
        if sheet:
            url += '&gid=' + sheet
        return url

    def _suggest_filename(self, content_disposition):
        default_name = re.sub(r'[\\:/]', '_', urlparse(self.filename).path)

        # See https://tools.ietf.org/html/rfc6266#section-4.1
        matches = re.findall(r"filename\*?=(?:\"|.{0,10}?'[^']*')([^\"]+)",
                             content_disposition or '')
        return urlunquote(matches[-1]) if matches else default_name
