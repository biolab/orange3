from contextlib import contextmanager
from datetime import date, datetime, time
from typing import Optional, List, Iterable, Tuple, Any, Union

from sqlalchemy import create_engine, MetaData, select, Table, text, func
from sqlalchemy.exc import NoSuchTableError, ProgrammingError
from sqlalchemy.sql import Select
from sqlalchemy.sql.ddl import DDLElement
from sqlalchemy.sql.elements import and_, TextClause
from sqlalchemy.sql.sqltypes import NullType
from sqlalchemy.ext import compiler

from Orange.data import (
    StringVariable,
    TimeVariable,
    ContinuousVariable,
    DiscreteVariable,
    Domain,
)
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import ToSql, BackendError, TableDesc


class CreateTableAs(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


@compiler.compiles(CreateTableAs)
def compile(element, _, **kw):
    # in case any backend uses different syntax reimplement this function
    # for it
    return "CREATE TABLE %s AS %s" % (element.name, element.selectable)


class SQLAlchemyBackend(Backend):
    connection_string = (
        "{dialect_driver}://{user}:{password}@{host}:"
        "{port}/{database}?charset=utf8"
    )
    dialect_driver = None

    def __init__(self, connection_params: dict):
        print("init")
        super().__init__(connection_params)
        self.engine = create_engine(
            self.connection_string.format(
                dialect_driver=self.dialect_driver, **connection_params
            )
        )

    def list_tables(self, schema: Optional[str] = None):
        if not schema:
            schema = None
        tables = []
        for t in self.engine.table_names(schema=schema):
            s_t = (schema, t,) if schema else (t,)
            tables.append(TableDesc(t, schema, ".".join(s_t)))
        return tables

    def create_sql_query(
        self,
        table_name: str,
        fields: List[str],
        filters: Iterable[str] = (),
        group_by: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        use_time_sample: Optional[int] = None,
    ) -> Select:
        stn = table_name.split(".")
        schema, table_name = (None, stn[0]) if len(stn) == 1 else stn
        meta = MetaData(bind=self.engine, schema=schema)
        try:
            table = Table(table_name, meta, autoload=True)
        except NoSuchTableError:
            # when from SQL sentence - custom SQL in Orange
            table = text(table_name)

        columns = []
        for f in fields:
            if isinstance(table, TextClause):
                columns.append(text(f))
            elif "AS" in f:
                col, label = f.split("AS")
                columns.append(
                    table.c[col.strip("() ")].label(label.strip("() "))
                )
            elif "(" in f or f == "*":
                # fields is a function
                # TODO: think about not allowing this
                #  make separate functions for e.g. count
                columns.append(text(f))
            else:
                columns.append(table.c[f])

        query = select(columns).select_from(table)
        # MSSQL requires an order_by when using an OFFSET or a non-simple
        # LIMIT clause
        # TODO: check if order_by(None) would be fine
        if offset and not order_by:
            order_by = [x.strip('" ') for x in fields[0].split("AS")[1:]]

        if use_time_sample is not None:
            query = query.tablesample(func.system_time(1000))
        if filters:
            query = query.where(and_(text(f) for f in filters))
        if order_by is not None:
            query = query.order_by(*[text(o) for o in order_by])
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)
        if group_by is not None:
            query = query.group_by(*[text(g) for g in group_by])
        print(query)
        return query

    @contextmanager
    def execute_sql_query(
        self, query: Union[Select, str], params: Optional[Tuple[Any]] = ()
    ):
        with self.engine.connect() as connection:
            try:
                result = connection.execute(
                    text(query) if isinstance(query, str) else query, *params
                )
                yield result
                result.close()
            except ProgrammingError as ex:
                raise BackendError(str(ex)) from ex

    def get_fields(self, table_name: str):
        query = self.create_sql_query(table_name, ["*"], limit=3)
        types = {
            c.name: c.type.python_type
            for c in query.inner_columns
            if not isinstance(c.type, NullType)
        }

        # for plain textual SQL query types cannot be retrieved form the query
        # so we get missing types from data
        with self.execute_sql_query(query) as cur:
            res = cur.fetchall()
            missing = set(cur.keys()) - set(types.keys())
            for col in missing:
                if len(res) > 0:
                    t = set([type(r[col]) for r in res])
                    assert len(t) == 1  # types must match
                    (t,) = t  # unpack set
                else:
                    t = str
                types[col] = t
        return list(types.items())

    def _guess_variable(
        self,
        field_name: str,
        field_metadata: Tuple,
        inspect_table: Optional[str],
    ):
        type_ = field_metadata[0]

        if type_ == float:
            return ContinuousVariable.make(field_name)

        if type_ in (datetime, date, time):
            return TimeVariable(
                field_name,
                have_date=type_ in (date, datetime),
                have_time=type_ in (time, datetime),
            )

        if type_ == int:
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable(field_name, values)
            return ContinuousVariable(field_name)

        if type_ == bool:
            return DiscreteVariable(field_name, ["false", "true"])

        if type_ == str:
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                # remove trailing spaces
                values = [v.rstrip() for v in values]
                if values:
                    return DiscreteVariable(field_name, values)

        return StringVariable(field_name)

    def create_variable(
        self,
        field_name: str,
        field_metadata: Tuple[Any],
        type_hints: Domain,
        inspect_table: Optional[str] = None,
    ):
        if field_name in type_hints:
            var = type_hints[field_name]
        else:
            var = self._guess_variable(
                field_name, field_metadata, inspect_table
            )

        field_name_q = self.quote_identifier(field_name)
        if var.is_continuous:
            if isinstance(var, TimeVariable):
                var.to_sql = ToSql(field_name_q)
            else:
                var.to_sql = ToSql(field_name_q)
        else:  # discrete or string
            var.to_sql = ToSql(field_name_q)
        return var

    def count_approx(self, query: Select):
        """
        Count is faster than fetching complete table
        """
        q = query.alias("subquery")
        q = select([text("COUNT(*)")]).select_from(q)
        with self.execute_sql_query(q) as cur:
            return cur.fetchone()[0]

    def unquote_identifier(self, quoted_name: str) -> str:
        return quoted_name

    def quote_identifier(self, name: str) -> str:
        return name

    def create_table(self, name: str, sql: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(CreateTableAs(name, sql))

    def drop_table(self, name):
        stn = name.split(".")
        schema, table_name = (None, stn[0]) if len(stn) == 1 else stn
        meta = MetaData(bind=self.engine, schema=schema)
        try:
            table = Table(table_name, meta, autoload=True)
        except NoSuchTableError:
            return
        table.drop()

    def table_exists(self, name: str) -> bool:
        return self.engine.dialect.has_table(self.engine, name)


class MSSqlAlchemy(SQLAlchemyBackend):
    display_name = "MS Server Alchemy"
    dialect_driver = "mssql+pymssql"


class MySqlAlchemy(SQLAlchemyBackend):
    display_name = "MySQL Alchemy"
    # we decided to use mysqlclient from pypi
    # installed via: pip install mysqlclient
    dialect_driver = "mysql+mysqldb"


class SqliteAlchemy(SQLAlchemyBackend):
    display_name = "Sqllite"
    # requirement sqlite3 - included in the standard module
    dialect_driver = "sqlite+pysqlite"
    connection_string = "{dialect_driver}:///{database}"
