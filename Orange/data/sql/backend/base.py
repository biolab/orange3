import logging
from contextlib import contextmanager

from Orange.util import Registry

log = logging.getLogger(__name__)


class BackendError(Exception):
    pass


class Backend(metaclass=Registry):
    """Base class for SqlTable backends. Implementations should define
    all of the methods defined below.

    Parameters
    ----------
    connection_params: dict
        connection params
    """

    display_name = ""

    def __init__(self, connection_params):
        self.connection_params = connection_params

    @classmethod
    def available_backends(cls):
        """Return a list of all available backends"""
        return cls.registry.values()

    # "meta" methods

    def list_tables_query(self, schema=None):
        """Return a list of tuples (schema, table_name)

        Parameters
        ----------
        schema : Optional[str]
            If set, only tables from schema should be listed

        Returns
        -------
        A list of tuples
        """
        raise NotImplementedError

    def list_tables(self, schema=None):
        """Return a list of tables in database

        Parameters
        ----------
        schema : Optional[str]
            If set, only tables from given schema will be listed

        Returns
        -------
        A list of TableDesc objects, describing the tables in the database
        """
        query = self.list_tables_query(schema)
        with self.execute_sql_query(query) as cur:
            tables = []
            for schema, name in cur.fetchall():
                sql = "{}.{}".format(
                    self.quote_identifier(schema),
                    self.quote_identifier(name)) if schema else self.quote_identifier(name)
                tables.append(TableDesc(name, schema, sql))
            return tables

    def get_fields(self, table_name):
        """Return a list of field names and metadata in the given table

        Parameters
        ----------
        table_name: str

        Returns
        -------
        a list of tuples (field_name, *field_metadata)
        both will be passed to create_variable
        """
        query = self.create_sql_query(table_name, ["*"], limit=0)
        with self.execute_sql_query(query) as cur:
            return cur.description

    def get_distinct_values(self, field_name, table_name):
        """Return a list of distinct values of field

        Parameters
        ----------
        field_name : name of the field
        table_name : name of the table or query to search

        Returns
        -------
        List[str] of values
        """
        fields = [self.quote_identifier(field_name)]

        query = self.create_sql_query(table_name, fields,
                                      group_by=fields, order_by=fields,
                                      limit=21)
        with self.execute_sql_query(query) as cur:
            values = cur.fetchall()
        if len(values) > 20:
            return ()
        else:
            return tuple(str(x[0]) for x in values)

    def create_variable(self, field_name, field_metadata,
                        type_hints, inspect_table=None):
        """Create variable based on field information

        Parameters
        ----------
        field_name : str
            name do the field
        field_metadata : tuple
            data to guess field type from
        type_hints : Domain
            domain with variable templates
        inspect_table : Option[str]
            name of the table to expect the field values or None
            if no inspection is to be performed

        Returns
        -------
        Variable representing the field
        """
        raise NotImplementedError

    def count_approx(self, query):
        """Return estimated number of rows returned by query.

        Parameters
        ----------
        query : str

        Returns
        -------
        Approximate number of rows
        """
        raise NotImplementedError

    # query related methods

    def create_sql_query(
            self, table_name, fields, filters=(),
            group_by=None, order_by=None, offset=None, limit=None,
            use_time_sample=None):
        """Construct an sql query using the provided elements.

        Parameters
        ----------
        table_name : str
        fields : List[str]
        filters : List[str]
        group_by: List[str]
        order_by: List[str]
        offset: int
        limit: int
        use_time_sample: int

        Returns
        -------
        string containing sql query
        """
        raise NotImplementedError

    @contextmanager
    def execute_sql_query(self, query, params=None):
        """Context manager for execution of sql queries

        Usage:
            ```
            with backend.execute_sql_query("SELECT * FROM foo") as cur:
                cur.fetch_all()
            ```

        Parameters
        ----------
        query : string
            query to be executed
        params: tuple
            parameters to be passed to the query

        Returns
        -------
        yields a cursor that can be used to access the data
        """
        raise NotImplementedError

    def quote_identifier(self, name):
        """Quote identifier name so it can be safely used in queries

        Parameters
        ----------
        name: str
            name of the parameter

        Returns
        -------
        quoted parameter that can be used in sql queries
        """
        raise NotImplementedError

    def unquote_identifier(self, quoted_name):
        """Remove quotes from identifier name
        Used when sql table name is used in where parameter to
        query special tables

        Parameters
        ----------
        quoted_name : str

        Returns
        -------
        unquoted name
        """
        raise NotImplementedError


class TableDesc:
    def __init__(self, name, schema, sql):
        self.name = name
        self.schema = schema
        self.sql = sql

    def __str__(self):
        return self.name

class ToSql:
    def __init__(self, sql):
        self.sql = sql

    def __call__(self):
        return self.sql
