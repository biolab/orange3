import logging
from contextlib import contextmanager

from Orange.util import Registry

log = logging.getLogger(__name__)


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
        raise NotImplementedError

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
        raise NotImplementedError

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
