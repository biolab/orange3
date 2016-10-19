from contextlib import contextmanager


class Backend:
    """Base class for SqlTable backends. Implementations should define
    all of the methods defined below.

    Parameters
    ----------
    connection_params: dict
        connection params
    """

    def __init__(self, connection_params):
        self.connection_params = connection_params

    # "meta" methods

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


class ToSql:
    def __init__(self, sql):
        self.sql = sql

    def __call__(self):
        return self.sql
