import Orange
import Orange.misc
psycopg2 = Orange.misc.import_late_warning("psycopg2")


def create_stored_procedures(uri=None, host=None, user=None, password=None, database=None, schema=None,
                            **kwargs):
    connection_args = dict(
        host=host,
        user=user,
        password=password,
        database=database,
        schema=schema
    )
    if uri is not None:
        parameters = Orange.data.sql.SqlTable.parse_uri(uri)
        connection_args.update(parameters)
    connection_args.update(kwargs)

    connection = psycopg2.connect(**connection_args)
    cur = connection.cursor()
    cur.execute("""
            DROP TYPE IF EXISTS distribution;
            CREATE TYPE distribution AS (
              value double precision,
              count bigint
            );
    """)
    connection.commit()

    cur = connection.cursor()
    cur.execute("CREATE LANGUAGE plpythonu;")
    connection.commit()

    cur.execute("""
            CREATE OR REPLACE FUNCTION get_distribution(data text, attribute text, filters text[])
              RETURNS SETOF distribution
            AS $$
              query = [ 'SELECT "{0}" AS value, COUNT("{0}") FROM "{1}"'.format(attribute, data) ]
              if filters is not None:
                query.extend([ 'WHERE {0}'.format(filters[0]) ])
              query.extend([ 'GROUP BY "{0}" ORDER BY "{0}"'.format(attribute) ])
              query = " ".join(query)
              dist = plpy.execute(query)

              return dist
            $$ LANGUAGE plpythonu;
    """)
    connection.commit()
