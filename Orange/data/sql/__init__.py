import Orange
import psycopg2

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

    cur = connection.cursor()
    cur.execute("""
            CREATE OR REPLACE FUNCTION EqualFreq(data text, attribute text, filters text[], n integer)
              RETURNS SETOF double precision
            AS $$
              plan = plpy.prepare( "SELECT count, value FROM get_distribution($1, $2, $3)", ["text", "text", "text[]"] )
              dist = plpy.execute( plan, [data, attribute, filters] )

              if len(dist) == 0:
                return []

              llen = len(dist)     # llen = dist.shape[1]

              if n >= llen:
                return [ (v1["value"]+v2["value"])/2 for v1, v2 in zip(dist[0:], dist[1:]) ]

              N = sum([ d["count"] for d in dist ])       # N = dist[1].sum()

              toGo = n
              inthis = 0
              prevel = -1
              inone = N/float(toGo)
              points = []

              for i in range(llen):
                v = dist[i]["value"]
                k = dist[i]["count"]
                if toGo <= 1:
                  break
                inthis += k
                if inthis < inone or i == 0:
                  prevel = v
                else:
                  if i < llen -1 and inthis - inone < k / 2.0:
                    vn = dist[i+1]["value"]
                    points.append((vn + v)/2.0)
                    N -= inthis
                    inthis = 0
                    prevel = vn
                  else:
                    points.append((prevel + v)/2.0)
                    N -= inthis - k
                    inthis = k
                    prevel = v
                  toGo -= 1
                  if toGo:
                    inone = N/float(toGo)

              return points
            $$ LANGUAGE plpythonu;
    """)
    connection.commit()

    cur = connection.cursor()
    cur.execute("""
            CREATE OR REPLACE FUNCTION EqualWidth(data text, attribute text, filters text[], n integer)
              RETURNS SETOF double precision
            AS $$
              # get_distribution
              query = [ 'SELECT min("{0}") AS min, max("{0}") AS max FROM "{1}"'.format(attribute, data) ]
              if filters is not None:
                query.extend([ 'WHERE {0}'.format(filters[0]) ])
              query = " ".join(query)
              dist = plpy.execute(query)

              if dist.nrows() != 1 or dist[0]["min"] == None or dist[0]["max"] == None:
                 return []

              # split_equal_width
              min = dist[0]["min"]         # min = dist[0][0]
              max = dist[0]["max"]         # max = dist[0][-1]
              dif = (max-min)/n

              return [ min + (i+1)*dif for i in range(n-1) ]
            $$ LANGUAGE plpythonu;
    """)
    connection.commit()
