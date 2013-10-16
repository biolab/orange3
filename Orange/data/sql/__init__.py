import Orange
import psycopg2

def create_stored_procedure(uri=None, host=None, user=None, password=None, database=None, schema=None,
                            **kwargs):
    assert uri is not None

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
        CREATE OR REPLACE FUNCTION EqualWidth(data text, attribute text, filters text[], n integer)
          RETURNS SETOF double precision
        AS $$
          # get_distribution
          query = [ 'SELECT "{0}" AS value, COUNT("{0}") FROM "{1}"'.format(attribute, data) ]
          if filters is not None:
            query.extend([ 'WHERE {0}'.format(filters[0]) ])
          query.extend([ 'GROUP BY "{0}" ORDER BY "{0}"'.format(attribute) ])
          query = " ".join(query)
          dist = plpy.execute(query)

          if dist.nrows() == 0:
             return []

          # split_equal_width
          min = dist[0]["value"]
          max = dist[-1]["value"]
          dif = (max-min)/n

          return [ min + (i+1)*dif for i in range(n-1) ]
        $$ LANGUAGE plpythonu;
    """)
    connection.commit()

    cur = connection.cursor()
    cur.execute("""
        CREATE OR REPLACE FUNCTION EqualFreq(data text, attribute text, filters text[], n integer)
          RETURNS SETOF double precision
        AS $$
          # get_distribution
          query = [ 'SELECT "{0}" AS value, COUNT("{0}") FROM "{1}"'.format(attribute, data) ]
          if filters is not None:
            query.extend([ 'WHERE {0}'.format(filters[0]) ])
          query.extend([ 'GROUP BY "{0}" ORDER BY "{0}"'.format(attribute) ])
          query = " ".join(query)
          dist = plpy.execute(query)

          if dist.nrows() == 0:
            return []

          # split_equal_freq
          llen = len(dist)

          if n >= llen:
            return [ (v1["value"]+v2["value"])/2 for v1, v2 in zip(dist[0:], dist[1:]) ]

          counts, values = [], []
          for d in dist:
            counts.append(d["count"])
            values.append(d["value"])

          N = sum(counts)
          toGo = n
          inthis = 0
          prevel = -1
          inone = N/float(toGo)
          points = []

          for i in range(llen):
            v = values[i]
            k = counts[i]
            if toGo <= 1:
              break
            inthis += k
            if inthis < inone or i == 0:
              prevel = v
            else:
              if i < llen -1 and inthis - inone < k / 2.0:
                vn = values[i+1]
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