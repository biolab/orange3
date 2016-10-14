from collections import defaultdict

import numpy as np
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq


def create_sql_contingency(X, columns, m):
    def convert(row):
        c = len(row) - 1
        return [m[columns[i]].get(v) if i != c else v
                for i, v in enumerate(row)]

    group_by = [a.to_sql() for a in (X.domain[c] for c in columns)]
    filters = ['%s IS NOT NULL' % a for a in group_by]
    fields = group_by + ['COUNT(%s)' % group_by[0]]
    query = X._sql_query(fields, group_by=group_by, filters=filters)
    with X._execute_sql_query(query) as cur:
        cont = np.array(list(map(convert, cur.fetchall())), dtype='float')
    return cont[:, :-1], cont[:, -1:].flatten()


def initialize_random(conts, k):
    mu = np.zeros((k, len(conts)))
    sigma = np.zeros((k, len(conts)))
    for i, (c, cw) in enumerate(conts):
        w = np.random.random((len(c), k))
        w /= w.sum(axis=1)[:, None]

        c = c[:, 0] if i == 0 else c[:, 1]

        for j in range(k):
            mu1 = np.dot(w[:, j] * cw, c) / (w[:, j] * cw).sum()
            cn = c - mu1
            sigma1 = np.sum(cn ** 2 * w[:, j] * cw, axis=0) / (w[:, j] * cw).sum()

            mu[j, i] = mu1
            sigma[j, i] = sigma1

    return mu, sigma


def initialize_kmeans(conts, k):
    x = []
    xm = {}
    for i, (c, cw) in enumerate(conts[1:-1]):
        oldx, oldxm, x, xm = x, xm, [], {}
        if i == 0:
            for a, w in zip(c, cw):
                x.append((tuple(a), w))
                xm.setdefault(tuple(a)[1:], []).append(len(x) - 1)
        else:
            for a, w in zip(c, cw):
                for l in oldxm[tuple(a[:2])]:
                    olda, oldw = oldx[l]
                    x.append((olda + (a[2],), oldw+w))
                    xm.setdefault(tuple(a)[1:], []).append(len(x) - 1)

    X = np.array([y[0] for y in x])

    import sklearn.cluster as skl_cluster
    kmeans = skl_cluster.KMeans(n_clusters=k)
    Y = kmeans.fit_predict(X)
    means = kmeans.cluster_centers_
    covars = np.zeros((k, len(conts)))
    for j in range(k):
        xn = X[Y == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / len(xn)

    return means, covars


def lac(conts, k, nsteps=30, window_size=1):
    """
    k expected classes,
    m data points,
    each with dim dimensions
    """
    dim = len(conts)

    np.random.seed(42)
    # Initialize parameters
    priors = np.ones(k) / k


    print("Initializing")
    import sys; sys.stdout.flush()
    means, covars = initialize_random(conts, k)
    #means, covars = initialize_kmeans(conts, k)
    print("Done")

    w = [np.empty((k, len(c[0]),)) for c in conts]
    active = np.ones(k, dtype=np.bool)

    for i in range(1, nsteps + 1):
        for l, (c, cw) in enumerate(conts):
            lower = l - window_size if l - window_size >= 0 else None
            upper = l + window_size + 1 if l + window_size + 1 <= dim else None
            dims = slice(lower, upper)
            active_dim = min(l, window_size)

            x = c

            # E step
            for j in range(k):
                if any(np.abs(covars[j, dims]) < 1e-15):
                    active[j] = 0

                if active[j]:
                    det = covars[j, dims].prod()
                    inv_covars = 1. / covars[j, dims]
                    xn = x - means[j, dims]
                    factor = (2.0 * np.pi) ** (x.shape[1]/ 2.0) * det ** 0.5
                    w[l][j] = priors[j] * np.exp(np.sum(xn * inv_covars * xn, axis=1) * -.5) / factor
                else:
                    w[l][j] = 0
            w[l][active] /= w[l][active].sum(axis=0)

            # M step
            n = np.sum(w[l], axis=1)
            priors = n / np.sum(n)
            for j in range(k):
                if n[j]:
                    mu = np.dot(w[l][j, :] * cw, x[:, active_dim]) / (w[l][j, :] * cw).sum()

                    xn = x[:, active_dim] - mu
                    sigma = np.sum(xn ** 2 * w[l][j] * cw, axis=0) / (w[l][j, :] * cw).sum()

                    if np.isnan(mu).any() or np.isnan(sigma).any():
                        return w, means, covars, priors
                else:
                    active[j] = 0
                    mu = 0.
                    sigma = 0.
                means[j, l] = mu
                covars[j, l] = sigma

    # w = np.zeros((k, m))
    # for j in range(k):
    #     if active[j]:
    #         det = covars[j].prod()
    #         inv_covars = 1. / covars[j]
    #         xn = X - means[j]
    #         factor = (2.0 * np.pi) ** (xn.shape[1] / 2.0) * det ** 0.5
    #         w[j] = priors[j] * exp(-.5 * np.sum(xn * inv_covars * xn, axis=1)) / factor
    # w[active] /= w[active].sum(axis=0)

    return w, means, covars, priors


def create_contingencies(X, callback=None):
    window_size = 1
    dim = len(X.domain)

    X_ = Discretize(method=EqualFreq(n=10))(X)
    m = get_bin_centers(X_)

    from Orange.data.sql.table import SqlTable
    if isinstance(X, SqlTable):
        conts = []
        al = len(X.domain)
        if al > 1:
            conts.append(create_sql_contingency(X_, [0, 1], m))
            if callback:
                callback(1, al)
            for a1, a2, a3 in zip(range(al), range(1, al), range(2, al)):
                conts.append(create_sql_contingency(X_, [a1, a2, a3], m))
                if callback:
                    callback(a3, al)
            if al > 2:
                conts.append(create_sql_contingency(X_, [al - 2, al - 1], m))
                if callback:
                    callback(al, al)
    else:
        conts = [defaultdict(float) for i in range(len(X_.domain))]
        for i, r in enumerate(X_):
            if any(np.isnan(r)):
                continue
            row = tuple(m[vi].get(v) for vi, v in enumerate(r))
            for l in range(len(X_.domain)):
                lower = l - window_size if l - window_size >= 0 else None
                upper = l + window_size + 1 if l + window_size + 1 <= dim else None
                dims = slice(lower, upper)

                conts[l][row[dims]] += 1
        conts = [zip(*x.items()) for x in conts]
        conts = [(np.array(c), np.array(cw)) for c, cw in conts]

    # for i, ((c1, cw1), (c2, cw2)) in enumerate(zip(contss, conts)):
    #     a = np.sort(np.hstack((c1, cw1[:, None])), axis=0)
    #     b = np.sort(np.hstack((c2, cw2[:, None])), axis=0)
    #     assert_almost_equal(a, b)

    return conts


def get_bin_centers(X_):
    m = []
    for i, var in enumerate(X_.domain):
        cleaned_values = [tuple(map(str.strip, v.strip('[]()<>=≥').split('-')))
                          for v in var.values]
        try:
            float_values = [[float(v) for v in vals] for vals in cleaned_values]
            bin_centers = {
                i: v[0] if len(v) == 1 else v[0] + (v[1] - v[0])
                for i, v in enumerate(float_values)
                }
        except ValueError:
            bin_centers = {
                i: i
                for i, v in enumerate(cleaned_values)
                }
        m.append(bin_centers)
    return m
