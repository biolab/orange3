#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: embedsignature=True
#cython: infer_types=False
#cython: language_level=3

"""
ReliefF and RReliefF feature scoring algorithms from:

    Robnik-Šikonja, M., Kononenko, I.,
    Theoretical and Empirical Analysis of ReliefF and RReliefF. MLJ. 2003.
    http://lkm.fri.uni-lj.si/rmarko/papers/robnik03-mlj.pdf
"""

cimport numpy as np
import numpy as np

from libc.math cimport fabs, exp
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.algorithm cimport make_heap, pop_heap

# Import C99 features from numpy's npy_math (MSVC 2010)
# Note we cannot import isnan due to mixing C++ and C
# (at least on OSX the <cmath> undefines the isnan macro)
from numpy.math cimport INFINITY, NAN

ctypedef np.float64_t   double
ctypedef np.int8_t[:]   arr_i1_t
ctypedef np.intp_t[:]   arr_intp_t
ctypedef double[:, :]   arr_f2_t
ctypedef double[:]      arr_f1_t
ctypedef pair[double, Py_ssize_t] HeapPair


cdef inline bint isnan(double x) nogil:
    return x != x


cdef inline double nanmax(arr_f1_t A) nogil:
    """NaN-skipping max() for 1D-arrays"""
    cdef:
        double ai, max = -INFINITY
        Py_ssize_t i
    for i in range(A.shape[0]):
        ai = A[i]
        if not isnan(ai) and ai > max:
            max = ai
    if max == -INFINITY:
        return NAN
    return max


cdef inline double norm_pdf(double x, double mean, double std) nogil:
    """Normal PDF, based on scipy.stats.norm.pdf() """
    x = (x - mean) / std
    return exp(-x**2/2.) / 2.5066282746310002 / std


cdef inline void calc_difference(arr_f2_t X,
                                 arr_f1_t y,
                                 Py_ssize_t i,
                                 Py_ssize_t j,
                                 arr_i1_t is_discrete,
                                 arr_f2_t attr_stats,
                                 contingencies,
                                 arr_f1_t difference,
                                 double * difference_sum) nogil:
    """Calculate difference between two instance vectors."""
    cdef:
        double val, xi, xj
        Py_ssize_t a, xv
        arr_f2_t cont
    difference_sum[0] = 0
    for a in range(X.shape[1]):
        val = fabs(X[i, a] - X[j, a])
        # Differences in discrete attributes can be either 0 or 1
        if is_discrete[a] and val > 0:
            val = 1
        if isnan(val):
            # Replace missing values with their conditional probabilities
            xi, xj = X[i, a], X[j, a]
            if is_discrete[a]:
                with gil:
                    cont = contingencies[a]

                # TODO: what if the attribute only has a single non-nan value?
                if isnan(xi) and isnan(xj):
                    # ibid. §2.2, eq. 4
                    val = 0
                    for xv in range(cont.shape[0]):
                        val += cont[xv, <Py_ssize_t>y[i]] * cont[xv, <Py_ssize_t>y[j]]
                # ibid. §2.2, eq. 3
                elif isnan(xi):
                    val = cont[<Py_ssize_t>xj, <Py_ssize_t>y[j]]
                else:
                    val = cont[<Py_ssize_t>xi, <Py_ssize_t>y[i]]
            else:
                # Assuming the continuous attributes are normally
                # distributed, if any of the two instances are nan,
                # the difference in that attribute equals to normal
                # probability density.
                # If both nan, the value equals to two std deviations.
                # All this, devised by Janez Demšar, is omitted from the
                # original ReliefF algorithm.
                if isnan(xi) and isnan(xj):
                    val = 2*attr_stats[1, a]
                elif isnan(xi):
                    val = norm_pdf(xj, attr_stats[0, a], attr_stats[1, a])
                else:
                    val = norm_pdf(xi, attr_stats[0, a], attr_stats[1, a])
        difference[a] = val
        difference_sum[0] += val


cdef void k_nearest_reg(arr_f2_t X,
                        arr_f1_t y,
                        Py_ssize_t i,
                        Py_ssize_t k_nearest,
                        arr_i1_t is_discrete,
                        arr_f2_t attr_stats,
                        contingencies,
                        arr_f1_t difference,
                        double * Nc,
                        arr_f1_t Na,
                        arr_f1_t Nca) nogil:
    """The k-nearest search for RReliefF."""
    cdef:
        Py_ssize_t j, a, _
        # The heap that gets "sorted"
        vector[HeapPair] nearest = vector[HeapPair]()
        double cls_diff, difference_sum = 0

        # Pleasure yourself with the following mystery:
        #
        # Instance influence (ibid. §2.3, eq. 10, eq. 11), the notion that
        # nearer instances should exert greater effect on the outcome weights,
        # was first set to 1/(d*d), d being the manhattan distance between
        # i-th and j-th instances (ibid. §3.3, eq. 37). With it, RReliefF
        # worked well with a simple XOR dataset (class = A1 > .5 ^ A2 < .5),
        # but didn't work at all with common UCI regression datasets.
        # Setting influence to proposed alternatives (ibid., eq. 10, eq. 11)
        # didn't work, and neither did setting it to 1/k_nearest (all
        # instances have the same, constant influence), which is what Orange2
        # uses.
        # The following, however, does work, for reasons unknown. The constant
        # denominator is arbitrary, but must be, for k_nearest=50, > ~5.
        # Yes; 5 or 1e9 work equally well!

        # Influence of each nearest neighbor
        double influence = 1. / k_nearest / 5
        # The downside is that simple XOR dataset is no longer as precise, i.e.
        # even random, insignificant features get a positive score (as opposed
        # to ~0). The order of feature importances is preserved, though.
    nearest.reserve(X.shape[0])
    for j in range(X.shape[0]):
        # Calculate difference between i-th and j-th instance
        calc_difference(X, y, i, j, is_discrete, attr_stats, contingencies, difference, &difference_sum)
        # Map the manhattan distance to the instance
        nearest.push_back(HeapPair(-difference_sum, j))
    # Heapify the nearest vectors and extract the k nearest neighbors
    make_heap(nearest.begin(), nearest.end())
    # Update the counts
    for _ in range(k_nearest):
        # Pop the i-th instance, "distance to self", in first iteration,
        # then follow up with as many nearest instances as needed (k), in order
        pop_heap(nearest.begin(), nearest.end())
        nearest.pop_back()

        j = nearest.front().second
        cls_diff = fabs(y[i] - y[j])
        Nc[0] += cls_diff * influence
        # Recalculate the distance that was thrown away before
        calc_difference(X, y, i, j, is_discrete, attr_stats, contingencies, difference, &difference_sum)
        for a in range(X.shape[1]):
            Na[a] += difference[a] * influence
            Nca[a] += cls_diff * difference[a] * influence


cdef void k_nearest_per_class(arr_f2_t X,
                              arr_f1_t y,
                              Py_ssize_t i,
                              Py_ssize_t k_nearest,
                              Py_ssize_t n_classes,
                              arr_i1_t is_discrete,
                              arr_f2_t attr_stats,
                              contingencies,
                              arr_f2_t weights_adj,
                              arr_f1_t difference) nogil:
    """The k-nearest search for ReliefF."""
    cdef:
        Py_ssize_t j, a, cls, _, yi = int(y[i])
        HeapPair hp
        vector[vector[HeapPair]] nearest = vector[vector[HeapPair]](n_classes)
        double difference_sum = 0
    for j in range(X.shape[0]):
        # Calculate difference between i-th and j-th instance
        calc_difference(X, y, i, j, is_discrete, attr_stats, contingencies, difference, &difference_sum)
        # Map the manhattan distance to the instance
        nearest[<Py_ssize_t>y[j]].push_back(HeapPair(-difference_sum, j))
    # Heapify the nearest vectors and extract the k nearest neighbors
    for cls in range(n_classes):
        make_heap(nearest[cls].begin(), nearest[cls].end())
    # First, pop the i-th instance, "distance to self"
    pop_heap(nearest[yi].begin(), nearest[yi].end())
    nearest[yi].pop_back()

    for cls in range(n_classes):
        for _ in range(min(k_nearest, <Py_ssize_t>nearest[cls].size())):
            hp = nearest[cls].front()
            pop_heap(nearest[cls].begin(), nearest[cls].end())
            nearest[cls].pop_back()
            # Recalculate the distance that was thrown away before.
            calc_difference(X, y, i, hp.second, is_discrete, attr_stats, contingencies, difference, &difference_sum)
            # Adjust the weights of the class
            for a in range(X.shape[1]):
                weights_adj[cls, a] += difference[a]



cdef arr_f1_t _relieff_reg_(arr_f2_t X,
                            arr_f1_t y,
                            arr_intp_t R,
                            int k_nearest,
                            arr_i1_t is_discrete,
                            arr_f2_t attr_stats,
                            contingencies):
    """
    The main loop of the RReliefF for regression (ibid. §2.3, Figure 3).
    """
    cdef:
        Py_ssize_t i, a, ri
        double Nc = 0
        arr_f1_t Na = np.zeros(X.shape[1])
        arr_f1_t Nca = np.zeros(X.shape[1])

        arr_f1_t weights = np.empty(X.shape[1])
        arr_f1_t difference = np.empty(X.shape[1])
        Py_ssize_t n_iter = R.shape[0]

    with nogil:
        k_nearest = min(k_nearest, X.shape[0] - 1)
        for ri in range(n_iter):
            # Select a random instance
            i = R[ri]
            # Find its k nearest neighbors and update the Nx counts
            k_nearest_reg(X, y, i, k_nearest,
                          is_discrete, attr_stats, contingencies, difference,
                          &Nc, Na, Nca)
        # Update weights
        for a in range(X.shape[1]):
            weights[a] = Nca[a] / Nc - (Na[a] - Nca[a]) / (n_iter - Nc)
    return weights


cdef arr_f1_t _relieff_cls_(arr_f2_t X,
                            arr_f1_t y,
                            arr_intp_t R,
                            int n_classes,
                            int k_nearest,
                            arr_i1_t is_discrete,
                            arr_f1_t prior_proba,
                            arr_f2_t attr_stats,
                            contingencies):
    """
    The main loop of the ReliefF for classification (ibid. §2.1, Figure 2).
    """
    cdef:
        double p
        Py_ssize_t cls, a, i, yi, ri
        arr_f1_t weights = np.zeros(X.shape[1])
        arr_f2_t weights_adj = np.empty((n_classes, X.shape[1]))
        arr_f1_t difference = np.empty(X.shape[1])
        Py_ssize_t n_iter = len(R)
    with nogil:
        k_nearest = min(k_nearest, X.shape[0] - 1)
        # TODO: stratify per class value?
        for ri in range(n_iter):
            # Clear weight adjustment buffer
            weights_adj[:, :] = 0
            # Select a random instance
            i = R[ri]
            # Put the weight adjustments k-nearest-of-each-class make into weights_adj
            k_nearest_per_class(X, y, i, k_nearest, n_classes,
                                is_discrete, attr_stats, contingencies, weights_adj, difference)
            # Update the weights for each class
            yi = <Py_ssize_t>y[i]
            for a in range(X.shape[1]):
                weights[a] -= weights_adj[yi, a]
            for cls in range(n_classes):
                if cls == yi: continue
                p = prior_proba[cls] / (1 - prior_proba[yi])
                for a in range(X.shape[1]):
                    weights[a] += p * weights_adj[cls, a]
        for a in range(X.shape[1]):
            weights[a] /= (n_iter * k_nearest)
    return weights


cdef inline void _contingency_table(np.ndarray x1,
                                    int n_unique1,
                                    np.ndarray x2,
                                    int n_unique2,
                                    tables,
                                    Py_ssize_t attribute):
    cdef:
        np.ndarray table = np.zeros((n_unique1, n_unique2))
        np.ndarray row_sums
        double x1i, x2i
    for i in range(x1.shape[0]):
        x1i, x2i = x1[i], x2[i]
        if isnan(x1i) and isnan(x2i): pass
        elif isnan(x1i): table[:, <Py_ssize_t>x2i] += 1
        elif isnan(x2i): table[<Py_ssize_t>x1i, :] += 1
        else: table[<Py_ssize_t>x1i, <Py_ssize_t>x2i] += 1
    row_sums = table.sum(0)
    row_sums[row_sums == 0] = np.inf  # Avoid zero-division
    table /= row_sums
    tables[attribute] = table


def contingency_table(x1, x2):
    """Return contingency array between x1 and x2."""
    cdef:
        tables = {}
        arr_f2_t table
        int n1 = int(nanmax(x1) + 1), n2 = int(nanmax(x2) + 1)
    if isnan(n1) or isnan(n2):
        return np.array([])
    _contingency_table(x1, n1, x2, n2, tables, 0)
    table = tables[0]
    return np.asarray(table)


cdef void contingency_tables(np.ndarray X,
                             np.ndarray y,
                             arr_i1_t is_discrete,
                             tables):
    """
    Populate contingency tables between attributes of `X` and class values `y`.
    """
    cdef:
        Py_ssize_t a, ny = int(nanmax(y) + 1)
    for a in range(X.shape[1]):
        if (is_discrete[a] and
            # Don't calculate+store contingencies if not required
            np.isnan(X[:, a]).any()):
            _contingency_table(X[:, a], int(nanmax(X[:, a]) + 1),
                              y, ny, tables, a)


cdef tuple prepare(X, y, is_discrete, contingencies):
    X = np.array(X, dtype=np.float64, order='C')
    is_discrete = np.asarray(is_discrete, dtype=np.bool8)
    is_continuous = ~is_discrete
    if is_continuous.any():
        row_min = np.nanmin(X, 0)
        row_ptp = np.nanmax(X, 0) - row_min
        row_ptp[row_ptp == 0] = np.inf  # Avoid zero-division
        X[:, is_continuous] -= row_min[is_continuous]
        X[:, is_continuous] /= row_ptp[is_continuous]
    if y.ndim > 1:
        if y.shape[1] > 1:
            raise ValueError("ReliefF expects a single class")
        y = np.array(y[:, 0], dtype=np.float64)
    else:
        y = np.array(y, dtype=np.float64)
    is_defined = np.logical_not(np.isnan(y))
    X = X[is_defined]
    y = y[is_defined]
    attr_stats = np.row_stack((np.nanmean(X, 0), np.nanstd(X, 0)))
    is_discrete = np.asarray(is_discrete, dtype=np.int8)
    contingency_tables(X, y, is_discrete, contingencies)
    return X, y, attr_stats, is_discrete


cpdef arr_f1_t relieff(np.ndarray X,
                       np.ndarray y,
                       Py_ssize_t n_iter,
                       Py_ssize_t k_nearest,
                       np.ndarray is_discrete,
                       rstate):
    """
    Score attributes of `X` according to ReliefF and return their weights.
    """
    cdef:
        contingencies = {}
    if not isinstance(rstate, np.random.RandomState):
        raise TypeError('rstate')
    cdef:
        arr_intp_t R = rstate.randint(X.shape[0], size=n_iter, dtype=np.intp)

    X, y, attr_stats, is_discrete = prepare(X, y, is_discrete, contingencies)
    prior_proba = np.bincount(y.astype(int)).astype(np.float64) / len(y)
    n_classes = int(nanmax(y) + 1)
    return _relieff_cls_(X, y, R, n_classes, k_nearest,
                         is_discrete, prior_proba, attr_stats, contingencies)


cpdef arr_f1_t rrelieff(np.ndarray X,
                        np.ndarray y,
                        Py_ssize_t n_iter,
                        Py_ssize_t k_nearest,
                        np.ndarray is_discrete,
                        rstate):
    """
    Score attributes of `X` according to RReliefF and return their weights.
    """
    cdef:
        contingencies = {}
    if not isinstance(rstate, np.random.RandomState):
        raise TypeError('rstate')

    cdef:
        arr_intp_t R = rstate.randint(X.shape[0], size=n_iter, dtype=np.intp)
    X, y, attr_stats, is_discrete = prepare(X, y, is_discrete, contingencies)
    y = (y - np.min(y)) / np.ptp(y)
    return _relieff_reg_(X, y, R, k_nearest,
                         is_discrete, attr_stats, contingencies)
