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
    Theoretical and Empirical Analysis of ReliefF and RReliefF,
    MLJ 2003
"""

cimport numpy as np
import numpy as np

from libc.stdlib cimport rand
from libc.math cimport fabs, exp
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.algorithm cimport make_heap, pop_heap
from libcpp.map cimport map as cpp_map

# Import C99 features from numpy's npy_math (MSVC 2010)
# Note we cannot import isnan due to mixing C++ and C
# (at least on OSX the <cmath> undefines the isnan macro)
from numpy.math cimport INFINITY

ctypedef np.float64_t   double
ctypedef np.int8_t[:]   arr_i1_t
ctypedef double[:, :]   arr_f2_t
ctypedef double[:]      arr_f1_t
ctypedef pair[double, Py_ssize_t] HeapPair
ctypedef cpp_map[Py_ssize_t, arr_f2_t] Contingencies


cdef inline bint isnan(double x) nogil:
    return x != x


cdef inline double nansum(arr_f1_t A) nogil:
    """NaN-skipping sum() for 1D-arrays"""
    cdef:
        double ai, sum = 0
        Py_ssize_t i
    for i in range(A.shape[0]):
        ai = A[i]
        if not isnan(ai):
            sum += ai
    return sum


cdef inline double nanmax(arr_f1_t A) nogil:
    """NaN-skipping max() for 1D-arrays"""
    cdef:
        double ai, max = -INFINITY
        Py_ssize_t i
    for i in range(A.shape[0]):
        ai = A[i]
        if not isnan(ai) and ai > max:
            max = ai
    return max


cdef inline Py_ssize_t randint(Py_ssize_t max) nogil:
    """ TODO: use better random generator"""
    return rand() % max


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
                                 Contingencies &contingencies,
                                 arr_f1_t difference) nogil:
    """Calculate difference between two instance vectors."""
    cdef:
        double val, xi, xj
        Py_ssize_t a, xv
        arr_f2_t cont
    for a in range(X.shape[1]):
        val = fabs(X[i, a] - X[j, a])
        # Differences in discrete attributes can be either 0 or 1
        if is_discrete[a] and val > 0:
            val = 1
        if isnan(val):
            # Replace missing values with their conditional probabilities
            xi, xj = X[i, a], X[j, a]
            if is_discrete[a]:
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


cdef void k_nearest_reg(arr_f2_t X,
                        arr_f1_t y,
                        Py_ssize_t i,
                        Py_ssize_t k_nearest,
                        arr_i1_t is_discrete,
                        arr_f2_t attr_stats,
                        Contingencies &contingencies,
                        arr_f1_t difference,
                        double * Nc,
                        arr_f1_t Na,
                        arr_f1_t Nca,
                        arr_f1_t dist) nogil:
    """The k-nearest search for RReliefF."""
    cdef:
        Py_ssize_t j, a, _
        # The heap that gets "sorted"
        vector[HeapPair] nearest = vector[HeapPair]()
        # Vector of only k nearest as they are needed all at once
        vector[HeapPair] knearest = vector[HeapPair]()
        double distsum = 0, cls_diff, d
    nearest.reserve(X.shape[0])
    knearest.reserve(k_nearest)
    for j in range(X.shape[0]):
        # Calculate difference between i-th and j-th instance
        calc_difference(X, y, i, j, is_discrete, attr_stats, contingencies, difference)
        # Map the manhattan distance to the instance
        nearest.push_back(HeapPair(-nansum(difference), j))
    # Heapify the nearest vectors and extract the k nearest neighbors
    make_heap(nearest.begin(), nearest.end())
    # Pop the i-th instance, "distance to self"
    pop_heap(nearest.begin(), nearest.end())
    nearest.pop_back()
    for _ in range(k_nearest):
        knearest.push_back(nearest.front())
        pop_heap(nearest.begin(), nearest.end())
        nearest.pop_back()
    # Instance influence from ibid. §3.3, eq. 37
    for j in range(k_nearest):
        d = knearest[j].first
        dist[j] = 1 / (d * d)
        distsum += dist[j]
    for j in range(k_nearest):
        dist[j] /= distsum
    # Update the counts
    for j in range(k_nearest):
        cls_diff = fabs(y[i] - y[j])
        Nc[0] += cls_diff * dist[j]
        # Recalculate the distance that was thrown away before
        calc_difference(X, y, i, knearest[j].second, is_discrete, attr_stats, contingencies, difference)
        for a in range(X.shape[1]):
            Na[a] += difference[a] * dist[j]
            Nca[a] += cls_diff * difference[a] * dist[j]


cdef void k_nearest_per_class(arr_f2_t X,
                              arr_f1_t y,
                              Py_ssize_t i,
                              Py_ssize_t k_nearest,
                              Py_ssize_t n_classes,
                              arr_i1_t is_discrete,
                              arr_f2_t attr_stats,
                              Contingencies &contingencies,
                              arr_f2_t weights_adj,
                              arr_f1_t difference) nogil:
    """The k-nearest search for ReliefF."""
    cdef:
        Py_ssize_t j, a, cls, _, yi = int(y[i])
        HeapPair hp
        vector[vector[HeapPair]] nearest = vector[vector[HeapPair]](n_classes)
    for j in range(X.shape[0]):
        # Calculate difference between i-th and j-th instance
        calc_difference(X, y, i, j, is_discrete, attr_stats, contingencies, difference)
        # Map the manhattan distance to the instance
        nearest[<Py_ssize_t>y[j]].push_back(HeapPair(-nansum(difference), j))
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
            calc_difference(X, y, i, hp.second, is_discrete, attr_stats, contingencies, difference)
            # Adjust the weights of the class
            for a in range(X.shape[1]):
                weights_adj[cls, a] += difference[a]



cdef arr_f1_t _relieff_reg_(arr_f2_t X,
                            arr_f1_t y,
                            int n_iter,
                            int k_nearest,
                            arr_i1_t is_discrete,
                            arr_f2_t attr_stats,
                            Contingencies &contingencies):
    """
    The main loop of the RReliefF for regression (ibid. §2.3, Figure 3).
    """
    cdef:
        Py_ssize_t i, a, _
        double Nc = 0
        arr_f1_t Na = np.zeros(X.shape[1])
        arr_f1_t Nca = np.zeros(X.shape[1])
        # Influence (inverse distance) of each neighbor
        arr_f1_t dist = np.empty(k_nearest)

        arr_f1_t weights = np.zeros(X.shape[1])
        arr_f1_t difference = np.empty(X.shape[1])
    with nogil:
        k_nearest = min(k_nearest, X.shape[0] - 1)
        # TODO: stratify per class value?
        for _ in range(n_iter):
            # Select a random instance
            i = randint(X.shape[0])
            # Put the weight adjustments k-nearest-of-each-class make into weights_adj
            k_nearest_reg(X, y, i, k_nearest,
                          is_discrete, attr_stats, contingencies, difference,
                          &Nc, Na, Nca, dist)
        # Update weights
        for a in range(X.shape[1]):
            weights[a] = Nca[a] / Nc - (Na[a] - Nca[a]) / (n_iter - Nc)
    return weights


cdef arr_f1_t _relieff_cls_(arr_f2_t X,
                            arr_f1_t y,
                            int n_classes,
                            int n_iter,
                            int k_nearest,
                            arr_i1_t is_discrete,
                            arr_f1_t prior_proba,
                            arr_f2_t attr_stats,
                            Contingencies &contingencies):
    """
    The main loop of the ReliefF for classification (ibid. §2.1, Figure 2).
    """
    cdef:
        double p
        Py_ssize_t cls, a, i, _, yi
        arr_f1_t weights = np.zeros(X.shape[1])
        arr_f2_t weights_adj = np.empty((n_classes, X.shape[1]))
        arr_f1_t difference = np.empty(X.shape[1])
    with nogil:
        k_nearest = min(k_nearest, X.shape[0] - 1)
        # TODO: stratify per class value?
        for _ in range(n_iter):
            # Clear weight adjustment buffer
            weights_adj[:, :] = 0
            # Select a random instance
            i = randint(X.shape[0])
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


cdef inline void contingency_table(np.ndarray x1,
                                    int n_unique1,
                                    np.ndarray x2,
                                    int n_unique2,
                                    Contingencies &tables,
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
    tables.insert((attribute, table))


cdef void contingency_tables(np.ndarray X,
                             np.ndarray y,
                             arr_i1_t is_discrete,
                             Contingencies &tables):
    """
    Populate contingency tables between attributes of `X` and class values `y`.
    """
    cdef:
        Py_ssize_t a, ny = int(nanmax(y) + 1)
    for a in range(X.shape[1]):
        if (is_discrete[a] and
            # Don't calculate+store contingencies if not required
            np.isnan(X[:, a]).any()):
            contingency_table(X[:, a], int(nanmax(X[:, a]) + 1),
                              y, ny, tables, a)


cdef tuple prepare(X, y, is_discrete, Contingencies &contingencies):
    X = np.array(X, dtype=np.float64, order='C')
    is_discrete = np.asarray(is_discrete, dtype=np.bool8)
    is_continuous = ~is_discrete
    if is_continuous.any():
        row_ptp = np.nanmax(X[:, is_continuous], 0) - np.nanmin(X[:, is_continuous], 0)
        row_ptp[row_ptp == 0] = np.inf  # Avoid zero-division
        X[:, is_continuous] /= row_ptp
    y = np.array(y, dtype=np.float64)
    attr_stats = np.row_stack((np.nanmean(X, 0), np.nanstd(X, 0)))
    is_discrete = np.asarray(is_discrete, dtype=np.int8)
    contingency_tables(X, y, is_discrete, contingencies)
    return X, y, attr_stats, is_discrete


cpdef arr_f1_t relieff(np.ndarray X,
                       np.ndarray y,
                       Py_ssize_t n_iter,
                       Py_ssize_t k_nearest,
                       np.ndarray is_discrete):
    """
    Score attributes of `X` according to ReliefF and return their weights.
    """
    cdef:
        Contingencies contingencies = Contingencies()
    X, y, attr_stats, is_discrete = prepare(X, y, is_discrete, contingencies)
    prior_proba = np.bincount(y.astype(int)).astype(np.float64) / len(y)
    n_classes = int(nanmax(y) + 1)
    return _relieff_cls_(X, y, n_classes, n_iter, k_nearest,
                         is_discrete, prior_proba, attr_stats, contingencies)


cpdef arr_f1_t rrelieff(np.ndarray X,
                        np.ndarray y,
                        Py_ssize_t n_iter,
                        Py_ssize_t k_nearest,
                        np.ndarray is_discrete):
    """
    Score attributes of `X` according to RReliefF and return their weights.
    """
    cdef:
        Contingencies contingencies = Contingencies()
    X, y, attr_stats, is_discrete = prepare(X, y, is_discrete, contingencies)
    return _relieff_reg_(X, y, n_iter, k_nearest,
                         is_discrete, attr_stats, contingencies)
