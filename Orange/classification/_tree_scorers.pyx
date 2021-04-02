#cython: embedsignature=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: cdivision=True
#cython: language_level=3


import numpy as np
cimport numpy as np

from libc.math cimport log

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x) nogil

cpdef enum:
    NULL_BRANCH = -1

def contingency(const double[:] x, int nx, const double[:] y, int ny):
    cdef:
        np.ndarray[np.uint32_t, ndim=2] cont = np.zeros((ny, nx), dtype=np.uint32)
        int n = len(x), yi, xi

    for i in range(n):
        if not npy_isnan(x[i]) and not npy_isnan(y[i]):
            yi, xi = int(y[i]), int(x[i])
            cont[yi, xi] += 1
    return cont

def find_threshold_entropy(const double[:] x, const double[:] y,
                           const np.intp_t[:] idx,
                           int n_classes, int min_leaf):
    """
    Find the threshold for continuous attribute values that maximizes
    information gain.

    Argument min_leaf sets the minimal number of data instances on each side
    of the threshold. If there is no threshold within that limits with positive
    information gain, the function returns (0, 0).

    Args:
        x: attribute values
        y: class values
        idx: arg-sorted indices of x (and y)
        n_classes: the number of classes
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        (highest information gain, the corresponding optimal threshold)
    """
    cdef:
        unsigned int[:] distr = np.zeros(2 * n_classes, dtype=np.uint32)
        Py_ssize_t i, j
        double entro, class_entro, best_entro
        unsigned int p, curr_y
        unsigned int best_idx = 0
        unsigned int N = idx.shape[0]

    # Initial split (min_leaf on the left)
    if N <= min_leaf:
        return 0, 0
    with nogil:
        for i in range(min_leaf - 1):  # one will be added in the loop
            distr[n_classes + <int>y[idx[i]]] += 1
        for i in range(min_leaf - 1, N):
            distr[<int>y[idx[i]]] += 1

        # Compute class entropy
        class_entro = N * log(N)
        for j in range(n_classes):
            p = distr[j] + distr[j + n_classes]
            if p:
                class_entro -= p * log(p)
        best_entro = class_entro

        # Loop through
        for i in range(min_leaf - 1, N - min_leaf):
            curr_y = <int>y[idx[i]]
            distr[curr_y] -= 1
            distr[n_classes + curr_y] += 1
            if curr_y != y[idx[i + 1]] and x[idx[i]] != x[idx[i + 1]]:
                entro = (i + 1) * log(i + 1) + (N - i - 1) * log(N - i - 1)
                for j in range(2 * n_classes):
                    if distr[j]:
                        entro -= distr[j] * log(distr[j])
                if entro < best_entro:
                    best_entro = entro
                    best_idx = i
    return (class_entro - best_entro) / N / log(2), x[idx[best_idx]]


def find_binarization_entropy(const double[:, :] cont,
                              const double[:] class_distr,
                              const double[:] val_distr, int min_leaf):
    """
    Find the split of discrete values into two groups that optimizes information
    gain.

    The split is returned as an int in which the lower bits give the group
    membership of the corresponding values; the first value of the attribute
    corresponding to bit zero and so forth.

    Argument min_leaf sets the minimal number of data instances in each group.
    If there is no split (within that limits) with positive information gain,
    the function returns (0, 0).

    The function works by traversing over the 2 ** (n - 1) possible states in
    the order of Gray encoding. With this, the function does not recompute the
    sum of distributions but just moves one distribution at a time to the left
    or to the right.

    Args:
        cont: contingency matrix
        class_distr: marginal class distribution (sum of cont over axis 1)
        val_distr: marginal attribute value distribution (sum of cont over axis 0)
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        (highest information gain, the corresponding optimal mapping)
    """
    cdef:
        unsigned int n_classes = cont.shape[0]
        unsigned int n_values = cont.shape[1]
        double[:] distr = np.zeros(2 * n_classes)
        double[:] mfrom
        double[:] mto
        double left, right
        unsigned int i, change, to_right, allowed, m
        unsigned int best_mapping = 0, move = 0, mapping, previous
        double entro, class_entro, best_entro
        double N = 0

    with nogil:
        class_entro = 0
        for i in range(n_classes):
            distr[i + n_classes] = 0
            distr[i] = class_distr[i]
            if class_distr[i] > 0:
                N += class_distr[i]
                class_entro -= class_distr[i] * log(class_distr[i])
        class_entro += N * log(N)
        best_entro = class_entro
        left = N
        right = 0

        previous = 0
        # Gray code
        for m in range(1, 1 << (n_values - 1)):
            # What moves where
            mapping = m ^ (m >> 1)
            change = mapping ^ previous
            to_right = change & mapping
            for move in range(n_values):
                if change & 1:
                    break
                change = change >> 1
            previous = mapping

            if to_right:
                left -= val_distr[move]
                right += val_distr[move]
                mfrom = distr
                mto = distr[n_classes:]
            else:
                left += val_distr[move]
                right -= val_distr[move]
                mfrom = distr[n_classes:]
                mto = distr

            allowed = left >= min_leaf and right >= min_leaf
            # Move distribution to the other side and
            # compute entropy by the way, if the split is allowed
            entro = 0
            for i in range(n_classes):
                mfrom[i] -= cont[i, move]
                mto[i] += cont[i, move]
                if allowed:
                    if mfrom[i]:
                        entro -= mfrom[i] * log(mfrom[i])
                    if mto[i]:
                        entro -= mto[i] * log(mto[i])

            if allowed:
                entro += left * log(left) + right * log(right)
                if entro < best_entro:
                    best_entro = entro
                    best_mapping = mapping
    return (class_entro - best_entro) / N / log(2), best_mapping


def find_threshold_MSE(const double[:] x,
                       const double[:] y,
                       const np.intp_t[:] idx, int min_leaf):
    """
    Find the threshold for continuous attribute values that minimizes MSE.

    Argument min_leaf sets the minimal number of data instances on each side
    of the threshold. If there is no threshold within that limits that decreases
    the MSE with respect to the prior MSE, the function returns (0, 0).

    Args:
        x: attribute values
        y: target values
        idx: arg sorted indices of x (and y)
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        (largest MSE decrease, the corresponding optimal threshold)
    """
    cdef:
        double sleft = 0, sum, inter, best_inter
        unsigned int i, best_idx = 0
        unsigned int N = idx.shape[0]

    # Initial split (min_leaf on the left)
    if N <= min_leaf:
        return 0, 0
    with nogil:
        sum = 0
        for i in range(min_leaf - 1):  # one will be added in the loop
            sum += y[idx[i]]
        sleft = sum
        for i in range(min_leaf - 1, N):
            sum += y[idx[i]]

        best_inter = (sum * sum) / N
        for i in range(min_leaf - 1, N - min_leaf):
            sleft += y[idx[i]]
            if x[idx[i]] == x[idx[i + 1]]:
                continue
            inter = sleft * sleft / (i + 1) + (sum - sleft) * (sum - sleft) / (N - i - 1)
            if inter > best_inter:
                best_inter = inter
                best_idx = i
    return (best_inter - (sum * sum) / N) / N, x[idx[best_idx]]


def find_binarization_MSE(const double[:] x,
                          const double[:] y, int n_values, int min_leaf):
    """
    Find the split of discrete values into two groups that minimizes the MSE.

    The split is returned as an int in which the lower bits give the group
    membership of the corresponding values; the first value of the attribute
    corresponding to bit zero and so forth.

    The score is decreased in proportion with the number of missing values in x.

    Argument min_leaf sets the minimal number of data instances in each group.
    If there is no split (within that limits) that decreases the average
    MSE with respect to the prior MSE, the function returns (0, 0).

    The function works by traversing over the 2 ** (n - 1) possible states in
    the order of Gray encoding. With this, the function does not recompute the
    sums but just moves one value at a time to the left or to the right.

    Args:
        x: attribute values
        y: target values
        n_values: the number of attribute values
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        (largest MSE decrease, the corresponding optimal mapping)
    """
    cdef:
        double sleft, sum = 0, val
        unsigned int left
        unsigned int i, change, to_right, m
        unsigned int best_mapping = 0, move = 0, mapping, previous
        double inter, best_inter, start_inter
        unsigned int N

        np.int32_t[:] group_sizes = np.zeros(n_values, dtype=np.int32)
        double[:] group_sums = np.zeros(n_values)

    N = 0
    for i in range(x.shape[0]):
        val = x[i]
        if not npy_isnan(val):
            group_sizes[<int>val] += 1
            group_sums[<int>val] += y[i]
            sum += y[i]
            N += 1
    if N == 0:
        return 0, 0
    with nogil:
        left = N
        sleft = sum
        best_inter = start_inter = (sum * sum) / N

        previous = 0
        # Gray code
        for m in range(1, 1 << (n_values - 1)):
            # What moves where
            mapping = m ^ (m >> 1)
            change = mapping ^ previous
            to_right = change & mapping
            for move in range(n_values):
                if change & 1:
                    break
                change = change >> 1
            previous = mapping

            if to_right:
                left -= group_sizes[move]
                sleft -= group_sums[move]
            else:
                left += group_sizes[move]
                sleft += group_sums[move]

            if left >= min_leaf and (N - left) >= min_leaf:
                inter = sleft * sleft / left + (sum - sleft) * (sum - sleft) / (N - left)
                if inter > best_inter:
                    best_inter = inter
                    best_mapping = mapping
        # factor N / x.shape[0] is the punishment for missing values
        # return (best_inter - start_inter) / N * (N / x.shape[0]), best_mapping
    return (best_inter - start_inter) / x.shape[0], best_mapping


def compute_grouped_MSE(const double[:] x,
                        const double[:] y,
                        int n_values, int min_leaf):
    """
    Compute the MSE decrease of the given split into groups.

    Argument min_leaf sets the minimal number of data instances in each group.
    If there are less than two groups with such number of instances, the
    function returns (0, 0).

    The score is decreased in proportion with the number of missing values in x.

    Args:
        x: attribute values
        y: target values
        n_values: the number of attribute values
        min_leaf: the minimal number of instances on each side of the threshold

    Returns:
        MSE decrease
    """

    cdef:
        int i, n
        #: number of valid nodes (having at least `min_leaf` instances)
        int nvalid = 0
        double sum = 0, inter, tx

        np.int32_t[:] group_sizes = np.zeros(n_values, dtype=np.int32)
        double[:] group_sums = np.zeros(n_values)

    with nogil:
        for i in range(x.shape[0]):
            tx = x[i]
            if not npy_isnan(tx):
                group_sizes[<int>tx] += 1
                group_sums[<int>tx] += y[i]
        inter = 0
        n = 0
        for i in range(n_values):
            if group_sizes[i] < min_leaf:
                # We don't construct nodes with less than min_leaf instances
                # If there is only one non-null node, the split will yield a
                # score of 0
                continue
            inter += group_sums[i] * group_sums[i] / group_sizes[i]
            sum += group_sums[i]
            n += group_sizes[i]
            nvalid += 1
    if nvalid < 2:
        # NOTE: the `inter - sum * sum / n` below does not necessarily
        # cancel out
        return 0
    # factor n / x.shape[0] is the punishment for missing values
    #return (inter - sum * sum / n) / n * n / x.shape[0]
    return (inter - sum * sum / n) / x.shape[0]


def compute_predictions(const double[:, :] X,
                        const int[:] code,
                        const double[:, :] values,
                        const double[:] thresholds):
    """
    Return the values (distributions, means and variances) stored in the nodes
    to which the tree classify the rows in X.

    The tree is encoded by :obj:`Orange.tree.OrangeTreeMode._compile`.

    The result is a matrix of shape (X.shape[0], values.shape[1])

    Args:
        X: data for which the predictions are made
        code: encoded tree
        values: values corresponding to tree nodes
        thresholds: thresholds for numeric nodes

    Returns:
        a matrix of values
    """
    cdef:
        unsigned int node_ptr, i, j, val_idx
        signed int next_node_ptr, node_idx
        np.float64_t val
        double[: ,:] predictions = np.empty(
            (X.shape[0], values.shape[1]), dtype=np.float64)

    with nogil:
        for i in range(X.shape[0]):
            node_ptr = 0
            while code[node_ptr]:
                val = X[i, code[node_ptr + 2]]
                if npy_isnan(val):
                    break
                if code[node_ptr] == 3:
                    node_idx = code[node_ptr + 1]
                    val_idx = int(val > thresholds[node_idx])
                else:
                    val_idx = int(val)
                next_node_ptr = code[node_ptr + 3 + val_idx]
                if next_node_ptr == NULL_BRANCH:
                    break
                node_ptr = next_node_ptr
            node_idx = code[node_ptr + 1]
            for j in range(values.shape[1]):
                predictions[i, j] = values[node_idx, j]
    return np.asarray(predictions)


def compute_predictions_csr(X,
                            const int[:] code,
                            const double[:, :] values,
                            const double[:] thresholds):
    """
    Same as compute_predictions except for sparse data
    """
    cdef:
        unsigned int node_ptr, i, j, val_idx
        signed int next_node_ptr, node_idx
        np.float64_t val
        double[: ,:] predictions = np.empty(
            (X.shape[0], values.shape[1]), dtype=np.float64)

        const double[:] data = X.data
        const np.int32_t[:] indptr = X.indptr
        const np.int32_t[:] indices = X.indices
        int ind, attr, n_rows

    n_rows = X.shape[0]

    with nogil:
        for i in range(n_rows):
            node_ptr = 0
            while code[node_ptr]:
                attr = code[node_ptr + 2]
                ind = indptr[i]
                while ind < indptr[i + 1] and indices[ind] != attr:
                    ind += 1
                val = data[ind] if ind < indptr[i + 1] else 0
                if npy_isnan(val):
                    break
                if code[node_ptr] == 3:
                    node_idx = code[node_ptr + 1]
                    val_idx = int(val > thresholds[node_idx])
                else:
                    val_idx = int(val)
                next_node_ptr = code[node_ptr + 3 + val_idx]
                if next_node_ptr == NULL_BRANCH:
                    break
                node_ptr = next_node_ptr
            node_idx = code[node_ptr + 1]
            for j in range(values.shape[1]):
                predictions[i, j] = values[node_idx, j]
    return np.asarray(predictions)

def compute_predictions_csc(X,
                            const int[:] code,
                            const double[:, :] values,
                            const double[:] thresholds):
    """
    Same as compute_predictions except for sparse data
    """
    cdef:
        unsigned int node_ptr, i, j, val_idx
        signed int next_node_ptr, node_idx
        np.float64_t val
        double[: ,:] predictions = np.empty(
            (X.shape[0], values.shape[1]), dtype=np.float64)

        const double[:] data = X.data
        const np.int32_t[:] indptr = X.indptr
        const np.int32_t[:] indices = X.indices
        int ind, attr, n_rows

    n_rows = X.shape[0]

    with nogil:
        for i in range(n_rows):
            node_ptr = 0
            while code[node_ptr]:
                attr = code[node_ptr + 2]
                ind = indptr[attr]
                while ind < indptr[attr + 1] and indices[ind] != i:
                    ind += 1
                val = data[ind] if ind < indptr[attr + 1] else 0
                if npy_isnan(val):
                    break
                if code[node_ptr] == 3:
                    node_idx = code[node_ptr + 1]
                    val_idx = int(val > thresholds[node_idx])
                else:
                    val_idx = int(val)
                next_node_ptr = code[node_ptr + 3 + val_idx]
                if next_node_ptr == NULL_BRANCH:
                    break
                node_ptr = next_node_ptr
            node_idx = code[node_ptr + 1]
            for j in range(values.shape[1]):
                predictions[i, j] = values[node_idx, j]
    return np.asarray(predictions)
