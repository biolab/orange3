import numpy as np
cimport numpy as np
import scipy.sparse as sp
import cython

from libc.stdio cimport *
from libc.string cimport *
from libc.stdlib cimport malloc, free

cdef extern from "stdio.h" nogil:
    int fgetc(FILE *STREAM)
    int ungetc(char c, FILE *STREAM)

cdef enum State:
    BEGIN_LINE, READ_START_ATOM, READ, QUOTED, END_QUOTED, END_ATOM,
    COMMENT, CARRIAGE_RETURNED, ESCAPE, EQUALS, END_LINE,
    SET_VALUE, WAIT_VALUE, READ_VALUE, READ_DECS, TO_NEXT

cpdef sparse_prescan_fast(fname):
    cdef:
        State state
        char c
        int ci
        int n_attributes = 0
        int n_classes = 0
        int n_metas = 0
        int n_lines = 0
        int *output_count

    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))

    state = BEGIN_LINE
    while True:
        ci = fgetc(f)
        if ci == EOF:
            output_count[0] += 1
            break
        c = <char>ci

        if c == "\n":
            state = BEGIN_LINE
            output_count[0] += 1
            continue
        if c == "\r":
            state = CARRIAGE_RETURNED
            continue

        if state == CARRIAGE_RETURNED:
            state = BEGIN_LINE
            output_count[0] += 1
            # read one more if needed, else not
            if c == "\n":
                continue

        if state == BEGIN_LINE:
            n_lines += 1
            output_count = &n_attributes
            state = READ

        if state == QUOTED:
            if c == '"':
                state = READ
                continue

        if state == READ:
            if c == ",":
                output_count[0] += 1
            elif c == '"':
                state = QUOTED
            elif c == "|":
                output_count[0] += 1
                output_count = &n_classes
            elif c == ";":
                output_count[0] += 1
                output_count = &n_metas
            elif c == "#":
                state = COMMENT

    fclose(f)
    return n_attributes, n_classes, n_metas, n_lines


cpdef check_csr_matrix(np.ndarray[np.int32_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] indices, int n_attrs):
    cdef:
        int row, col, j
        char *used = <char *>malloc(n_attrs)

    try:
        for row in range(len(indptr) - 1):
            for j in range(n_attrs):
                used[j] = 0
            for j in range(indptr[row], indptr[row + 1]):
                col = indices[j]
                if used[col]:
                    return row, col
                else:
                    used[col] = 1
        return -1, -1
    finally:
        free(used)


cdef inline void resize_if_needed(np.ndarray a, size):
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    if dim[0] != size:
        a.resize(size, refcheck=False)

cdef enum ColKinds:
    ATTRIBUTE, CLASS, META

@cython.wraparound(False)
def sparse_read_float(fname):
    n_attrs, n_classes, n_metas, n_lines = sparse_prescan_fast(fname)

    cdef:
        State state
        char c
        int ci
        char *not_in_atom = "#,|;\n\r\x00"
        int col, line, cur_line
        int in_line
        char atom[10240]
        char *atomp
        char *atome = atom + 10240
        char *endc
        char f_eof = 0
        int ii
        int attr_index
        int row_err
        float value, decs
        char col_kind

        # n_lines + 2 -- +2 instead of +1 is needed for the empty last line
        # it is removed in the end
        np.ndarray[np.float_t, ndim=1] X_data = np.empty(n_attrs, float)
        np.ndarray[np.int32_t, ndim=1] X_indices = np.empty(n_attrs, np.int32)
        np.ndarray[np.int32_t, ndim=1] X_indptr = np.empty(n_lines + 2, np.int32)

        np.ndarray[np.float_t, ndim=1] Y_data = np.empty(n_classes, float)
        np.ndarray[np.int32_t, ndim=1] Y_indices = np.empty(n_classes, np.int32)
        np.ndarray[np.int32_t, ndim=1] Y_indptr = np.empty(n_lines + 2, np.int32)

        np.ndarray[np.float_t, ndim=1] metas_data = np.empty(n_metas, float)
        np.ndarray[np.int32_t, ndim=1] metas_indices = np.empty(n_metas, np.int32)
        np.ndarray[np.int32_t, ndim=1] metas_indptr = np.empty(n_lines + 2, np.int32)

        dict attr_indices = {}
        dict class_indices = {}
        dict meta_indices = {}

    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))


    X_indptr[0] = Y_indptr[0] = metas_indptr[0] = 0
    line = 0
    cur_line = 0
    in_line = 0

    try:
        state = BEGIN_LINE
        while not (f_eof and state == BEGIN_LINE):
            if state == BEGIN_LINE:
                col_kind = ATTRIBUTE
                if in_line or line == 0:
                    line += 1
                    X_indptr[line] = X_indptr[line - 1]
                    Y_indptr[line] = Y_indptr[line - 1]
                    metas_indptr[line] = metas_indptr[line - 1]
                    t_names = attr_indices
                cur_line += 1
                col = in_line = 0
                state = READ_START_ATOM

            if state != END_LINE and state != SET_VALUE:
                ci = fgetc(f)
                if ci == EOF:
                    f_eof = 1
                    c = "\x00"
                else:
                    c = <char>ci
                col += 1

            if state == READ_START_ATOM:
                atomp = atom
                value = 1
                if c == "," or c == " " or c == "\t":
                    continue
                elif c == '"':
                    state = QUOTED
                    continue
                elif c == "=":
                    raise ValueError("{}:{}:{}: missing value name"
                        .format(fname, cur_line, col))
                # fall through
                state = READ

            if state == ESCAPE:
                if c == "t":    c = "\t"
                elif c == "n":    c = "\n"
                elif c == "r":    c = "\r"
                elif c == '"' or c == "'" or c == "\\" or c == " ":
                    pass
                elif c == "\r" or c == "\n":
                    raise ValueError("{}:{}:{}: end of line in escape sequence"
                        .format(fname, cur_line, col))
                elif f_eof:
                    raise ValueError("{}:{}:{}: end of file in escape sequence"
                        .format(fname, cur_line, col))
                else:
                    raise ValueError("{}:{}:{}: unrecognized escape sequence"
                        .format(fname, cur_line, col))
                # fall through
                state = READ

            if state == READ:
                if c == "\\":
                    state = ESCAPE
                    continue
                endc = strchr(not_in_atom, c)
                if endc == NULL and c != "=":
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, cur_line, col))
                    continue
                else:
                    # fall through to END_ATOM
                    state = END_ATOM

            if state == QUOTED:
                if c == "\r" or c == "\n":
                    raise ValueError(
                        "{}:{}:{}: end of line within a quoted value"
                        .format(fname, cur_line, col))
                elif f_eof:
                    raise ValueError(
                        "{}:{}:{}: end of file within a quoted value"
                        .format(fname, cur_line, col))
                elif c != '"':
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, cur_line, col))
                else:
                    state = END_QUOTED
                    continue

            if state == END_QUOTED:
                if c == " " or c == "\t":
                    continue
                endc = strchr(not_in_atom, c)
                if endc == NULL and c != "=":
                    raise ValueError("{}:{}:{}: quoted value should be "
                        "followed by value separator or end of line"
                        .format(fname, cur_line, col))
                # fall through
                state = END_ATOM

            if state == END_ATOM:
                while atomp != atom and (
                        atomp[-1] == " " or atomp[-1] == "\t"):
                    atomp -= 1
                if atomp == atom:
                    if c == "=":
                        raise ValueError("{}:{}:{}: empty value name"
                            .format(fname, cur_line, col))
                    else:
                        state = TO_NEXT
                else:
                    atomp[0] = 0
                    b_atom = atom
                    """
                    attr_index = t_names.get(b_atom, -1)
                    if attr_index < 0:
                        attr_index = t_names[b_atom] = len(t_names)
                    """
                    attr_index = t_names.setdefault(b_atom,len(t_names))

                    atomp = atom
                    if c == "=":
                        value = 0
                        state = WAIT_VALUE
                        continue
                    # fall through
                    state = SET_VALUE

            if state == WAIT_VALUE:
                if c == " " or c == "\t":
                    continue
                else:
                    # fall through
                    state = READ_VALUE

            if state == READ_VALUE:
                if "0" <= c <= "9":
                    value = value * 10 + (c & 0xf)
                elif c == ".":
                    decs = 0.1
                    state = READ_DECS
                else:
                    endc = strchr(not_in_atom, c)
                    if endc != NULL:
                        state = SET_VALUE
                    else:
                        raise ValueError("{}:{}:{}: invalid value"
                            .format(fname, cur_line, col))
                continue

            if state == READ_DECS:
                if "0" <= c <= "9":
                    value = value * decs + (c & 0xf)
                    decs /= 10
                else:
                    endc = strchr(not_in_atom, c)
                    if endc != NULL:
                        state = SET_VALUE
                    else:
                        raise ValueError("{}:{}:{}: invalid value"
                            .format(fname, cur_line, col))
                continue

            if state == SET_VALUE:
                if col_kind == ATTRIBUTE:
                    ii = X_indptr[line]
                    X_data[ii] = value
                    X_indices[ii] = attr_index
                    X_indptr[line] = X_indptr[line] + 1
                elif col_kind == CLASS:
                    ii = Y_indptr[line]
                    Y_data[ii] = value
                    Y_indices[ii] = attr_index
                    Y_indptr[line] = Y_indptr[line] + 1
                else:
                    ii = metas_indptr[line]
                    metas_data[ii] = value
                    metas_indices[ii] = attr_index
                    metas_indptr[line] = metas_indptr[line] + 1
                in_line += 1
                state = TO_NEXT

            if state == TO_NEXT:
                if c == "|":
                    if col_kind != ATTRIBUTE:
                        raise ValueError(
                            "{}:{}:{}: classes should follow attributes"
                            .format(fname, cur_line, col))
                    col_kind = CLASS
                    t_names = class_indices
                    state = READ_START_ATOM
                elif c == ";":
                    if col_kind == META:
                        raise ValueError("{}:{}:{} duplicated semi-colons"
                            .format(fname, cur_line, col))
                    col_kind = META
                    t_names = meta_indices
                    state = READ_START_ATOM
                elif c == ",":
                    state = READ_START_ATOM
                elif c == "#":
                    state = COMMENT
                elif c == "\\":
                    state = ESCAPE
                elif c == "\n" or f_eof:
                    state = BEGIN_LINE
                elif c == "\r":
                    state = CARRIAGE_RETURNED
                else:
                    state = READ_START_ATOM
                continue

            elif state == COMMENT:
                if c == "\n" or f_eof:
                    state = BEGIN_LINE
                elif c == "\r":
                    state = CARRIAGE_RETURNED

            elif state == CARRIAGE_RETURNED:
                if c != "\n" and not f_eof:
                    ungetc(c, f)
                state = BEGIN_LINE
    finally:
        fclose(f)

    if in_line == 0:
        line -= 1

    res = []
    for t_data, t_indices, t_indptr, ll in [
             (X_data, X_indices, X_indptr, attr_indices),
             (Y_data, Y_indices, Y_indptr, class_indices),
             (metas_data, metas_indices, metas_indptr, metas_indices)]:
        if t_indptr[line]:
            resize_if_needed(t_indptr, line + 1)
            resize_if_needed(t_indices, t_indptr[line])
            resize_if_needed(t_data, t_indptr[line])
            row_err, col = check_csr_matrix(t_indptr, t_indices, len(t_names))
            if row_err >= 0:
                for name, attr_col in t_names.items():
                    if col == attr_col:
                        break
                raise ValueError("Duplicate values of '{}' in row {}".
                                 format(name.decode("utf-8"), row_err + 1))
            mat = sp.csr_matrix((t_data, t_indices, t_indptr), (line, len(ll)))
            mat.sort_indices()
        else:
            mat = None
        res.append(mat)

    return tuple(res) + (attr_indices, class_indices, meta_indices)


# TODO: this function needs to be bottlenecked (not literally;
# adding 'char' and 'float32' should suffice)
