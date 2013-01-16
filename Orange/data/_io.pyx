cimport numpy as np
import cython

#from numpy cimport NPY_FLOAT64 as NPY_float64

from libc.stdio cimport *
from libc.string cimport *

cdef extern from "stdio.h" nogil:
    int fgetc(FILE *STREAM)
    int ungetc(char c, FILE *STREAM)

cdef extern from "stdlib.h" nogil:
    double atof(char *c)

cdef enum State:
    BEGIN_LINE, READ_START_ATOM, READ, QUOTED, END_QUOTED, END_ATOM,
    COMMENT, CARRIAGE_RETURNED, ESCAPE, EQUALS, END_LINE,
    SET_VALUE, WAIT_VALUE, READ_VALUE, READ_DECS

def sparse_prescan(fname):
    cdef:
        State state
        char c
        int ci
        char *not_in_atom = "#=,|;\t\n\r\x00"
        int col, line
        int in_line
        char atom[10240], *atomp, *atome, *endc
        int n_attributes = 0
        int n_classes = 0
        int n_metas = 0
        int n_lines = 0
        int *output_count
        char f_eof = 0

    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))

    atome = atom + 10240
    attributes = set()
    classes = set()
    metas = set()
    line = 0

    try:
        state = BEGIN_LINE
        while True:
            if state == BEGIN_LINE:
                output = attributes
                output_count = &n_attributes
                col = in_line = 0
                line += 1
                state = READ_START_ATOM

            if state != END_LINE:
                ci = fgetc(f)
                if ci == EOF:
                    f_eof = 1
                    c = "\x00"
                else:
                    c = <char>ci
                col += 1
                if col > 100:
                    return

            if state == READ_START_ATOM:
                atomp = atom
                if c == "," or c == " " or c == "\t":
                    continue
                elif c == '"':
                    state = QUOTED
                    continue
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
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError("{}:{}:{}: end of file in escape sequence"
                        .format(fname, line, col))
                else:
                    raise ValueError("{}:{}:{}: unrecognized escape sequence"
                        .format(fname, line, col))
                # fall through
                state = READ

            if state == READ or state == EQUALS:
                endc = strchr(not_in_atom, c)
                if endc == NULL:
                    if state == READ:
                        atomp[0] = c
                        atomp += 1
                        if atomp == atome:
                            raise ValueError("{}:{}:{}: value name too long"
                                .format(fname, line, col))
                else:
                    if state == READ and atomp != atom:
                        # strip whitespace on the right
                        while atomp != atom and (
                                atomp[-1] == " " or atomp[-1] == "\t"):
                            atomp -= 1
                        if atomp != atom:
                            atomp[0] = "\x00"
                            output.add(atom)
                            output_count[0] += 1
                            in_line += 1
                    if c == "|":
                        if output_count != &n_attributes:
                            raise ValueError(
                                "{}:{}:{}: classes should follow attributes"
                                .format(fname, line, col))
                        output = classes
                        output_count = &n_classes
                        state = READ_START_ATOM
                    elif c == ";":
                        if output_count == &n_metas:
                            raise ValueError("{}:{}:{} duplicated semi-colons"
                                .format(fname, line, col))
                        output = metas
                        output_count = &n_metas
                        state = READ_START_ATOM
                    elif c == "," or c == "\t":
                        state = READ_START_ATOM
                    elif c == "#":
                        state = COMMENT
                    elif c == "\\":
                        state = ESCAPE
                    elif c == "\n" or f_eof:
                        state = END_LINE
                    elif c == "\r":
                        state = CARRIAGE_RETURNED
                    elif c == "=":
                        if state == EQUALS:
                            raise ValueError("{}:{}:{}: invalid value"
                                .format(fname, line, col))
                        state = EQUALS

            elif state == QUOTED:
                if c == "\r" or c == "\n":
                    raise ValueError(
                        "{}:{}:{}: end of line within a quoted value"
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError(
                        "{}:{}:{}: end of file within a quoted value"
                        .format(fname, line, col))
                elif c == '"':
                    state = END_QUOTED
                else:
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, line, col))

            elif state == END_QUOTED:
                endc = strchr(not_in_atom, c)
                if endc == NULL:
                    raise ValueError("{}:{}:{}: quoted value should be "
                        "followed by value separator or end of line"
                        .format(fname, line, col))
                if atomp != atom:
                    atomp = "\x00"
                    output.add(atom)
                    output_count[0] += 1
                    in_line += 1
                    if c == "#":
                        state = COMMENT
                    elif c == "\n" or f_eof:
                        state = END_LINE
                    elif c == "\r":
                        state = CARRIAGE_RETURNED
                    elif c == "=":
                        state = EQUALS
                    else:
                        state = READ_START_ATOM

            elif state == COMMENT:
                if c == "\n" or f_eof:
                    state = END_LINE
                elif c == "\r":
                    state = CARRIAGE_RETURNED

            elif state == CARRIAGE_RETURNED:
                if c != "\n" and not f_eof:
                    ungetc(c, f)
                state = END_LINE

            elif state == END_LINE:
                if in_line:
                    n_lines += 1
                if f_eof:
                    return (attributes, classes, metas,
                            n_attributes, n_classes, n_metas, n_lines)
                state = BEGIN_LINE
    finally:
        fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_read_float(fname,
                      attr_indices, class_indices, meta_indices,
                      X, Y, metas):
    cdef:
        State state
        char c
        int ci
        char *not_in_atom = "#,|;\t\n\r\x00"
        int col, line, cur_line
        int in_line
        char atom[10240], *atomp, *eq_pos, *endc
        char *atome = atom + 10240
        char f_eof = 0
        int ii
        int attr_index
        float value, decs

        np.ndarray[np.float_t, ndim=1] X_data = X.data
        np.ndarray[int, ndim=1] X_indices = X.indices
        np.ndarray[int, ndim=1] X_indptr = X.indptr

        np.ndarray[np.float_t, ndim=1] Y_data = Y.data
        np.ndarray[int, ndim=1] Y_indices = Y.indices
        np.ndarray[int, ndim=1] Y_indptr = Y.indptr

        np.ndarray[np.float_t, ndim=1] metas_data = metas.data
        np.ndarray[int, ndim=1] metas_indices = metas.indices
        np.ndarray[int, ndim=1] metas_indptr = metas.indptr

        np.ndarray[np.float_t, ndim=1] t_data
        np.ndarray[int, ndim=1] t_indices
        np.ndarray[int, ndim=1] t_indptr


    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))

    X_indptr[0] = Y_indptr[0] = metas_indptr[0] = 0
    line = 0
    cur_line = 0
    in_line = 0

    try:
        state = BEGIN_LINE
        while not f_eof and state == BEGIN_LINE:
            if state == BEGIN_LINE:
                t_data = X_data
                t_indices = X_indices
                t_indptr = X_indptr
                t_names = attr_indices
                if in_line:
                    cur_line += 1
                col = in_line = 0
                line += 1
                X_indptr[line] = X_indptr[line - 1]
                Y_indptr[line] = Y_indptr[line - 1]
                metas_indptr[line] = metas_indptr[line - 1]
                state = READ_START_ATOM

            if state != END_LINE and state != SET_VALUE:
                ci = fgetc(f)
                if ci == EOF:
                    f_eof = 1
                    c = "\x00"
                else:
                    c = <char>ci
                col += 1
                if col > 100:
                    return

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
                        .format(fname, line, col))
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
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError("{}:{}:{}: end of file in escape sequence"
                        .format(fname, line, col))
                else:
                    raise ValueError("{}:{}:{}: unrecognized escape sequence"
                        .format(fname, line, col))
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
                            .format(fname, line, col))
                    continue
                else:
                    # fall through to END_ATOM
                    state = END_ATOM


            if state == QUOTED:
                if c == "\r" or c == "\n":
                    raise ValueError(
                        "{}:{}:{}: end of line within a quoted value"
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError(
                        "{}:{}:{}: end of file within a quoted value"
                        .format(fname, line, col))
                elif c != '"':
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, line, col))
                else:
                    state = END_QUOTED

            if state == END_QUOTED:
                if c == " " or c == "\t":
                    continue
                endc = strchr(not_in_atom, c)
                if endc == NULL and c != "=":
                    raise ValueError("{}:{}:{}: quoted value should be "
                        "followed by value separator or end of line"
                        .format(fname, line, col))
                # fall through
                state = END_ATOM

            if state == END_ATOM:
                if atomp == atom:
                    raise ValueError("{}:{}:{}: empty value name"
                        .format(fname, line, col))
                while atomp != atom and (
                        atomp[-1] == " " or atomp[-1] == "\t"):
                    atomp -= 1
                    attr_index = t_names[atom]
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
                if "0" < c < "9":
                    value = value * 10 + (c - "0")
                elif c == ".":
                    decs = 0.1
                    state = READ_DECS
                else:
                    endc = strchr(not_in_atom, c)
                    if endc != NULL:
                        state = SET_VALUE
                    else:
                        raise ValueError("{}:{}:{}: invalid value"
                            .format(fname, line, col))
                continue

            if state == READ_DECS:
                if "0" < c < "9":
                    value = value * decs + (c - "0")
                    decs /= 10
                else:
                    endc = strchr(not_in_atom, c)
                    if endc != NULL:
                        state = SET_VALUE
                    else:
                        raise ValueError("{}:{}:{}: invalid value"
                            .format(fname, line, col))
                continue

            if state == SET_VALUE:
                ii = t_indptr[line]
                t_data[ii] = value
                t_indices[ii] = attr_index
                t_indptr[line] += 1
                in_line += 1

                if c == "|":
                    if t_data != X_data:
                        raise ValueError(
                            "{}:{}:{}: classes should follow attributes"
                            .format(fname, line, col))
                    t_data = Y_data
                    t_indices = Y_indices
                    t_indptr = Y_indptr
                    t_names = class_indices
                    state = READ_START_ATOM
                elif c == ";":
                    if t_data == metas_data:
                        raise ValueError("{}:{}:{} duplicated semi-colons"
                            .format(fname, line, col))
                    t_data = metas_data
                    t_indices = metas_indices
                    t_indptr = metas_indptr
                    t_names = metas_indices
                    state = READ_START_ATOM
                elif c == "," or c == "\t":
                    state = READ_START_ATOM
                elif c == "#":
                    state = COMMENT
                elif c == "\\":
                    state = ESCAPE
                elif c == "\n" or f_eof:
                    state = BEGIN_LINE
                elif c == "\r":
                    state = CARRIAGE_RETURNED
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