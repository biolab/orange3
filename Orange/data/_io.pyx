import scipy.sparse as sp

from libc.stdio cimport fopen, fclose, fgetc, ungetc, EOF, FILE
from libc.string cimport strchr


cdef enum State:
    BEGIN_LINE, READ_START_ATOM, READ, QUOTED, END_QUOTED, END_ATOM,
    COMMENT, CARRIAGE_RETURNED, ESCAPE, EQUALS,
    SET_VALUE, WAIT_VALUE, READ_VALUE, READ_DECS, TO_NEXT

cdef enum ColKinds:
    ATTRIBUTE, CLASS, META


def sparse_read_float(fname):
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
        int attr_index
        float value, decs
        char col_kind

        list X_data = []
        list X_rows = []
        list X_cols = []

        list Y_data = []
        list Y_rows = []
        list Y_cols = []

        list M_data = []
        list M_rows = []
        list M_cols = []

        dict attr_indices = {}
        dict class_indices = {}
        dict meta_indices = {}

    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))

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
                    t_names = attr_indices
                cur_line += 1
                col = in_line = 0
                state = READ_START_ATOM

            if state != SET_VALUE:
                ci = fgetc(f)
                if ci == EOF:
                    f_eof = 1
                    c = b"\x00"
                else:
                    c = <char>ci
                col += 1

            if state == READ_START_ATOM:
                atomp = atom
                value = 1
                if c == b"," or c == b" " or c == b"\t":
                    continue
                elif c == b'"':
                    state = QUOTED
                    continue
                elif c == b"=":
                    raise ValueError("{}:{}:{}: missing value name"
                        .format(fname, cur_line, col))
                # fall through
                state = READ

            if state == ESCAPE:
                if c == b"t":    c = b"\t"
                elif c == b"n":    c = b"\n"
                elif c == b"r":    c = b"\r"
                elif c == b'"' or c == b"'" or c == b"\\" or c == b" ":
                    pass
                elif c == b"\r" or c == b"\n":
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
                if c == b"\\":
                    state = ESCAPE
                    continue
                endc = strchr(not_in_atom, c)
                if endc == NULL and c != b"=":
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
                if c == b"\r" or c == b"\n":
                    raise ValueError(
                        "{}:{}:{}: end of line within a quoted value"
                        .format(fname, cur_line, col))
                elif f_eof:
                    raise ValueError(
                        "{}:{}:{}: end of file within a quoted value"
                        .format(fname, cur_line, col))
                elif c != b'"':
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, cur_line, col))
                else:
                    state = END_QUOTED
                    continue

            if state == END_QUOTED:
                if c == b" " or c == b"\t":
                    continue
                endc = strchr(not_in_atom, c)
                if endc == NULL and c != b"=":
                    raise ValueError("{}:{}:{}: quoted value should be "
                        "followed by value separator or end of line"
                        .format(fname, cur_line, col))
                # fall through
                state = END_ATOM

            if state == END_ATOM:
                while atomp != atom and (
                        atomp[-1] == b" " or atomp[-1] == b"\t"):
                    atomp -= 1
                if atomp == atom:
                    if c == b"=":
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
                    if c == b"=":
                        value = 0
                        state = WAIT_VALUE
                        continue
                    # fall through
                    state = SET_VALUE

            if state == WAIT_VALUE:
                if c == b" " or c == b"\t":
                    continue
                else:
                    # fall through
                    state = READ_VALUE

            if state == READ_VALUE:
                if b"0" <= c <= b"9":
                    value = value * 10 + (c & 0xf)
                elif c == b".":
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
                if b"0" <= c <= b"9":
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
                    X_data.append(value)
                    X_rows.append(line - 1)
                    X_cols.append(attr_index)
                elif col_kind == CLASS:
                    Y_data.append(value)
                    Y_rows.append(line - 1)
                    Y_cols.append(attr_index)
                else:
                    M_data.append(value)
                    M_rows.append(line - 1)
                    M_cols.append(attr_index)
                in_line += 1
                state = TO_NEXT

            if state == TO_NEXT:
                if c == b"|":
                    if col_kind != ATTRIBUTE:
                        raise ValueError(
                            "{}:{}:{}: classes should follow attributes"
                            .format(fname, cur_line, col))
                    col_kind = CLASS
                    t_names = class_indices
                    state = READ_START_ATOM
                elif c == b";":
                    if col_kind == META:
                        raise ValueError("{}:{}:{} duplicated semi-colons"
                            .format(fname, cur_line, col))
                    col_kind = META
                    t_names = meta_indices
                    state = READ_START_ATOM
                elif c == b",":
                    state = READ_START_ATOM
                elif c == b"#":
                    state = COMMENT
                elif c == b"\\":
                    state = ESCAPE
                elif c == b"\n" or f_eof:
                    state = BEGIN_LINE
                elif c == b"\r":
                    state = CARRIAGE_RETURNED
                else:
                    state = READ_START_ATOM
                continue

            elif state == COMMENT:
                if c == b"\n" or f_eof:
                    state = BEGIN_LINE
                elif c == b"\r":
                    state = CARRIAGE_RETURNED

            elif state == CARRIAGE_RETURNED:
                if c != b"\n" and not f_eof:
                    ungetc(c, f)
                state = BEGIN_LINE
    finally:
        fclose(f)

    if in_line == 0:
        line -= 1

    res = []
    for t_data, t_rows, t_cols, ll in (
             (X_data, X_rows, X_cols, attr_indices),
             (Y_data, Y_rows, Y_cols, class_indices),
             (M_data, M_rows, M_cols, meta_indices)):
        if len(t_data):
            mat = sp.coo_matrix((t_data, (t_rows, t_cols)), (line, len(ll))).tocsr()
            mat.sort_indices()
        else:
            mat = None
        res.append(mat)

    return tuple(res) + (attr_indices, class_indices, meta_indices)


# TODO: this function needs to be bottlenecked (not literally;
# adding 'char' and 'float32' should suffice)
